"""Distilled two-stage generation for several clips in one process.

``python -m ltx_pipelines.distilled_batch --manifest scenes.json <distilled CLI args>``

The per-clip CLI (``ltx_pipelines.distilled``) reloads Gemma, rebuilds the 22B transformer
source and re-streams every weight for every clip. On an 8 GB card that fixed cost is most
of a clip's time: Stage 1 runs about 7 minutes whether the clip lasts 5 or 17 seconds.
Here the prompts are encoded with one Gemma load, the transformer is built once per stage,
and each denoising step streams its weights once for all clips (``utils/scene_batch.py``).

Each clip is planned, conditioned, noised, sampled and decoded exactly as
``DistilledPipeline`` does it for a single clip -- own seed, noise generator, ancestral
sampler seeds (LTX-2.5+), conditioning and token count -- so its output does not depend on
the batch.

Stage 1 results are written next to each output as ``<output>.stage1.pt``: a crash in
Stage 2 or decoding (this box sees intermittent access violations) then costs only the
remaining work on the next run, not the whole batch. The checkpoint carries a fingerprint
of the clip's inputs and is ignored when they changed.

Manifest format::

    {"scenes": [{"prompt": "...", "output_path": "...", "num_frames": 121, "seed": 42,
                 "images": [["start.png", 0, 1.0], ["end.png", 120, 1.0]]}]}

An image entry is ``[path, frame_idx, strength]`` plus an optional H.264 CRF; without one
the CRF the checkpoint was trained with is used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.video_vae import AUTO_TILING, TilingConfig, get_video_chunks_number
from ltx_core.types import VideoPixelShape
from ltx_pipelines.chunks import Chunk, ChunkConfig, decode_chunks, generate_uniform_chunks, pipeline_output_from_chunks
from ltx_pipelines.chunks.conditionings import assert_image_frames_in_clip, image_conditionings_for_chunk
from ltx_pipelines.distilled import (
    ANCESTRAL_ETA,
    ANCESTRAL_NOISE_SEED_OFFSET,
    ANCESTRAL_S_NOISE,
    ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET,
    DistilledPipeline,
    add_output_crf_arg,
)
from ltx_pipelines.utils.args import default_2_stage_distilled_arg_parser, resolve_cli_params, resolve_path
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cap_cuda_memory_to_free_vram,
    ensure_tiling_config,
    snap_frames_to_grid,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.scene_batch import (
    AncestralSceneSampler,
    EulerSceneSampler,
    SceneStageInput,
    scene_batched_stage,
)
from ltx_pipelines.utils.types import ImageConditioningInput, ModalitySpec, VideoAudio

logger = logging.getLogger(__name__)

STAGE_1_CHECKPOINT_SUFFIX = ".stage1.pt"
_MIB = 1024 * 1024


@dataclass(frozen=True)
class SceneJob:
    prompt: str
    output_path: str
    num_frames: int
    images: tuple[ImageConditioningInput, ...]
    seed: int = 42

    @classmethod
    def from_manifest(cls, item: dict) -> SceneJob:
        images = tuple(
            ImageConditioningInput(
                path=resolve_path(str(entry[0])),
                frame_idx=int(entry[1]),
                strength=float(entry[2]),
                crf=int(entry[3]) if len(entry) > 3 and entry[3] is not None else None,
            )
            for entry in item.get("images", [])
        )
        return cls(
            prompt=str(item["prompt"]),
            output_path=str(resolve_path(str(item["output_path"]))),
            num_frames=snap_frames_to_grid(int(item["num_frames"])),
            images=images,
            seed=int(item.get("seed", 42)),
        )

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.output_path + STAGE_1_CHECKPOINT_SUFFIX)


@dataclass
class _Stage1Result:
    """What Stage 2 and decoding need from Stage 1, plus the clip's RNG state."""

    upscaled_video_latent: torch.Tensor
    audio_latent: torch.Tensor
    generator_state: torch.Tensor


def _file_signature(path: str) -> list:
    try:
        stat = Path(path).stat()
        return [str(path), stat.st_size, int(stat.st_mtime)]
    except OSError:
        return [str(path), None, None]


def _fingerprint(job: SceneJob, *, width: int, height: int, frame_rate: float, model: str, ancestral: bool) -> str:
    payload = {
        "prompt": job.prompt,
        "seed": job.seed,
        "num_frames": job.num_frames,
        "images": [[*_file_signature(img.path), img.frame_idx, img.strength, img.crf] for img in job.images],
        "size": [width, height],
        "frame_rate": frame_rate,
        "model": Path(model).name,
        "sampler": "ancestral" if ancestral else "euler",
        "sigmas": [DISTILLED_SIGMAS.tolist(), STAGE_2_DISTILLED_SIGMAS.tolist()],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class SceneBatchDistilledPipeline(DistilledPipeline):
    """``DistilledPipeline`` that generates a list of clips with shared model passes."""

    def generate_batch(
        self,
        jobs: list[SceneJob],
        *,
        height: int,
        width: int,
        frame_rate: float,
        transformer_path: str,
        output_crf: int,
        gpu_state_budget_mb: int = 1024,
    ) -> list[str]:
        assert_resolution(height=height, width=width, is_two_stage=True)
        started_at = time.monotonic()
        jobs = [replace(job, images=tuple(self.image_conditioner.resolve_crf(job.images))) for job in jobs]
        for job in jobs:
            assert_image_frames_in_clip(job.images, job.num_frames)

        fingerprints = {
            job.output_path: _fingerprint(
                job,
                width=width,
                height=height,
                frame_rate=frame_rate,
                model=transformer_path,
                ancestral=self.use_ancestral_sampler,
            )
            for job in jobs
        }
        stage_1: dict[str, _Stage1Result] = {}
        for job in jobs:
            restored = self._load_stage_1(job, fingerprints[job.output_path])
            if restored is not None:
                stage_1[job.output_path] = restored
        pending = [job for job in jobs if job.output_path not in stage_1]
        logger.info(
            "[LTX scene batch] %s clips at %sx%s: %s need Stage 1, %s resume from a Stage 1 checkpoint",
            len(jobs),
            width,
            height,
            len(pending),
            len(jobs) - len(pending),
        )

        contexts = self._encode_prompts(jobs)
        cap_cuda_memory_to_free_vram()  # see DistilledPipeline._prepare_run
        chunks = {
            job.output_path: self._plan_chunk(job, contexts[job.output_path], width // 2, height // 2, frame_rate)
            for job in jobs
        }

        if pending:
            stage_1.update(self._run_stage_1(pending, chunks, frame_rate, gpu_state_budget_mb))
            for job in pending:
                self._save_stage_1(job, stage_1[job.output_path], fingerprints[job.output_path])

        refined, generators = self._run_stage_2(jobs, chunks, stage_1, frame_rate)

        outputs = []
        for job in jobs:
            clip_started = time.monotonic()
            self._decode_and_write(
                job, refined.pop(job.output_path), generators[job.output_path], height, width, frame_rate, output_crf
            )
            job.checkpoint_path.unlink(missing_ok=True)
            outputs.append(job.output_path)
            logger.info(
                "[LTX scene batch] Saved %s (%s frames) in %.1fs",
                job.output_path,
                job.num_frames,
                time.monotonic() - clip_started,
            )

        logger.info("[LTX scene batch] %s clips finished in %.1fs", len(jobs), time.monotonic() - started_at)
        return outputs

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _encode_prompts(self, jobs: list[SceneJob]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        unique_prompts = list(dict.fromkeys(job.prompt for job in jobs))
        logger.info("[LTX scene batch] Encoding %s prompts with one text-encoder load", len(unique_prompts))
        encoded = self.prompt_encoder.encode_each(unique_prompts)
        by_prompt = {
            prompt: (output.video_encoding, output.audio_encoding)
            for prompt, output in zip(unique_prompts, encoded, strict=True)
        }
        return {job.output_path: by_prompt[job.prompt] for job in jobs}

    def _plan_chunk(
        self, job: SceneJob, context: tuple[torch.Tensor, torch.Tensor], width: int, height: int, frame_rate: float
    ) -> Chunk:
        """The single full-clip chunk ``DistilledPipeline.stream_chunks`` plans for this clip."""
        target = VideoPixelShape(batch=1, frames=job.num_frames, height=height, width=width, fps=frame_rate)
        (chunk,) = generate_uniform_chunks(
            target=target,
            context=VideoAudio(video=context[0], audio=context[1]),
            device=self.device,
            dtype=self.dtype,
            config=ChunkConfig(chunk_pixel_frames=job.num_frames, next_video_carry_frames=0),
            video_scale_factors=self.stage.video_scale_factors,
        )
        return chunk

    def _conditionings(self, jobs: list[SceneJob], chunks: list[Chunk]) -> list[list]:
        """Each clip's image conditionings at its chunk's latent size, with one encoder load."""
        return self.image_conditioner(
            lambda enc: [
                image_conditionings_for_chunk(chunk, images=job.images, video_encoder=enc)
                for job, chunk in zip(jobs, chunks, strict=True)
            ]
        )

    def _sampler(self, seed: int, offset: int) -> EulerSceneSampler | AncestralSceneSampler:
        """What ``DistilledPipeline._sampler_kwargs`` selects for this clip and pass."""
        if not self.use_ancestral_sampler:
            return EulerSceneSampler()
        return AncestralSceneSampler(
            noise_seed=seed + offset, model_dtype=self.dtype, eta=ANCESTRAL_ETA, s_noise=ANCESTRAL_S_NOISE
        )

    @staticmethod
    def _stage_input(
        chunk: Chunk,
        conditionings: list,
        frame_rate: float,
        noise_scale: float,
        noiser: GaussianNoiser,
        sampler: EulerSceneSampler | AncestralSceneSampler,
    ) -> SceneStageInput:
        """The modality specs ``denoise_chunks`` builds for a chunk without carry."""
        return SceneStageInput(
            modalities=VideoAudio(
                video=ModalitySpec(
                    latent=chunk.video,
                    conditioning_fps=frame_rate,
                    context=chunk.video_context,
                    conditionings=list(conditionings),
                    noise_scale=noise_scale,
                ),
                audio=ModalitySpec(
                    latent=chunk.audio,
                    conditioning_fps=frame_rate,
                    context=chunk.audio_context,
                    conditionings=list(chunk.make_audio_conditionings(chunk)),
                    noise_scale=noise_scale,
                ),
            ),
            noiser=noiser,
            sampler=sampler,
        )

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _run_stage_1(
        self, jobs: list[SceneJob], chunks: dict[str, Chunk], frame_rate: float, gpu_state_budget_mb: int
    ) -> dict[str, _Stage1Result]:
        stage_started = time.monotonic()
        job_chunks = [chunks[job.output_path] for job in jobs]
        conditionings = self._conditionings(jobs, job_chunks)
        generators = {}
        inputs = []
        for job, chunk, job_conditionings in zip(jobs, job_chunks, conditionings, strict=True):
            # Same RNG use as DistilledPipeline: one generator per clip for the initial
            # noise of both stages and the decoder; the ancestral sampler seeds its own.
            generator = torch.Generator(device=self.device).manual_seed(job.seed)
            generators[job.output_path] = generator
            inputs.append(
                self._stage_input(
                    chunk,
                    job_conditionings,
                    frame_rate,
                    1.0,
                    GaussianNoiser(generator=generator),
                    self._sampler(job.seed, ANCESTRAL_NOISE_SEED_OFFSET),
                )
            )
        del conditionings

        logger.info("[LTX scene batch] Stage 1 denoising: %s clips", len(jobs))
        sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device)
        states = scene_batched_stage(self.stage, inputs, sigmas, gpu_state_budget_mb * _MIB)
        del inputs

        logger.info("[LTX scene batch] Stage 2 latent upsampling for %s clips", len(jobs))
        upscaled = self.upsampler.map([video_state.latent[:1] for video_state, _ in states])
        results = {}
        for job, (_, audio_state), video_latent in zip(jobs, states, upscaled, strict=True):
            results[job.output_path] = _Stage1Result(
                upscaled_video_latent=video_latent,
                audio_latent=audio_state.latent,
                generator_state=generators[job.output_path].get_state(),
            )
        logger.info("[LTX scene batch] Stage 1 done for %s clips in %.1fs", len(jobs), time.monotonic() - stage_started)
        return results

    def _run_stage_2(
        self,
        jobs: list[SceneJob],
        chunks: dict[str, Chunk],
        stage_1: dict[str, _Stage1Result],
        frame_rate: float,
    ) -> tuple[dict[str, Chunk], dict[str, torch.Generator]]:
        stage_started = time.monotonic()
        sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device)
        upscaled_chunks = []
        generators = {}
        for job in jobs:
            result = stage_1[job.output_path]
            upscaled_chunks.append(
                replace(
                    chunks[job.output_path],
                    video=result.upscaled_video_latent.to(self.device),
                    audio=result.audio_latent.to(self.device),
                )
            )
            generator = torch.Generator(device=self.device)
            generator.set_state(result.generator_state)
            generators[job.output_path] = generator
        conditionings = self._conditionings(jobs, upscaled_chunks)
        inputs = [
            self._stage_input(
                chunk,
                job_conditionings,
                frame_rate,
                sigmas[0].item(),
                GaussianNoiser(generator=generators[job.output_path]),
                self._sampler(job.seed, ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET),
            )
            for job, chunk, job_conditionings in zip(jobs, upscaled_chunks, conditionings, strict=True)
        ]
        del conditionings

        logger.info("[LTX scene batch] Stage 2 denoising: %s clips", len(jobs))
        # Stage 2 hidden states are 4x larger and the card has no headroom left at full
        # resolution: park every clip in host RAM between blocks.
        states = scene_batched_stage(self.stage, inputs, sigmas, gpu_state_budget_bytes=0)
        del inputs
        refined = {
            job.output_path: replace(chunk, video=video_state.latent, audio=audio_state.latent)
            for job, chunk, (video_state, audio_state) in zip(jobs, upscaled_chunks, states, strict=True)
        }
        logger.info("[LTX scene batch] Stage 2 done for %s clips in %.1fs", len(jobs), time.monotonic() - stage_started)
        return refined, generators

    def _decode_and_write(
        self,
        job: SceneJob,
        chunk: Chunk,
        generator: torch.Generator,
        height: int,
        width: int,
        frame_rate: float,
        output_crf: int,
    ) -> None:
        tiling_config: TilingConfig = ensure_tiling_config(
            AUTO_TILING,
            scale_factors=tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path),
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(batch=1, frames=job.num_frames, height=height, width=width, fps=frame_rate),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
            keyframes=False,
        )
        result = pipeline_output_from_chunks(
            decode_chunks(
                iter((chunk,)),
                self.video_decoder,
                self.audio_decoder,
                fps=frame_rate,
                tiling_config=tiling_config,
                generator=generator,
                dtype=self.dtype,
                keyframes=False,
            ),
            num_frames=job.num_frames,
            tiling_config=tiling_config,
        )
        output = Path(job.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Written under a temporary name and renamed when complete: a rerun after a crash
        # skips clips whose output exists, so a half-written one must never carry the name.
        partial = output.with_name(output.stem + ".partial" + output.suffix)
        encode_video(
            video=result.video,
            fps=frame_rate,
            audio=result.audio,
            output_path=str(partial),
            video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
            crf=output_crf,
        )
        partial.replace(output)

    # ------------------------------------------------------------------
    # Stage 1 checkpoints
    # ------------------------------------------------------------------

    @staticmethod
    def _save_stage_1(job: SceneJob, result: _Stage1Result, fingerprint: str) -> None:
        payload = {
            "fingerprint": fingerprint,
            "upscaled_video_latent": result.upscaled_video_latent.detach().to("cpu"),
            "audio_latent": result.audio_latent.detach().to("cpu"),
            "generator_state": result.generator_state.detach().to("cpu"),
        }
        path = job.checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)

    def _load_stage_1(self, job: SceneJob, fingerprint: str) -> _Stage1Result | None:
        path = job.checkpoint_path
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # a half-written or foreign file must not stop the batch
            logger.warning("[LTX scene batch] Ignoring unreadable Stage 1 checkpoint %s: %s", path, exc)
            return None
        if payload.get("fingerprint") != fingerprint or payload.get("audio_latent") is None:
            logger.info("[LTX scene batch] Stage 1 checkpoint %s is for different inputs; regenerating", path)
            return None
        logger.info("[LTX scene batch] Resuming %s from its Stage 1 checkpoint", job.output_path)
        return _Stage1Result(
            upscaled_video_latent=payload["upscaled_video_latent"],
            audio_latent=payload["audio_latent"],
            generator_state=payload["generator_state"],
        )


def batch_arg_parser(params: object) -> object:
    parser = add_output_crf_arg(default_2_stage_distilled_arg_parser(params=params))
    for action in parser._actions:
        if action.dest in ("prompt", "output_path"):
            action.required = False
    parser.add_argument(
        "--manifest",
        type=resolve_path,
        required=True,
        help="JSON file with the clips to generate (see module docstring).",
    )
    parser.add_argument(
        "--scene-state-gpu-mb",
        type=int,
        default=1024,
        help="VRAM for Stage 1 hidden states kept between blocks; clips beyond it wait in host RAM.",
    )
    return parser


def load_jobs(manifest_path: str) -> list[SceneJob]:
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    jobs = [SceneJob.from_manifest(item) for item in manifest.get("scenes", [])]
    if not jobs:
        raise ValueError(f"Manifest {manifest_path} lists no scenes")
    outputs = [job.output_path for job in jobs]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Manifest scenes must have distinct output paths")
    return jobs


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)
    started_at = time.monotonic()
    params = resolve_cli_params(distilled=True)
    args = batch_arg_parser(params).parse_args()

    jobs = load_jobs(args.manifest)
    # Outputs only ever appear complete (see _decode_and_write), so one that exists was
    # finished by an earlier attempt of this batch that crashed later; the caller removes
    # stale outputs before the first attempt.
    done = [job for job in jobs if Path(job.output_path).exists()]
    if done:
        logger.info("[LTX scene batch] %s clips already finished by an earlier attempt", len(done))
        jobs = [job for job in jobs if job not in done]
    if not jobs:
        logger.info("[LTX scene batch] Nothing left to generate")
        return

    pipeline = SceneBatchDistilledPipeline(
        model_paths=args.model_paths,
        spatial_upsampler_path=args.spatial_upsampler_path,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        diffvae_optimization=args.diffvae_optimization,
    )
    pipeline.generate_batch(
        jobs,
        height=args.height,
        width=args.width,
        frame_rate=args.frame_rate,
        transformer_path=args.model_paths.transformer(),
        output_crf=args.output_crf,
        gpu_state_budget_mb=args.scene_state_gpu_mb,
    )
    logger.info("[LTX scene batch] Finished in %.1fs", time.monotonic() - started_at)


if __name__ == "__main__":
    main()
