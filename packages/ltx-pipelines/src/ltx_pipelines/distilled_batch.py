"""Distilled two-stage generation for several clips in one process.

``python -m ltx_pipelines.distilled_batch --manifest scenes.json <distilled CLI args>``

The per-clip CLI (``ltx_pipelines.distilled``) reloads Gemma, rebuilds the 22B
transformer source and re-streams every weight for every clip. On an 8 GB card that
fixed cost is most of a clip's time: Stage 1 runs about 7 minutes whether the clip
lasts 5 or 17 seconds. Here the prompts are encoded with one Gemma load, the
transformer is built once per stage, and each denoising step streams its weights once
for all clips (``utils/scene_batch.py``).

Each clip uses its own seed, noise generator, conditioning and token count, exactly as
the per-clip pipeline does, so the output of a clip does not depend on the batch.

Stage 1 results are written next to each output as ``<output>.stage1.pt``: a crash in
Stage 2 or decoding (this box sees intermittent access violations) then costs only the
remaining work on the next run, not the whole batch. The checkpoint carries a
fingerprint of the clip's inputs and is ignored when they changed.

Manifest format::

    {"scenes": [{"prompt": "...", "output_path": "...", "num_frames": 121, "seed": 42,
                 "images": [["start.png", 0, 1.0, 0], ["end.png", 120, 1.0, 0]]}]}
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
from ltx_pipelines.distilled import DistilledPipeline, build_pipeline_from_args, parse_devices
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    default_2_stage_distilled_arg_parser,
    detect_checkpoint_path,
    resolve_path,
)
from ltx_pipelines.utils.blocks import _build_state
from ltx_pipelines.utils.constants import (
    DEFAULT_IMAGE_CRF,
    DISTILLED_SIGMAS,
    STAGE_2_DISTILLED_SIGMAS,
    detect_params,
)
from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.scene_batch import scene_batched_euler_loop
from ltx_pipelines.utils.types import ModalitySpec

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
                crf=int(entry[3]) if len(entry) > 3 else DEFAULT_IMAGE_CRF,
            )
            for entry in item.get("images", [])
        )
        return cls(
            prompt=str(item["prompt"]),
            output_path=str(resolve_path(str(item["output_path"]))),
            num_frames=int(item["num_frames"]),
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
    audio_latent: torch.Tensor | None
    generator_state: torch.Tensor


def _file_signature(path: str) -> list:
    try:
        stat = Path(path).stat()
        return [str(path), stat.st_size, int(stat.st_mtime)]
    except OSError:
        return [str(path), None, None]


def _fingerprint(
    job: SceneJob, *, width: int, height: int, frame_rate: float, generate_audio: bool, checkpoint: str
) -> str:
    payload = {
        "prompt": job.prompt,
        "seed": job.seed,
        "num_frames": job.num_frames,
        "images": [[*_file_signature(img.path), img.frame_idx, img.strength, img.crf] for img in job.images],
        "size": [width, height],
        "frame_rate": frame_rate,
        "audio": generate_audio,
        "checkpoint": Path(checkpoint).name,
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
        generate_audio: bool = True,
        output_crf: int | None = None,
        gpu_state_budget_mb: int = 1024,
    ) -> list[str]:
        assert_resolution(height=height, width=width, is_two_stage=True)
        if self.stage_2_device != self.device:
            raise ValueError("Scene batching needs Stage 2 on the primary device; use the per-clip CLI instead")

        started_at = time.monotonic()
        dtype = torch.bfloat16
        fingerprints = {
            job.output_path: _fingerprint(
                job,
                width=width,
                height=height,
                frame_rate=frame_rate,
                generate_audio=generate_audio,
                checkpoint=self.distilled_checkpoint_path,
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

        contexts = self._encode_prompts(jobs, generate_audio)

        if pending:
            stage_1.update(
                self._run_stage_1(
                    pending,
                    contexts,
                    height=height,
                    width=width,
                    frame_rate=frame_rate,
                    generate_audio=generate_audio,
                    dtype=dtype,
                    gpu_state_budget_mb=gpu_state_budget_mb,
                )
            )
            for job in pending:
                self._save_stage_1(job, stage_1[job.output_path], fingerprints[job.output_path])

        final_latents, generators = self._run_stage_2(
            jobs,
            contexts,
            stage_1,
            height=height,
            width=width,
            frame_rate=frame_rate,
            dtype=dtype,
        )

        outputs = []
        tiling_config = TilingConfig.default()
        for job in jobs:
            clip_started = time.monotonic()
            result = stage_1[job.output_path]
            decoded_video = self.video_decoder(
                final_latents.pop(job.output_path), tiling_config, generators[job.output_path]
            )
            decoded_audio = (
                self.audio_decoder(result.audio_latent.to(self.device))
                if generate_audio and result.audio_latent is not None
                else None
            )
            output = Path(job.output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            # Written under a temporary name and renamed when complete: a rerun after a
            # crash skips clips whose output exists, so a half-written one must never
            # carry the final name.
            partial = output.with_name(output.stem + ".partial" + output.suffix)
            encode_video(
                video=decoded_video,
                fps=frame_rate,
                audio=decoded_audio,
                output_path=str(partial),
                video_chunks_number=get_video_chunks_number(job.num_frames, tiling_config),
                crf=output_crf,
            )
            partial.replace(output)
            del decoded_video, decoded_audio
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
    # Stages
    # ------------------------------------------------------------------

    def _encode_prompts(
        self, jobs: list[SceneJob], generate_audio: bool
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
        unique_prompts = list(dict.fromkeys(job.prompt for job in jobs))
        logger.info("[LTX scene batch] Encoding %s prompts with one text-encoder load", len(unique_prompts))
        encoded = self.prompt_encoder(unique_prompts)
        by_prompt = {}
        for prompt, output in zip(unique_prompts, encoded, strict=True):
            video_context = output.video_encoding.to(self.device)
            audio_context = output.audio_encoding.to(self.device) if generate_audio else None
            by_prompt[prompt] = (video_context, audio_context)
        return {job.output_path: by_prompt[job.prompt] for job in jobs}

    def _run_stage_1(
        self,
        jobs: list[SceneJob],
        contexts: dict[str, tuple[torch.Tensor, torch.Tensor | None]],
        *,
        height: int,
        width: int,
        frame_rate: float,
        generate_audio: bool,
        dtype: torch.dtype,
        gpu_state_budget_mb: int,
    ) -> dict[str, _Stage1Result]:
        stage_started = time.monotonic()
        stage_1_w, stage_1_h = width // 2, height // 2
        conditionings = self.image_conditioner(
            lambda enc: [
                combined_image_conditionings(
                    images=list(job.images),
                    height=stage_1_h,
                    width=stage_1_w,
                    video_encoder=enc,
                    dtype=dtype,
                    device=self.device,
                )
                for job in jobs
            ]
        )

        generators = {}
        states = []
        tools = []
        scene_contexts = []
        for job, job_conditionings in zip(jobs, conditionings, strict=True):
            # Same RNG consumption order as DistilledPipeline: video noise, then audio
            # noise, then Stage 2 and the decoder continue the same generator.
            generator = torch.Generator(device=self.device).manual_seed(job.seed)
            generators[job.output_path] = generator
            noiser = GaussianNoiser(generator=generator)
            pixel_shape = VideoPixelShape(
                batch=1, frames=job.num_frames, height=stage_1_h, width=stage_1_w, fps=frame_rate
            )
            video_tools = VideoLatentTools(
                VideoLatentPatchifier(patch_size=1), VideoLatentShape.from_pixel_shape(pixel_shape), frame_rate
            )
            video_context, audio_context = contexts[job.output_path]
            video_state = _build_state(
                ModalitySpec(context=video_context, conditionings=job_conditionings),
                video_tools,
                noiser,
                dtype,
                self.device,
            )
            audio_tools = None
            audio_state = None
            if generate_audio:
                audio_tools = AudioLatentTools(
                    AudioPatchifier(patch_size=1), AudioLatentShape.from_video_pixel_shape(pixel_shape)
                )
                audio_state = _build_state(ModalitySpec(context=audio_context), audio_tools, noiser, dtype, self.device)
            states.append((video_state, audio_state))
            tools.append((video_tools, audio_tools))
            scene_contexts.append((video_context, audio_context))
        del conditionings

        logger.info("[LTX scene batch] Stage 1 denoising: %s clips at %sx%s", len(jobs), stage_1_w, stage_1_h)
        sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device)
        with self.stage.model_context() as transformer:
            states = scene_batched_euler_loop(
                transformer,
                sigmas,
                states,
                scene_contexts,
                stepper=EulerDiffusionStep(),
                device=self.device,
                gpu_state_budget_bytes=gpu_state_budget_mb * _MIB,
            )

        finished = []
        for (video_state, audio_state), (video_tools, audio_tools) in zip(states, tools, strict=True):
            video_out = video_tools.unpatchify(video_tools.clear_conditioning(video_state))
            audio_latent = None
            if audio_state is not None:
                audio_latent = audio_tools.unpatchify(audio_tools.clear_conditioning(audio_state)).latent
            finished.append((video_out.latent[:1], audio_latent))
        del states
        self._empty_cuda_cache(self.device)

        logger.info("[LTX scene batch] Stage 2 latent upsampling for %s clips", len(jobs))
        results = {}
        upscaled = self.upsampler_many([video_latent for video_latent, _ in finished])
        for job, (_, audio_latent), video_latent in zip(jobs, finished, upscaled, strict=True):
            results[job.output_path] = _Stage1Result(
                upscaled_video_latent=video_latent,
                audio_latent=audio_latent,
                generator_state=generators[job.output_path].get_state(),
            )
        logger.info(
            "[LTX scene batch] Stage 1 done for %s clips in %.1fs", len(jobs), time.monotonic() - stage_started
        )
        return results

    def _run_stage_2(
        self,
        jobs: list[SceneJob],
        contexts: dict[str, tuple[torch.Tensor, torch.Tensor | None]],
        stage_1: dict[str, _Stage1Result],
        *,
        height: int,
        width: int,
        frame_rate: float,
        dtype: torch.dtype,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Generator]]:
        stage_started = time.monotonic()
        sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device)
        conditionings = self.image_conditioner(
            lambda enc: [
                combined_image_conditionings(
                    images=list(job.images),
                    height=height,
                    width=width,
                    video_encoder=enc,
                    dtype=dtype,
                    device=self.device,
                )
                for job in jobs
            ]
        )

        generators = {}
        states = []
        tools = []
        scene_contexts = []
        for job, job_conditionings in zip(jobs, conditionings, strict=True):
            result = stage_1[job.output_path]
            generator = torch.Generator(device=self.device)
            generator.set_state(result.generator_state)
            generators[job.output_path] = generator
            pixel_shape = VideoPixelShape(
                batch=1, frames=job.num_frames, height=height, width=width, fps=frame_rate
            )
            video_tools = VideoLatentTools(
                VideoLatentPatchifier(patch_size=1), VideoLatentShape.from_pixel_shape(pixel_shape), frame_rate
            )
            video_context, _ = contexts[job.output_path]
            video_state = _build_state(
                ModalitySpec(
                    context=video_context,
                    conditionings=job_conditionings,
                    noise_scale=sigmas[0].item(),
                    initial_latent=result.upscaled_video_latent.to(self.device),
                ),
                video_tools,
                GaussianNoiser(generator=generator),
                dtype,
                self.device,
            )
            states.append((video_state, None))
            tools.append(video_tools)
            scene_contexts.append((video_context, None))
        del conditionings

        logger.info("[LTX scene batch] Stage 2 denoising: %s clips at %sx%s", len(jobs), width, height)
        with self.stage.model_context() as transformer:
            # Stage 2 hidden states are 4x larger and the card has no headroom left at
            # full resolution: park every clip in host RAM between blocks.
            states = scene_batched_euler_loop(
                transformer,
                sigmas,
                states,
                scene_contexts,
                stepper=EulerDiffusionStep(),
                device=self.device,
                gpu_state_budget_bytes=0,
            )

        latents = {}
        for job, (video_state, _), video_tools in zip(jobs, states, tools, strict=True):
            latents[job.output_path] = video_tools.unpatchify(video_tools.clear_conditioning(video_state)).latent
        del states
        self._empty_cuda_cache(self.device)
        logger.info(
            "[LTX scene batch] Stage 2 done for %s clips in %.1fs", len(jobs), time.monotonic() - stage_started
        )
        return latents, generators

    def upsampler_many(self, latents: list[torch.Tensor]) -> list[torch.Tensor]:
        """Spatially upsample several Stage 1 latents with one encoder/upsampler load."""
        return self.upsampler.map(latents)

    # ------------------------------------------------------------------
    # Stage 1 checkpoints
    # ------------------------------------------------------------------

    @staticmethod
    def _save_stage_1(job: SceneJob, result: _Stage1Result, fingerprint: str) -> None:
        payload = {
            "fingerprint": fingerprint,
            "upscaled_video_latent": result.upscaled_video_latent.detach().to("cpu"),
            "audio_latent": result.audio_latent.detach().to("cpu") if result.audio_latent is not None else None,
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
        if payload.get("fingerprint") != fingerprint:
            logger.info("[LTX scene batch] Stage 1 checkpoint %s is for different inputs; regenerating", path)
            return None
        logger.info("[LTX scene batch] Resuming %s from its Stage 1 checkpoint", job.output_path)
        return _Stage1Result(
            upscaled_video_latent=payload["upscaled_video_latent"],
            audio_latent=payload.get("audio_latent"),
            generator_state=payload["generator_state"],
        )


def batch_arg_parser(params: object) -> object:
    parser = default_2_stage_distilled_arg_parser(params=params)
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
    logging.getLogger().setLevel(logging.INFO)
    started_at = time.monotonic()
    checkpoint_path = detect_checkpoint_path(distilled=True)
    params = detect_params(checkpoint_path)
    args = batch_arg_parser(params).parse_args()
    if args.tiled_denoising or parse_devices(args.tiled_devices):
        raise SystemExit("Scene batching does not support tiled denoising; use ltx_pipelines.distilled per clip")

    jobs = load_jobs(args.manifest)
    # Outputs only ever appear complete (see generate_batch), so one that exists was
    # finished by an earlier attempt of this batch that crashed later; the caller
    # removes stale outputs before the first attempt.
    done = [job for job in jobs if Path(job.output_path).exists()]
    if done:
        logger.info("[LTX scene batch] %s clips already finished by an earlier attempt", len(done))
        jobs = [job for job in jobs if job not in done]
    if not jobs:
        logger.info("[LTX scene batch] Nothing left to generate")
        return
    threshold = int(args.stage_2_tiled_threshold_frames or 0)
    if threshold > 0:
        too_long = [job.output_path for job in jobs if job.num_frames > threshold]
        if too_long:
            raise SystemExit(
                f"{len(too_long)} clips exceed the Stage 2 tiling threshold ({threshold} frames) and need the "
                f"per-clip CLI: {', '.join(too_long)}"
            )

    pipeline = build_pipeline_from_args(args, SceneBatchDistilledPipeline)
    pipeline.generate_batch(
        jobs,
        height=args.height,
        width=args.width,
        frame_rate=args.frame_rate,
        generate_audio=not args.disable_audio,
        output_crf=args.output_crf,
        gpu_state_budget_mb=args.scene_state_gpu_mb,
    )
    logger.info("[LTX scene batch] Finished in %.1fs", time.monotonic() - started_at)


if __name__ == "__main__":
    main()
