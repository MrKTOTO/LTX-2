"""The scene-batch CLI's pipeline must produce, per clip, the latents DistilledPipeline produces.

Both pipelines run on a tiny transformer with stand-ins for the text encoder, the image
encoder, the latent upsampler and the decoders, so the test covers what the batch pipeline
wires together itself: chunk planning, conditionings, noise scales, the per-clip generator
carried from Stage 1 into Stage 2 (also through a Stage 1 checkpoint), and the ancestral
sampler seeds of each pass.
"""

from types import SimpleNamespace

import pytest
import torch
from _tiny_model import AUDIO_HEAD_DIM, AUDIO_HEADS, CONFIGS, VIDEO_HEAD_DIM, VIDEO_HEADS
from test_scene_batch import _TinyBuilder

from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ltx_pipelines import distilled, distilled_batch
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.distilled_batch import SceneBatchDistilledPipeline, SceneJob
from ltx_pipelines.utils.blocks import DiffusionStage
from ltx_pipelines.utils.types import ImageConditioningInput

CPU = torch.device("cpu")
LATENT_CHANNELS = 128  # what the pipeline allocates for video, and audio channels x mel bins
HEIGHT = WIDTH = 64
FPS = 24.0


class _Tiny128Builder(_TinyBuilder):
    def build(self, device: torch.device | None = None, **_kwargs: object) -> torch.nn.Module:
        torch.manual_seed(0)
        model = LTXModel(
            model_type=LTXModelType.AudioVideo,
            num_attention_heads=VIDEO_HEADS,
            attention_head_dim=VIDEO_HEAD_DIM,
            in_channels=LATENT_CHANNELS,
            out_channels=LATENT_CHANNELS,
            num_layers=2,
            cross_attention_dim=VIDEO_HEADS * VIDEO_HEAD_DIM,
            audio_num_attention_heads=AUDIO_HEADS,
            audio_attention_head_dim=AUDIO_HEAD_DIM,
            audio_in_channels=LATENT_CHANNELS,
            audio_out_channels=LATENT_CHANNELS,
            audio_cross_attention_dim=AUDIO_HEADS * AUDIO_HEAD_DIM,
            **CONFIGS["ltx25"],
        )
        with torch.no_grad():
            for param in model.parameters():
                param.normal_(std=0.02)
        # DistilledPipeline plans its latents in bfloat16 regardless of self.dtype.
        return model.to(device=device or CPU, dtype=torch.bfloat16)


def _prompt_output(prompt: str) -> EmbeddingsProcessorOutput:
    gen = torch.Generator().manual_seed(sum(map(ord, prompt)))
    return EmbeddingsProcessorOutput(
        video_encoding=torch.randn(1, 6, VIDEO_HEADS * VIDEO_HEAD_DIM, generator=gen).to(torch.bfloat16),
        audio_encoding=torch.randn(1, 6, AUDIO_HEADS * AUDIO_HEAD_DIM, generator=gen).to(torch.bfloat16),
        attention_mask=torch.ones(1, 6),
    )


class _PromptEncoder:
    def __call__(self, prompts: list[str], **_kwargs: object) -> list[EmbeddingsProcessorOutput]:
        return [_prompt_output(prompt) for prompt in prompts]

    def encode_each(self, prompts: list[str]) -> list[EmbeddingsProcessorOutput]:
        return [_prompt_output(prompt) for prompt in prompts]


class _ImageConditioner:
    def __call__(self, fn: object) -> object:
        return fn(None)

    def resolve_crf(self, images: object) -> list:
        return [image if image.crf is not None else image._replace(crf=18) for image in images]


class _Upsampler:
    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        return self.map([latent])[0]

    def map(self, latents: list[torch.Tensor]) -> list[torch.Tensor]:
        return [latent.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2) * 0.9 for latent in latents]


def _fake_image_conditionings(chunk: object, *, images: object, **_kwargs: object) -> list:
    """A latent keyframe per image, at the chunk's latent size, as the real encoder would give."""
    items = []
    for image in images:
        gen = torch.Generator().manual_seed(sum(map(ord, image.path)) + image.frame_idx)
        latent = torch.randn(1, LATENT_CHANNELS, 1, *chunk.video.shape[-2:], generator=gen).to(chunk.video.dtype)
        latent_idx = 0 if image.frame_idx == 0 else chunk.video.shape[2] - 1
        items.append(VideoConditionByLatentIndex(latent=latent, strength=image.strength, latent_idx=latent_idx))
    return items


def _wire(pipeline: DistilledPipeline, ancestral: bool) -> DistilledPipeline:
    pipeline.device = CPU
    pipeline.dtype = torch.bfloat16
    pipeline.prompt_encoder = _PromptEncoder()
    pipeline.image_conditioner = _ImageConditioner()
    pipeline.stage = DiffusionStage(_Tiny128Builder(), torch.bfloat16, CPU)
    pipeline.upsampler = _Upsampler()
    pipeline.video_decoder = SimpleNamespace(checkpoint_path="vae", diffvae_optimization=None)
    pipeline.audio_decoder = None
    pipeline.duration_predictor = None
    pipeline.use_ancestral_sampler = ancestral
    return pipeline


JOBS = [
    SceneJob(
        prompt="a lighthouse at dusk",
        output_path="clip_a.mp4",
        num_frames=17,
        images=(ImageConditioningInput("start_a.png", 0, 1.0), ImageConditioningInput("end_a.png", 16, 1.0)),
        seed=42,
    ),
    SceneJob(
        prompt="rain on a window",
        output_path="clip_b.mp4",
        num_frames=25,
        images=(ImageConditioningInput("start_b.png", 0, 1.0),),
        seed=7,
    ),
]


@pytest.mark.parametrize("ancestral", [True, False], ids=["ltx25_ancestral", "euler"])
@pytest.mark.parametrize("via_checkpoint", [False, True], ids=["one_run", "resumed_stage_1"])
def test_batch_pipeline_matches_distilled_pipeline(
    ancestral: bool, via_checkpoint: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    monkeypatch.setenv("LTX_BLOCK_CHUNK_TOKENS", "5")
    for module in (distilled, distilled_batch):
        monkeypatch.setattr(module, "image_conditionings_for_chunk", _fake_image_conditionings)
    monkeypatch.setattr(distilled, "tiling_scale_factors_for_vae", lambda _path: None)
    monkeypatch.setattr(distilled, "ensure_tiling_config", lambda *_args, **_kwargs: None)
    captured: list = []
    monkeypatch.setattr(distilled, "decode_chunks", lambda chunks, *_a, **_k: captured.extend(chunks) or iter(()))

    reference = _wire(object.__new__(DistilledPipeline), ancestral)
    batch = _wire(object.__new__(SceneBatchDistilledPipeline), ancestral)
    jobs = [job.__class__(**{**job.__dict__, "output_path": str(tmp_path / job.output_path)}) for job in JOBS]

    with torch.inference_mode():
        expected = []
        for job in jobs:
            captured.clear()
            reference.stream_chunks(
                prompt=job.prompt,
                seed=job.seed,
                height=HEIGHT,
                width=WIDTH,
                frame_rate=FPS,
                images=list(job.images),
                num_frames=job.num_frames,
            )
            (chunk,) = captured
            expected.append((chunk.video, chunk.audio))

        jobs = [
            job.__class__(**{**job.__dict__, "images": tuple(batch.image_conditioner.resolve_crf(job.images))})
            for job in jobs
        ]
        contexts = batch._encode_prompts(jobs)
        chunks = {
            job.output_path: batch._plan_chunk(job, contexts[job.output_path], WIDTH // 2, HEIGHT // 2, FPS)
            for job in jobs
        }
        stage_1 = batch._run_stage_1(jobs, chunks, FPS, gpu_state_budget_mb=0)
        if via_checkpoint:
            for job in jobs:
                batch._save_stage_1(job, stage_1[job.output_path], "fp")
            stage_1 = {job.output_path: batch._load_stage_1(job, "fp") for job in jobs}
        refined, _generators = batch._run_stage_2(jobs, chunks, stage_1, FPS)

    for job, (exp_video, exp_audio) in zip(jobs, expected, strict=True):
        got = refined[job.output_path]
        torch.testing.assert_close(got.video, exp_video, rtol=0, atol=0)
        torch.testing.assert_close(got.audio, exp_audio, rtol=0, atol=0)
