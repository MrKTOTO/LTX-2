import functools
from pathlib import Path

import safetensors
import torch
from transformers import AutoImageProcessor, Gemma3ForConditionalGeneration, Gemma3Processor

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_core.utils import find_matching_file


class GemmaTextEncoder(torch.nn.Module):
    """Pure Gemma text encoder — runs the LLM and returns raw hidden states.
    Prompt enhancement (generate) is also supported since the full
    Gemma3ForConditionalGeneration model (including lm_head) is loaded.
    """

    def __init__(
        self,
        model: Gemma3ForConditionalGeneration | None = None,
        tokenizer: LTXVGemmaTokenizer | None = None,
        processor: Gemma3Processor | None = None,
        dtype: torch.dtype = torch.bfloat16,
        embedding_weight_path: str | None = None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self._dtype = dtype
        self.embedding_weight_path = embedding_weight_path

    def encode(
        self,
        text: str,
        padding_side: str = "left",  # noqa: ARG002
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Run Gemma LLM and return raw hidden states + attention mask.
        Calls the inner model (self.model.model) to skip lm_head logits computation (~500 MiB saving).
        Returns:
            (hidden_states, attention_mask) where hidden_states is a tuple of per-layer tensors.
        """
        token_pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        token_ids = [int(t[0]) for t in token_pairs]
        device = self._language_model_device()
        attention_mask = torch.tensor([[int(w[1]) for w in token_pairs]], device=device)
        if self.embedding_weight_path:
            inputs_embeds = self._load_inputs_embeds(token_ids, device)
            outputs = self.model.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        else:
            input_ids = torch.tensor([token_ids], device=device)
            outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        del outputs
        return hidden_states, attention_mask

    def _language_model_device(self) -> torch.device:
        norm = self.model.model.language_model.norm
        for tensor in (*tuple(norm.parameters(recurse=False)), *tuple(norm.buffers(recurse=False))):
            if tensor.device.type != "meta":
                return tensor.device
        for tensor in self.model.model.language_model.parameters():
            if tensor.device.type != "meta":
                return tensor.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_inputs_embeds(self, token_ids: list[int], device: torch.device) -> torch.Tensor:
        with safetensors.safe_open(self.embedding_weight_path, framework="pt", device="cpu") as handle:
            rows = handle.get_slice("language_model.model.embed_tokens.weight")[token_ids, :]
        inputs_embeds = rows.unsqueeze(0).to(device=device, dtype=self._dtype)
        embed_scale = self.model.model.language_model.embed_tokens.embed_scale
        return inputs_embeds * embed_scale.to(device=device, dtype=inputs_embeds.dtype)

    # --- Prompt enhancement methods ---

    def _enhance(
        self,
        messages: list[dict[str, str]],
        image: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        seed: int = 10,
    ) -> str:
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)
        pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id=pad_token_id)

        with torch.inference_mode(), torch.random.fork_rng(devices=[self.model.device]):
            torch.manual_seed(seed)
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
            generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
            enhanced_prompt = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return enhanced_prompt

    def enhance_t2v(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 10,
    ) -> str:
        """Enhance a text prompt for T2V generation."""
        system_prompt = system_prompt or self.default_gemma_t2v_system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]

        return self._enhance(messages, max_new_tokens=max_new_tokens, seed=seed)

    def enhance_i2v(
        self,
        prompt: str,
        image: torch.Tensor,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 10,
    ) -> str:
        """Enhance a text prompt for I2V generation using a reference image."""
        system_prompt = system_prompt or self.default_gemma_i2v_system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                ],
            },
        ]
        return self._enhance(messages, image=image, max_new_tokens=max_new_tokens, seed=seed)

    @functools.cached_property
    def default_gemma_i2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_i2v_system_prompt.txt")

    @functools.cached_property
    def default_gemma_t2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_t2v_system_prompt.txt")


# --- Standalone utility functions ---


@functools.lru_cache(maxsize=2)
def _load_system_prompt(prompt_name: str) -> str:
    with open(Path(__file__).parent / "prompts" / f"{prompt_name}", "r") as f:
        return f.read()


def _cat_with_padding(
    tensor: torch.Tensor,
    padding_length: int,
    value: int | float,
) -> torch.Tensor:
    """Concatenate a tensor with a padding tensor of the given value."""
    return torch.cat(
        [
            tensor,
            torch.full(
                (1, padding_length),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            ),
        ],
        dim=1,
    )


def _pad_inputs_for_attention_alignment(
    model_inputs: dict[str, torch.Tensor],
    pad_token_id: int = 0,
    alignment: int = 8,
) -> dict[str, torch.Tensor]:
    """Pad sequence length to multiple of alignment for Flash Attention compatibility."""
    seq_len = model_inputs.input_ids.shape[1]
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    padding_length = padded_len - seq_len

    if padding_length > 0:
        model_inputs["input_ids"] = _cat_with_padding(model_inputs.input_ids, padding_length, pad_token_id)
        model_inputs["attention_mask"] = _cat_with_padding(model_inputs.attention_mask, padding_length, 0)
        if "token_type_ids" in model_inputs and model_inputs["token_type_ids"] is not None:
            model_inputs["token_type_ids"] = _cat_with_padding(model_inputs["token_type_ids"], padding_length, 0)

    return model_inputs


def module_ops_from_gemma_root(gemma_root: str) -> tuple[ModuleOps, ...]:
    tokenizer_root = str(find_matching_file(gemma_root, "tokenizer.model").parent)
    processor_root = str(find_matching_file(gemma_root, "preprocessor_config.json").parent)
    embedding_weight_path = _find_embedding_weight_path(gemma_root)

    def load_tokenizer(module: GemmaTextEncoder) -> GemmaTextEncoder:
        module.tokenizer = LTXVGemmaTokenizer(tokenizer_root, 1024)
        module.embedding_weight_path = embedding_weight_path
        return module

    def load_processor(module: GemmaTextEncoder) -> GemmaTextEncoder:
        image_processor = AutoImageProcessor.from_pretrained(processor_root, local_files_only=True)
        if not module.tokenizer:
            raise ValueError("Tokenizer model operation must be performed before processor model operation")
        module.processor = Gemma3Processor(image_processor=image_processor, tokenizer=module.tokenizer.tokenizer)
        return module

    tokenizer_load_ops = ModuleOps(
        "TokenizerLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoder) and module.tokenizer is None,
        mutator=load_tokenizer,
    )
    processor_load_ops = ModuleOps(
        "ProcessorLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoder) and module.processor is None,
        mutator=load_processor,
    )
    return (tokenizer_load_ops, processor_load_ops)


def _find_embedding_weight_path(gemma_root: str) -> str:
    for path in sorted(Path(gemma_root).rglob("*.safetensors")):
        with safetensors.safe_open(str(path), framework="pt", device="cpu") as handle:
            if "language_model.model.embed_tokens.weight" in handle.keys():
                return str(path)
    raise FileNotFoundError(f"Gemma embedding weights not found under {gemma_root}")
