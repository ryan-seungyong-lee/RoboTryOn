"""Caption embeddings for the video model, computed once so training and inference never hold
the 23 GB Gemma text encoder next to the transformer.

Two forms, because the two code paths enter the model at different points:

* **training** feeds the transformer directly and needs the text *connector outputs*
  ``{caption: {"video", "audio", "mask"}}``;
* **inference** calls the diffusers pipeline, which runs the connectors itself, and needs the
  *text-encoder outputs* ``{"max_seq", "embeds": {caption: (embeds, mask, neg, neg_mask)}}``
  with the pipeline's default negative prompt.

Both are encoded with ``max_sequence_length = 256``, the value training used.
"""

from __future__ import annotations

import torch

from .pipeline import MAX_SEQ, StereoIcLoraPipeline, snapshot_dir


def _text_pipeline(base: str, device: str):
    """The pipeline with only tokenizer and text encoder loaded."""
    pipe = StereoIcLoraPipeline.from_pretrained(
        snapshot_dir(base), transformer=None, vae=None, connectors=None, audio_vae=None,
        vocoder=None, duration_head=None, dtype=torch.bfloat16)
    pipe.text_encoder.to(device)
    return pipe


@torch.no_grad()
def encode_for_inference(captions: list[str], base: str = "Lightricks/LTX-2.5-Diffusers",
                         device: str = "cuda") -> dict:
    """``{caption: (embeds, mask, negative embeds, negative mask)}``, exactly as the
    pipeline's ``__call__`` would compute them."""
    from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
    pipe = _text_pipeline(base, device)
    out = {}
    for c in dict.fromkeys(captions):
        pe, pm, npe, npm = pipe.encode_prompt(
            prompt=c, negative_prompt=DEFAULT_NEGATIVE_PROMPT, do_classifier_free_guidance=True,
            num_videos_per_prompt=1, max_sequence_length=MAX_SEQ, device=device)
        out[c] = tuple(x.cpu() for x in (pe, pm, npe, npm))
    del pipe
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def encode_for_training(captions: list[str], base: str = "Lightricks/LTX-2.5-Diffusers",
                        device: str = "cuda") -> dict:
    """Connector outputs for the training loop."""
    from diffusers.pipelines.ltx2 import LTX2TextConnectors
    pipe = _text_pipeline(base, device)
    conn = LTX2TextConnectors.from_pretrained(snapshot_dir(base), subfolder="connectors",
                                              torch_dtype=torch.bfloat16).to(device).eval()
    side = getattr(pipe.tokenizer, "padding_side", "left")
    out = {}
    for c in dict.fromkeys(captions):
        pe, pm, _, _ = pipe.encode_prompt(prompt=c, negative_prompt=None,
                                          do_classifier_free_guidance=False,
                                          num_videos_per_prompt=1, max_sequence_length=MAX_SEQ,
                                          device=device)
        v, a, m = conn(pe, pm, padding_side=side)
        out[c] = {"video": v[0].cpu(), "audio": a[0].cpu(), "mask": m[0].cpu()}
    del pipe, conn
    torch.cuda.empty_cache()
    return out

