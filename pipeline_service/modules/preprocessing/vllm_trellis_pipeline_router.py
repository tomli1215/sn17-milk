"""
VLM (OpenAI-compatible, e.g. vLLM) chooses Trellis pipeline 512 vs 1024 from an image.
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Literal

import httpx
from openai import AsyncOpenAI
from PIL import Image

from logger_config import logger

PipelineChoice = Literal["512", "1024"]

SYSTEM_PROMPT = """You route single-image 3D reconstruction to one of two GPU pipelines. Reply with ONLY the JSON line, no other text.

- **512**: Use when the subject has dense fine detail that would exhaust GPU at 1024:
  - **Fur**, **hair**, **crest**, **plume**, **fleece**, **feathers**, **mane**
  - Grass, foliage, lace, chain mail, dense wires, complex thin geometry
  - Fireworks, sparks, glitter, fluffy or highly intricate organic surfaces
  - Any subject that is visually very busy with fine detail

- **1024**: Use only for smooth, compact objects with clear silhouette (vehicles, furniture, rigid props, simple toys, plain metal/glass) and no prominent fur/hair/feathers.

If the image shows fur, hair, a crest, plume, or similar fine detail, you MUST choose 512.

Output exactly one of:
{"pipeline":"512"}
{"pipeline":"1024"}
"""


def _image_to_data_url(img: Image.Image, max_side: int = 768) -> str:
    rgb = img.convert("RGB")
    w, h = rgb.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        rgb = rgb.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _parse_pipeline_json(text: str) -> PipelineChoice:
    text = (text or "").strip()
    # Strip markdown code fence so we can parse JSON inside
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    # Try to find a JSON object with "pipeline" key (allow surrounding text)
    json_match = re.search(r'\{\s*"pipeline"\s*:\s*"(512|1024)"\s*\}', text)
    if json_match:
        return json_match.group(1)  # type: ignore[return-value]
    try:
        data = json.loads(text)
        p = str(data.get("pipeline", "")).strip().lower()
        if p in ("512", "1024"):
            return p  # type: ignore[return-value]
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    # Fallback when no valid JSON: prefer 512 if it appears (e.g. "use 512 not 1024")
    # so we don't wrongly pick 1024 when the model explains why 1024 is wrong
    has_512 = re.search(r"\b512\b", text)
    has_1024 = re.search(r"\b1024\b", text)
    if has_512 and not has_1024:
        return "512"
    if has_1024 and not has_512:
        return "1024"
    if has_512 and has_1024:
        # Both mentioned (e.g. "use 512 because 1024 would OOM") — prefer 512 for safety
        return "512"
    logger.warning(f"VLLM pipeline router: could not parse JSON, defaulting to 512. Raw: {text[:200]!r}")
    return "512"


async def choose_trellis_pipeline_via_vllm(
    image: Image.Image,
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 45.0,
    seed: int = 0,
    max_image_side: int = 768,
) -> PipelineChoice:
    """Returns \"512\" or \"1024\"; on failure returns \"512\" (OOM-safe)."""
    data_url = _image_to_data_url(image, max_side=max_image_side)
    client = AsyncOpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key or "local",
        http_client=httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=5)),
    )
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Which pipeline (512 or 1024) for 3D reconstruction? Reply with only: {\"pipeline\":\"512\"} or {\"pipeline\":\"1024\"}",
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=128,
            seed=seed,
        )
        raw = completion.choices[0].message.content or ""
        logger.info(f"VLLM Trellis pipeline router raw response: {raw!r}")
        choice = _parse_pipeline_json(raw)
        logger.info(f"VLLM Trellis pipeline router chose: {choice} (model={model})")
        return choice
    except Exception as e:
        logger.warning(f"VLLM Trellis pipeline router failed ({e}); defaulting to 512")
        return "512"
    finally:
        try:
            await client.http_client.aclose()
        except Exception:
            pass
