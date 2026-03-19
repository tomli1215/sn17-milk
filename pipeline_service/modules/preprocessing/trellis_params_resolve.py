"""
After RMBG: optional VLM 512/1024 routing + ``pipeline_profiles`` merge + per-request overrides.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

from logger_config import logger
from modules.judge.settings import JudgeConfig
from modules.mesh_generator.enums import TrellisPipeType
from modules.mesh_generator.schemas import TrellisParams
from modules.mesh_generator.settings import TrellisConfig

from .vllm_trellis_pipeline_router import choose_trellis_pipeline_via_vllm


async def resolve_trellis_params_after_rmbg(
    trellis_cfg: TrellisConfig,
    judge_cfg: JudgeConfig,
    *,
    trellis_params_override: Optional[TrellisParams.Overrides],
    trellis_run_overrides: Optional[dict],
    rmbg_preview_image: Image.Image,
    seed: int,
) -> TrellisParams.Overrides:
    """
    Decide effective ``pipeline_type`` (YAML explicit / VLM / default), apply
    ``trellis_cfg.pipeline_profiles``, then merge ``trellis_params_override``.
    When VLM picks the pipeline, any ``pipeline_type`` in overrides is ignored.
    """
    tro = trellis_run_overrides or {}
    route_vllm = trellis_cfg.pipeline_route_via_vllm
    if tro.get("pipeline_route_via_vllm") is not None:
        route_vllm = bool(tro["pipeline_route_via_vllm"])
    explicit_pt = bool(tro.get("explicit_pipeline_type"))

    def _effective_from_overrides() -> TrellisPipeType:
        if trellis_params_override is not None and trellis_params_override.pipeline_type is not None:
            return trellis_params_override.pipeline_type
        return trellis_cfg.pipeline_type

    vllm_chose = False
    if explicit_pt:
        effective_pt = trellis_params_override.pipeline_type  # type: ignore[union-attr]
    elif route_vllm:
        base_url = trellis_cfg.pipeline_route_vllm_url or judge_cfg.vllm_url
        model = trellis_cfg.pipeline_route_vllm_model or judge_cfg.vllm_model_name
        api_key = trellis_cfg.pipeline_route_vllm_api_key or judge_cfg.vllm_api_key
        if not base_url or not model:
            logger.warning(
                "pipeline_route_via_vllm is on but vLLM URL/model missing; using configured pipeline_type"
            )
            effective_pt = _effective_from_overrides()
        else:
            choice = await choose_trellis_pipeline_via_vllm(
                rmbg_preview_image,
                base_url=base_url,
                api_key=api_key or "local",
                model=model,
                timeout=trellis_cfg.pipeline_route_vllm_timeout,
                seed=seed,
                max_image_side=trellis_cfg.pipeline_route_vllm_max_image_side,
            )
            effective_pt = TrellisPipeType.MODE_512 if choice == "512" else TrellisPipeType.MODE_1024_CASCADE
            vllm_chose = True
            logger.info(f"VLLM Trellis router chose pipeline_type={effective_pt.value}")
    else:
        effective_pt = _effective_from_overrides()

    base = TrellisParams.from_settings(trellis_cfg, pipeline_type=effective_pt)
    merge_dict = (
        trellis_params_override.model_dump(exclude_none=True)
        if trellis_params_override is not None
        else {}
    )
    if vllm_chose:
        merge_dict.pop("pipeline_type", None)
    merged = base.overrided(TrellisParams.Overrides(**merge_dict))
    out = TrellisParams.Overrides(**merged.model_dump())
    logger.info(
        f"Trellis run: pipeline_type={merged.pipeline_type.value} "
        f"sparse={merged.sparse_structure_steps}/{merged.sparse_structure_cfg_strength} "
        f"shape={merged.shape_slat_steps}/{merged.shape_slat_cfg_strength} "
        f"tex={merged.tex_slat_steps}/{merged.tex_slat_cfg_strength} max_tokens={merged.max_num_tokens}"
    )
    return out
