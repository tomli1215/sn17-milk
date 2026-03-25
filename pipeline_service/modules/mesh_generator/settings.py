from typing import Optional

from pydantic import BaseModel, Field, model_validator
from .enums import TrellisPipeType, TrellisMode


class TrellisPipelineProfile(BaseModel):
    """Optional sampler overrides for a given pipeline_type key (e.g. \"512\", \"1024\")."""

    sparse_structure_steps: Optional[int] = None
    sparse_structure_cfg_strength: Optional[float] = None
    shape_slat_steps: Optional[int] = None
    shape_slat_cfg_strength: Optional[float] = None
    tex_slat_steps: Optional[int] = None
    tex_slat_cfg_strength: Optional[float] = None
    max_num_tokens: Optional[int] = None
    # If set, number of shape/texture candidates for this pipeline_type; else trellis.num_candidates.
    num_candidates: Optional[int] = None


class TrellisConfig(BaseModel):
    """TRELLIS.2 model configuration"""
    model_id: str = "microsoft/TRELLIS.2-4B"
    pipeline_config_path: str = "libs/trellis2/pipeline.json"
    sparse_structure_steps: int = 12
    sparse_structure_cfg_strength: float = 7.5
    shape_slat_steps: int = 12
    shape_slat_cfg_strength: float = 3.0
    tex_slat_steps: int = 12
    tex_slat_cfg_strength: float = 3.0
    pipeline_type: TrellisPipeType = TrellisPipeType.MODE_1024_CASCADE  # '512', '1024', '1024_cascade', '1536_cascade'
    # Per pipeline_type: overlay step/cfg/num_candidates on top of flat fields (missing keys keep flat defaults).
    pipeline_profiles: dict[str, TrellisPipelineProfile] = Field(default_factory=dict)
    max_num_tokens: int = 49152
    mode: TrellisMode = TrellisMode.MULTIDIFFUSION
    multiview: bool = False
    num_candidates: int = 2  # default when a pipeline_profiles entry omits num_candidates
    # If > 0 and time from pipeline start to end of Trellis texturing exceeds this (s), use 2 candidates for GLB conversion.
    texturing_time_limit_seconds: float = 0.0
    offload_decoders: bool = True
    gpu: int = 0
    # If true, VLM routing after RMBG (modules.preprocessing.trellis_params_resolve): 512 vs 1024.
    pipeline_route_via_vllm: bool = False
    pipeline_route_vllm_timeout: float = 45.0
    pipeline_route_vllm_url: Optional[str] = None  # default: judge.vllm_url
    pipeline_route_vllm_api_key: Optional[str] = None  # default: judge.vllm_api_key
    pipeline_route_vllm_model: Optional[str] = None  # default: judge.vllm_model_name
    pipeline_route_vllm_max_image_side: int = 768

    @model_validator(mode="after")
    def _normalize_pipeline_profile_keys(self) -> "TrellisConfig":
        if not self.pipeline_profiles:
            return self
        self.pipeline_profiles = {str(k): v for k, v in self.pipeline_profiles.items()}
        return self

    def pipeline_profile_for(self, pipeline_type: TrellisPipeType) -> Optional[TrellisPipelineProfile]:
        """Profile for this type; cascade modes fall back to the \"1024\" profile if present (same as TrellisParams.from_settings)."""
        prof = self.pipeline_profiles.get(pipeline_type.value)
        if not prof and pipeline_type.value in ("1024_cascade", "1536_cascade"):
            prof = self.pipeline_profiles.get("1024")
        return prof

    def num_candidates_for_pipeline_type(self, pipeline_type: TrellisPipeType) -> int:
        """Candidates for shape/texture when not overridden per-request (e.g. image YAML)."""
        prof = self.pipeline_profile_for(pipeline_type)
        if prof is not None and prof.num_candidates is not None:
            return prof.num_candidates
        return self.num_candidates
