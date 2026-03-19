from dataclasses import dataclass
from typing import Optional, TypeAlias
from PIL import Image

from .enums import TrellisMode, TrellisPipeType
from schemas.overridable import OverridableModel


class TrellisParams(OverridableModel):
    """TRELLIS.2 parameters with automatic fallback to settings."""
    sparse_structure_steps: int
    sparse_structure_cfg_strength: float
    shape_slat_steps: int
    shape_slat_cfg_strength: float
    tex_slat_steps: int
    tex_slat_cfg_strength: float
    mode: TrellisMode = TrellisMode.STOCHASTIC # Currently unused in TRELLIS.2
    pipeline_type: TrellisPipeType = TrellisPipeType.MODE_1024_CASCADE  # '512', '1024', '1024_cascade', '1536_cascade'
    max_num_tokens: int = 49152
    
    
    @classmethod
    def from_settings(cls, settings, pipeline_type=None) -> "TrellisParams":
        """
        Build params for a given pipeline resolution. ``pipeline_type`` defaults to ``settings.pipeline_type``.
        Applies ``settings.pipeline_profiles[pipeline_type.value]`` on top of flat TrellisConfig fields.
        """
        from modules.mesh_generator.settings import TrellisConfig

        if not isinstance(settings, TrellisConfig):
            raise TypeError("settings must be TrellisConfig")
        pt = pipeline_type if pipeline_type is not None else settings.pipeline_type
        data = {
            "sparse_structure_steps": settings.sparse_structure_steps,
            "sparse_structure_cfg_strength": settings.sparse_structure_cfg_strength,
            "shape_slat_steps": settings.shape_slat_steps,
            "shape_slat_cfg_strength": settings.shape_slat_cfg_strength,
            "tex_slat_steps": settings.tex_slat_steps,
            "tex_slat_cfg_strength": settings.tex_slat_cfg_strength,
            "pipeline_type": pt,
            "mode": settings.mode,
            "max_num_tokens": settings.max_num_tokens,
        }
        prof = settings.pipeline_profiles.get(pt.value)
        if not prof and pt.value in ("1024_cascade", "1536_cascade"):
            prof = settings.pipeline_profiles.get("1024")
        if prof:
            for k, v in prof.model_dump(exclude_none=True).items():
                data[k] = v
        return cls(**data)


TrellisParamsOverrides: TypeAlias = TrellisParams.Overrides


@dataclass
class TrellisRequest:
    """Request for TRELLIS.2 3D generation (internal use only)."""
    image: Image.Image
    seed: int
    num_candidates: int = 2
    params: Optional[TrellisParamsOverrides] = None


@dataclass(slots=True)
class TrellisResult:
    """Result from TRELLIS.2 3D generation."""
    file_bytes: bytes | None = None
