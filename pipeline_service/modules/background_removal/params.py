from typing import Optional, TypeAlias
from schemas.overridable import OverridableModel
from schemas.types import ImageSize

class BackgroundRemovalParams(OverridableModel):
    """Background removal parameters with automatic fallback to settings."""
    input_image_size: ImageSize = (1024, 1024)
    output_image_size: Optional[ImageSize] = None
    padding_percentage: float = 0.0
    limit_padding: bool = True

BackgroundRemovalParamsOverrides: TypeAlias = BackgroundRemovalParams.Overrides