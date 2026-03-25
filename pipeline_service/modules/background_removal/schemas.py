from typing import Iterable, Optional, Tuple

from modules.background_removal.background_removal_pipeline import BackgroundRemovalPipeline
from modules.background_removal.params import BackgroundRemovalParamsOverrides
from pydantic import BaseModel
from schemas.internal import Internal
from schemas.types import ImageTensor, ImagesTensor


class BackgroundRemovalInput(BaseModel):
    """Input for background removal."""
    model: Internal[BackgroundRemovalPipeline]
    images: ImagesTensor | Iterable[ImageTensor]
    params: Optional[BackgroundRemovalParamsOverrides] = None


class BackgroundRemovalOutput(BaseModel):
    """Output from background removal."""
    images: Tuple[ImageTensor, ...]
