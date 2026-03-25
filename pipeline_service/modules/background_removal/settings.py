from typing import Optional
from typing_extensions import Tuple

from config.types import ModelConfig
from modules.background_removal.enums import RMBGModelType
from modules.background_removal.params import BackgroundRemovalParams


class BackgroundRemovalConfig(ModelConfig):
    """Background removal configuration"""
    model_id: str = "ZhengPeng7/BiRefNet"
    model_type: RMBGModelType = RMBGModelType.BIREFNET
    params: BackgroundRemovalParams = BackgroundRemovalParams()
