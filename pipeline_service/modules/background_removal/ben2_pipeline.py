from __future__ import annotations

import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from ben2 import BEN_Base

from config.settings import ModelVersionsConfig
from schemas.types import ImageTensor, ImagesCHWTensor, ImagesTensor 

from .background_removal_pipeline import BackgroundRemovalPipeline
from .settings import BackgroundRemovalConfig


class BEN2BackgroundRemovalPipeline(BackgroundRemovalPipeline):
    def __init__(self, settings: BackgroundRemovalConfig, model_versions: ModelVersionsConfig):
        super().__init__(settings, model_versions)
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

    def _load_model(self) -> BEN_Base:
        revision = self.model_versions.get_revision(self.settings.model_id)
        model = BEN_Base.from_pretrained(self.settings.model_id, revision=revision)
        return model.to(self.device).eval()

    def predict_rgba(self, image: ImageTensor | ImagesTensor) -> ImagesCHWTensor:
        assert self.is_ready(), f"{self.settings.model_id} model not loaded."

        image = image.contiguous().view(-1, *image.shape[-3:])
        image_chw_batch = image.permute(0, 3, 1, 2)
        foreground_list: list[torch.Tensor] = []

        with torch.no_grad():
            for image_chw in image_chw_batch:
                rgb_image = to_pil_image(image_chw.cpu()).convert("RGB")
                foreground = self.model.inference(rgb_image.copy())
                foreground_tensor = self.transforms(foreground)
                foreground_list.append(foreground_tensor[:4])

        return torch.stack(foreground_list, dim=0)
