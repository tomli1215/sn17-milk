from __future__ import annotations

import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from config.settings import ModelVersionsConfig
from schemas.types import ImageTensor, ImagesCHWTensor, ImagesTensor

from .background_removal_pipeline import BackgroundRemovalPipeline
from .settings import BackgroundRemovalConfig


class BirefNetBackgroundRemovalPipeline(BackgroundRemovalPipeline):
    def __init__(self, settings: BackgroundRemovalConfig, model_versions: ModelVersionsConfig):
        super().__init__(settings, model_versions)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _load_model(self) -> AutoModelForImageSegmentation:
        revision = self.model_versions.get_revision(self.settings.model_id)
        model = AutoModelForImageSegmentation.from_pretrained(
            self.settings.model_id,
            revision=revision,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        return model.to(self.device)

    def predict_rgba(self, image: ImageTensor | ImagesTensor) -> ImagesCHWTensor:
        assert self.is_ready(), f"{self.settings.model_id} model not loaded."
        image = image.contiguous().view(-1, *image.shape[-3:])
        rgb_tensor = image.permute(0, 3, 1, 2).contiguous().to(self.device, dtype=torch.float32)
        input_tensor = self.normalize(rgb_tensor)

        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid()
            mask = preds[:, :1].mul_(255).round_().div_(255).float()
        return torch.cat([rgb_tensor, mask], dim=1)
