from __future__ import annotations

import time

from modules.background_removal.params import BackgroundRemovalParams
import torch
from torchvision.transforms.functional import crop, resize, resized_crop

from logger_config import logger
from schemas.types import ImageCHWTensor, ImageSize, ImageTensor
from modules.background_removal.schemas import BackgroundRemovalInput, BackgroundRemovalOutput

from .background_removal_pipeline import BackgroundRemovalPipeline


class BackgroundRemovalModule:
    """Shared image post-processing for background removal outputs."""

    def __init__(self, params: BackgroundRemovalParams):
        self.default_params = params

    def remove_background(self, request: BackgroundRemovalInput) -> BackgroundRemovalOutput:
        model: BackgroundRemovalPipeline = request.model
        start_time = time.time()

        params: BackgroundRemovalParams = self.default_params.overrided(request.params)

        images = list(request.images)
        outputs: list[ImageTensor] = []

        for image in images:
            if self._has_nonopaque_alpha(image):
                outputs.append(image[..., :3].contiguous())
                continue

            image_tensor = self._prepare_model_input_tensor(image, params.input_image_size)
            tensor_rgba_batch = model.predict_rgba(image_tensor.unsqueeze(0))
            tensor_rgba = tensor_rgba_batch[0]
            cropped_rgba = self._crop_and_center(tensor_rgba, params=params)
            cropped_rgb = self._blackout_background(cropped_rgba)
            outputs.append(cropped_rgb.permute(1, 2, 0).contiguous())

        removal_time = time.time() - start_time
        logger.success(
            f"Background remove - Time: {removal_time:.2f}s - "
            f"Images without background: {len(outputs)}"
        )

        return BackgroundRemovalOutput(images=tuple(outputs))

    def _has_nonopaque_alpha(self, image: ImageTensor) -> bool:
        if image.shape[-1] != 4:
            return False
        alpha = image[..., 3]
        return not torch.all(alpha >= 1.0 - (1.0 / 255.0)).item()

    @staticmethod
    def _prepare_model_input_tensor(image: ImageTensor, input_size: ImageSize) -> ImageTensor:

        image_rgb = image[..., :3].broadcast_to(*image.shape[:-1], 3)

        image_chw = image_rgb.permute(2, 0, 1)
        image_chw = resize(image_chw, size=input_size, antialias=False)
        return image_chw.permute(1, 2, 0).contiguous()

    def _crop_and_center(self, tensor_rgba: ImageCHWTensor, params: BackgroundRemovalParams) -> ImageCHWTensor:
        mask = tensor_rgba[3]

        bbox_indices = torch.argwhere(mask > 0.8)
        if len(bbox_indices) == 0:
            crop_args = dict(top=0, left=0, height=mask.shape[1], width=mask.shape[0])
        else:
            h_min, h_max = torch.aminmax(bbox_indices[:, 1])
            w_min, w_max = torch.aminmax(bbox_indices[:, 0])
            width, height = w_max - w_min, h_max - h_min
            center = (h_max + h_min) / 2, (w_max + w_min) / 2
            size = max(width, height)
            padded_size_factor = 1 + params.padding_percentage
            size = max(int(size * padded_size_factor), 2)

            top = int(center[1] - size // 2)
            left = int(center[0] - size // 2)
            bottom = int(center[1] + size // 2)
            right = int(center[0] + size // 2)

            if params.limit_padding:
                top = max(0, top)
                left = max(0, left)
                bottom = min(mask.shape[1], bottom)
                right = min(mask.shape[0], right)

            crop_args = dict(
                top=top,
                left=left,
                height=bottom - top,
                width=right - left,
            )

        if params.output_image_size is not None:
            return resized_crop(tensor_rgba, **crop_args, size=params.output_image_size, antialias=False)
        return crop(tensor_rgba, **crop_args)

    def _blackout_background(self, tensor_rgba: ImageCHWTensor) -> ImageCHWTensor:
        mask = tensor_rgba[3]
        mask_3ch = mask.unsqueeze(-3)
        return tensor_rgba[:3] * mask_3ch
