from __future__ import annotations

from typing import Iterable, Tuple

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from schemas.types import ImageTensor, ImagesTensor


def pil_to_image_tensor(image: Image.Image) -> ImageTensor:
    return to_tensor(image).permute(1, 2, 0).contiguous()


def image_tensor_to_pil(image: ImageTensor) -> Image.Image:
    return to_pil_image(image.detach().cpu().permute(2, 0, 1).contiguous())


def pil_images_to_images_tensor(images: Iterable[Image.Image]) -> ImagesTensor:
    return torch.stack([pil_to_image_tensor(image) for image in images], dim=0)
