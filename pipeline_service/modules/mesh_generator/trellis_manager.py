from __future__ import annotations

import os
import time
from typing import Iterable, Optional

import numpy as np
from libs.trellis2.representations.mesh.base import MeshWithVoxel
import torch
from PIL import Image

from .settings import TrellisConfig
from config.settings import ModelVersionsConfig
from logger_config import logger
from libs.trellis2.pipelines import Trellis2ImageTo3DPipeline
from .schemas import TrellisRequest, TrellisParams
from .enums import TrellisPipeType

class TrellisService:
    def __init__(self, trellis_config: TrellisConfig, model_versions: ModelVersionsConfig):
        self.settings = trellis_config
        self.model_versions = model_versions
        self.pipeline: Optional[Trellis2ImageTo3DPipeline] = None
        self.gpu = trellis_config.gpu
        self.default_params = TrellisParams.from_settings(
            self.settings, pipeline_type=self.settings.pipeline_type
        )

    def _get_model_revisions(self) -> tuple[Optional[str], dict[str, str]]:
        """
        Get pinned revisions for Trellis pipeline and related models.
        
        Returns:
            Tuple of (trellis_revision, model_revisions_dict)
        """
        trellis_revision = self.model_versions.get_revision(self.settings.model_id)
        
        model_revisions = {
            model_id: revision 
            for model_id, revision in self.model_versions.models.items()
            if model_id != self.settings.model_id
        }
        
        return trellis_revision, model_revisions

    async def startup(self) -> None:
        logger.info("Loading TRELLIS.2 pipeline...")
        os.environ.setdefault("ATTN_BACKEND", "flash-attn")
        os.environ.setdefault("SPCONV_ALGO", "native")
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu)

        trellis_revision, model_revisions = self._get_model_revisions()

        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            self.settings.model_id,
            config_file=self.settings.pipeline_config_path,
            revision=trellis_revision,
            model_revisions=model_revisions,
        )
        
        self.pipeline.cuda()
        self.pipeline.offload_decoders = self.settings.offload_decoders
        logger.success(f"{self.settings.model_id} pipeline ready (offload_decoders={self.settings.offload_decoders}).")

    async def shutdown(self) -> None:
        self.pipeline = None
        torch.cuda.empty_cache()
        logger.info(f"{self.settings.model_id} pipeline closed.")

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def generate(
        self,
        request: TrellisRequest,
    ) -> list[MeshWithVoxel]:
        if not self.pipeline:
            raise RuntimeError(f"{self.settings.model_id} pipeline not loaded.")

        images = request.image if isinstance(request.image, Iterable) else [request.image]
        images_rgb = [image.convert("RGB") for image in images]
        num_images = len(images_rgb)
        num_candidates = request.num_candidates

        params = self.default_params.overrided(request.params)

        logger.info(f"Generating Trellis {request.seed=} and image size {images[0].size} (Using {num_images} images, {num_candidates} candidates) | Pipeline: {params.pipeline_type.value} | Max Tokens: {params.max_num_tokens} | {'Mode: ' + params.mode.value if params.mode.value else ''}")

        logger.debug(f"Trellis generation parameters: {params}")

        start = time.time()
        try:
            sampler_kwargs = dict(
                seed=request.seed,
                sparse_structure_sampler_params={
                    "steps": params.sparse_structure_steps,
                    "guidance_strength": params.sparse_structure_cfg_strength,
                },
                shape_slat_sampler_params={
                    "steps": params.shape_slat_steps,
                    "guidance_strength": params.shape_slat_cfg_strength,
                },
                tex_slat_sampler_params={
                    "steps": params.tex_slat_steps,
                    "guidance_strength": params.tex_slat_cfg_strength,
                },
                pipeline_type=params.pipeline_type,
                max_num_tokens=params.max_num_tokens,
            )

            if num_images == 1:
                # Single-view: use run() with batch num_samples
                meshes = self.pipeline.run(
                    image=images_rgb[0],
                    num_samples=num_candidates,
                    **sampler_kwargs,
                )
            else:
                # Multi-view: use run_multi_image() with multiple images
                meshes = self.pipeline.run_multi_image(
                    images=images_rgb,
                    mode=params.mode,
                    **sampler_kwargs,
                )

            generation_time = time.time() - start
            logger.info(f"{self.settings.model_id} generated {len(meshes)} meshes in {generation_time:.2f}s")

            logger.success(
                f"{self.settings.model_id} finished in {generation_time:.2f}s. "
            )
            return meshes

        finally:
            torch.cuda.empty_cache()

    def generate_shape(
        self,
        request: TrellisRequest,
    ) -> dict:
        """
        Generate only the shape phase (sparse structure + shape SLat).
        Returns intermediate state for texture generation.
        """
        if not self.pipeline:
            raise RuntimeError(f"{self.settings.model_id} pipeline not loaded.")

        images = request.image if isinstance(request.image, Iterable) else [request.image]
        images_rgb = [image.convert("RGB") for image in images]
        num_images = len(images_rgb)
        num_candidates = request.num_candidates

        params = self.default_params.overrided(request.params)

        logger.info(f"Generating shape {request.seed=} and image size {images[0].size} (Using {num_images} images, {num_candidates} candidates) | Pipeline: {params.pipeline_type.value}")

        start = time.time()
        try:
            sampler_kwargs = dict(
                seed=request.seed,
                sparse_structure_sampler_params={
                    "steps": params.sparse_structure_steps,
                    "guidance_strength": params.sparse_structure_cfg_strength,
                },
                shape_slat_sampler_params={
                    "steps": params.shape_slat_steps,
                    "guidance_strength": params.shape_slat_cfg_strength,
                },
                tex_slat_sampler_params={
                    "steps": params.tex_slat_steps,
                    "guidance_strength": params.tex_slat_cfg_strength,
                },
                pipeline_type=params.pipeline_type,
                max_num_tokens=params.max_num_tokens,
            )

            if num_images == 1:
                try:
                    shape_result = self.pipeline.run_shape(
                        image=images_rgb[0],
                        num_samples=num_candidates,
                        **sampler_kwargs,
                    )
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    # Handle CUDA OOM during decode by retrying once with a safer pipeline.
                    # This prevents the entire request from failing with HTTP 500.
                    msg = str(e).lower()
                    is_oom = (
                        isinstance(e, torch.cuda.OutOfMemoryError)
                        or ("out of memory" in msg)
                    )
                    if not is_oom:
                        raise

                    if params.pipeline_type == TrellisPipeType.MODE_512:
                        raise

                    # On any OOM we want a deterministic, safer retry:
                    # - pipeline_type=512
                    # - exactly 2 candidates
                    fallback_num_candidates = 2
                    logger.warning(
                        f"CUDA OOM during Trellis shape (pipeline={params.pipeline_type.value}). "
                        f"Retrying with pipeline_type=512 (candidates={fallback_num_candidates})."
                    )
                    torch.cuda.empty_cache()

                    # Force pipeline_type=512 for the retry, but keep any other user/request
                    # overrides (except pipeline_type itself, which would otherwise undo the fallback).
                    override_dict = {}
                    if request.params is not None:
                        override_dict = request.params.model_dump(exclude_none=True)
                        override_dict.pop("pipeline_type", None)

                    fallback_params = TrellisParams.from_settings(
                        self.settings, pipeline_type=TrellisPipeType.MODE_512
                    ).overrided(TrellisParams.Overrides(**override_dict) if override_dict else None)

                    fallback_sampler_kwargs = dict(
                        seed=request.seed,
                        sparse_structure_sampler_params={
                            "steps": fallback_params.sparse_structure_steps,
                            "guidance_strength": fallback_params.sparse_structure_cfg_strength,
                        },
                        shape_slat_sampler_params={
                            "steps": fallback_params.shape_slat_steps,
                            "guidance_strength": fallback_params.shape_slat_cfg_strength,
                        },
                        tex_slat_sampler_params={
                            "steps": fallback_params.tex_slat_steps,
                            "guidance_strength": fallback_params.tex_slat_cfg_strength,
                        },
                        pipeline_type=fallback_params.pipeline_type,
                        max_num_tokens=fallback_params.max_num_tokens,
                    )

                    shape_result = self.pipeline.run_shape(
                        image=images_rgb[0],
                        num_samples=fallback_num_candidates,
                        **fallback_sampler_kwargs,
                    )
            else:
                # Multi-view: fall back to full generate (no split support yet)
                logger.info("Multi-view mode: using full generate (no split support)")
                meshes = self.pipeline.run_multi_image(
                    images=images_rgb,
                    mode=params.mode,
                    **sampler_kwargs,
                )
                return {"meshes": meshes, "is_complete": True}

            generation_time = time.time() - start
            logger.info(f"{self.settings.model_id} shape generated in {generation_time:.2f}s")

            # Measure voxel complexity (per-candidate token count)
            total_tokens = shape_result["shape_slat"].coords.shape[0]
            batch_size = shape_result["shape_slat"].shape[0]
            per_candidate_tokens = total_tokens // max(batch_size, 1)

            shape_result["voxel_count"] = per_candidate_tokens
            shape_result["is_complete"] = False

            logger.info(f"Voxel complexity: {per_candidate_tokens} tokens per candidate ({total_tokens} total, {batch_size} candidates)")

            return shape_result

        finally:
            torch.cuda.empty_cache()

    def generate_texture(
        self,
        shape_result: dict,
        num_candidates: int,
    ) -> list[MeshWithVoxel]:
        """
        Generate texture from pre-computed shape result.

        Args:
            shape_result: Dict returned by generate_shape().
            num_candidates: Number of candidates to generate texture for.
        """
        if not self.pipeline:
            raise RuntimeError(f"{self.settings.model_id} pipeline not loaded.")

        # If generate_shape fell back to full generation (multi-view), return directly
        if shape_result.get("is_complete", False):
            return shape_result["meshes"]

        start = time.time()
        try:
            try:
                meshes = self.pipeline.run_texture(
                    shape_result,
                    num_samples=num_candidates,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                # We intentionally do NOT swallow texture OOM here: pipeline_type must
                # be forced to 512, which requires regenerating shape_result.
                msg = str(e).lower()
                is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or ("out of memory" in msg)
                if not is_oom:
                    raise
                logger.warning(
                    f"CUDA OOM during Trellis texture (candidates={num_candidates}). "
                    "Reraising so the outer pipeline can retry with pipeline_type=512 and candidates=2."
                )
                torch.cuda.empty_cache()
                raise

            generation_time = time.time() - start
            logger.info(f"{self.settings.model_id} texture generated {len(meshes)} meshes in {generation_time:.2f}s")

            return meshes

        finally:
            torch.cuda.empty_cache()
