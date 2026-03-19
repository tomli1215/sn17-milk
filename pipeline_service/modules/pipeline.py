from __future__ import annotations

import base64
import io
import json
import random
import re
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Optional

from PIL import Image
from modules.converters.params import GLBConverterParams
import torch
import gc

from config.settings import SettingsConf
from config.prompting_library import PromptingLibrary
from logger_config import logger
from schemas.requests import GenerationRequest
from schemas.responses import GenerationResponse
from modules.mesh_generator.schemas import TrellisParams, TrellisRequest, TrellisResult
from modules.mesh_generator.enums import TrellisPipeType
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.background_removal.ben2_module import BEN2BackgroundRemovalService
from modules.background_removal.birefnet_module import BirefNetBackgroundRemovalService
from modules.grid_renderer.render import GridViewRenderer
from modules.mesh_generator.trellis_manager import TrellisService
from modules.converters.glb_converter import GLBConverter
from libs.trellis2.representations.mesh.base import MeshWithVoxel
from modules.judge.duel_manager import DuelManager
from modules.preprocessing.trellis_params_resolve import resolve_trellis_params_after_rmbg
from modules.utils import image_grid, secure_randint, set_random_seed, decode_image, to_png_base64, save_files


class GenerationPipeline:
    """
    Generation pipeline 
    """

    def __init__(self, settings: SettingsConf, renderer: Optional[GridViewRenderer] = None) -> None:
        self.settings = settings
        self.renderer = renderer

        # Initialize modules
        self.qwen_edit = QwenEditModule(settings.qwen, settings.model_versions)

        # Initialize background removal module
        if self.settings.background_removal.model_id == "PramaLLC/BEN2":
            self.rmbg = BEN2BackgroundRemovalService(settings.background_removal, settings.model_versions)
        elif self.settings.background_removal.model_id == "ZhengPeng7/BiRefNet_dynamic":
            self.rmbg = BirefNetBackgroundRemovalService(settings.background_removal, settings.model_versions)
        elif self.settings.background_removal.model_id == "ZhengPeng7/BiRefNet":
            self.rmbg = BirefNetBackgroundRemovalService(settings.background_removal, settings.model_versions)
        else:
            raise ValueError(f"Unsupported background removal model: {self.settings.background_removal.model_id}")

        # Initialize prompting libraries for both modes
        self.prompting_library = PromptingLibrary.from_file(settings.qwen.prompt_path_base)

        # Initialize Trellis module
        self.trellis = TrellisService(settings.trellis, settings.model_versions)
        self.glb_converter = GLBConverter(settings.glb_converter)

        # Initialize VLLM judge
        self.duel_manager = DuelManager(settings.judge) if settings.judge.enabled else None
        
    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_edit.startup()
        await self.rmbg.startup()
        await self.trellis.startup()
        
        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()
        
        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        # Shutdown all modules
        await self.qwen_edit.shutdown()
        await self.rmbg.shutdown()
        await self.trellis.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """
        Clean the GPU memory.
        """
        gc.collect()
        torch.cuda.empty_cache()

    async def warmup_generator(self) -> None:
        """Function for warming up the generator"""
        
        temp_image = Image.new("RGB",(512,512),color=(128,128,128))
        buffer = io.BytesIO()
        temp_image.save(buffer, format="PNG")
        temp_image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(temp_image_bytes).decode("utf-8")

        request = GenerationRequest(
            prompt_image=image_base64,
            prompt_type="image",
            seed=42
        )

        result = await self.generate(request)
        
        if result.glb_file_base64 and self.renderer:
            grid_view_bytes = self.renderer.grid_from_glb_bytes(result.glb_file_base64)
            if not grid_view_bytes:
                logger.warning("Grid view generation failed during warmup")

    async def generate_from_upload(self, image_bytes: bytes, seed: int, image_name: Optional[str] = None) -> bytes:
        """
        Generate 3D model from uploaded image file and return GLB as bytes.

        Args:
            image_bytes: Raw image bytes from uploaded file
            seed: Random seed for generation
            image_name: Optional original filename for naming output candidates (e.g. from upload)

        Returns:
            GLB file as bytes
        """
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Create request
        request = GenerationRequest(
            prompt_image=image_base64,
            prompt_type="image",
            seed=seed,
            image_name=image_name,
        )

        response = await self.generate(request)
        
        return response.glb_file_base64 # bytes
    
    def _get_dynamic_glb_params(self, mesh: MeshWithVoxel, request_params, elapsed_time: float):
        """
        Intelligent GLB parameter selection based on remaining time budget.
        Fast tasks get higher quality processing (texture, decimation).
        Slow tasks get reduced params to stay within timeout.
        """
        TIME_TARGET = 70  
        remaining = TIME_TARGET - elapsed_time
        face_count = mesh.faces.shape[0]

        if remaining > 55:
            dynamic = GLBConverterParams.Overrides(
                texture_size=3072,
                decimation_target=400000  
            )
            logger.debug(f"Dynamic GLB: FAST ({elapsed_time:.0f}s used, {remaining:.0f}s left, {face_count} faces) -> tex=3072, decim=400k")
        elif remaining > 40:
            dynamic = GLBConverterParams.Overrides(
                texture_size=2560,
                decimation_target=300000
            )
            logger.debug(f"Dynamic GLB: NORMAL ({elapsed_time:.0f}s used, {remaining:.0f}s left, {face_count} faces) -> tex=2560, decim=300k")
        elif remaining > 25:
            logger.debug(f"Dynamic GLB: MODERATE ({elapsed_time:.0f}s used, {remaining:.0f}s left, {face_count} faces) -> defaults")
            return request_params
        else:
            dynamic = GLBConverterParams.Overrides(
                texture_size=1536,
                decimation_target=180000
            )
            logger.debug(f"Dynamic GLB: SLOW ({elapsed_time:.0f}s used, {remaining:.0f}s left, {face_count} faces) -> tex=1536, decim=180k")

        if request_params:
            merged = dynamic.model_dump(exclude_none=True)
            merged.update(request_params.model_dump(exclude_none=True))
            return GLBConverterParams.Overrides(**merged)
        
        return dynamic
    
        
    def _edit_images(self, image: Image.Image, seed: int) -> list[Image.Image]:
        """
        Edit image based on current mode (multiview or base).

        Args:
            image: Input image to edit
            seed: Random seed for reproducibility

        Returns:
            List of edited images
        """
        prompting = self.prompting_library

        if self.settings.trellis.multiview:
            logger.info("Multiview mode: generating multiple views")
            views_prompt = prompting.promptings['views']

            edited_images = []
            for prompt_text in views_prompt.prompt:
                logger.debug(f"Editing view with prompt: {prompt_text}")
                result = self.qwen_edit.edit_image(
                    prompt_image=image,
                    seed=seed,
                    prompting=prompt_text,
                )
                edited_images.extend(result)

            edited_images.append(image.copy())  # Original image

            return edited_images

        # Base mode: only clean background, single view (1 image)
        logger.info("Base mode: single view with background cleaning and rotation")
        base_prompt = prompting.promptings['base']
        logger.debug(f"Editing base view with prompt: {base_prompt}")
        return self.qwen_edit.edit_image(
            prompt_image=image,
            seed=seed,
            prompting=base_prompt,
        )

    async def generate_meshes(
        self,
        request: GenerationRequest,
    ) -> tuple[list[MeshWithVoxel], list[Image.Image], list[Image.Image]]:
        """
        Generate meshes (batch) from Trellis pipeline, along with processed images.
        Uses two-phase generation: shape first, then texture.

        Args:
            request: Generation request with prompt and settings

        Returns:
            Tuple of (meshes, images_edited, images_without_background)
        """
        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
        set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)

        # 1. Edit the image using Qwen Edit
        # Qwen uses its own torch.Generator per call, but re-seed global state
        # before each step to ensure determinism regardless of consumed RNG ops.
        set_random_seed(request.seed)
        images_edited = list(self._edit_images(image, request.seed))

        # 2. Remove background
        set_random_seed(request.seed)
        images_with_background = list(image.copy() for image in images_edited)
        images_without_background = self.rmbg.remove_background(images_with_background)

        # Trellis OOM handling:
        # - On any CUDA OOM during shape/texture, we retry exactly with:
        #   pipeline_type=512 and num_candidates=2.
        # - This requires regenerating shape_result with the forced pipeline_type.
        did_oom_fallback = False

        while True:
            try:
                if not did_oom_fallback:
                    trellis_params = await resolve_trellis_params_after_rmbg(
                        self.settings.trellis,
                        self.settings.judge,
                        trellis_params_override=request.trellis_params,
                        trellis_run_overrides=None,
                        rmbg_preview_image=images_without_background[0],
                        seed=request.seed,
                    )

                    num_candidates = self.settings.trellis.num_candidates
                    texture_candidates = num_candidates
                else:
                    # Forced safer retry
                    forced_base = TrellisParams.from_settings(
                        self.settings.trellis,
                        pipeline_type=TrellisPipeType.MODE_512,
                    )
                    override_dict = {}
                    if request.trellis_params is not None:
                        override_dict = request.trellis_params.model_dump(exclude_none=True)
                        override_dict.pop("pipeline_type", None)
                    forced_merged = forced_base.overrided(
                        TrellisParams.Overrides(**override_dict) if override_dict else None
                    )
                    trellis_params = TrellisParams.Overrides(**forced_merged.model_dump())

                    num_candidates = 2
                    texture_candidates = 2

                # Phase 1: Generate shapes for all candidates
                set_random_seed(request.seed)
                trellis_request = TrellisRequest(
                    image=images_without_background,
                    seed=request.seed,
                    num_candidates=num_candidates,
                    params=trellis_params,
                )
                shape_result = self.trellis.generate_shape(trellis_request)

                # If multi-view fell back to full generation, return directly
                if shape_result.get("is_complete", False):
                    return shape_result["meshes"], images_edited, images_without_background

                # Phase 2: Generate texture
                meshes = self.trellis.generate_texture(shape_result, texture_candidates)
                return meshes, images_edited, images_without_background

            except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, RuntimeError) as e:
                msg = str(e).lower()
                is_oom = ("out of memory" in msg) or ("cuda" in msg and "memory" in msg)
                if not is_oom:
                    raise
                if did_oom_fallback:
                    raise

                did_oom_fallback = True
                logger.warning(
                    "CUDA OOM during Trellis. Retrying with pipeline_type=512 and candidates=2."
                )
                self._clean_gpu_memory()
                continue

    def convert_mesh_to_glb(self, mesh: MeshWithVoxel, glbconv_params: GLBConverterParams) -> bytes:
        """
        Convert mesh to GLB format using GLBConverter.

        Args:
            mesh: The mesh to convert
            glbconv_params: Optional override parameters for GLB conversion

        Returns:
            GLB file as bytes
        """
        start_time = time.time()
        glb_mesh = self.glb_converter.convert(mesh, params=glbconv_params)

        buffer = io.BytesIO()
        glb_mesh.export(file_obj=buffer, file_type="glb", extension_webp=False)
        buffer.seek(0)
        
        logger.info(f"GLB conversion time: {time.time() - start_time:.2f}s")
        return buffer.getvalue()

    def prepare_outputs(
        self,
        images_edited: list[Image.Image],
        images_without_background: list[Image.Image],
        glb_trellis_result: Optional[TrellisResult]
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Prepare output files: save to disk if configured and generate base64 strings if needed.

        Args:
            images_edited: List of edited images
            images_without_background: List of images with background removed
            glb_trellis_result: Generated GLB result (optional)

        Returns:
            Tuple of (image_edited_base64, image_without_background_base64)
        """
        start_time = time.time()
        # Create grid images once for both save and send operations
        image_edited_grid = image_grid(images_edited)
        image_without_background_grid = image_grid(images_without_background)

        # Save generated files if configured
        if self.settings.output.save_generated_files:
            save_files(glb_trellis_result, image_edited_grid, image_without_background_grid)

        # Convert to PNG base64 for response if configured
        image_edited_base64 = None
        image_without_background_base64 = None
        if self.settings.output.send_generated_files:
            image_edited_base64 = to_png_base64(image_edited_grid)
            image_without_background_base64 = to_png_base64(image_without_background_grid)
            
        logger.info(f"Output preparation time: {time.time() - start_time:.2f}s")

        return image_edited_base64, image_without_background_base64

    async def preview_edit(self, image_bytes: bytes, seed: int) -> dict:
        """
        Run Qwen edit + background removal only (no Trellis). Returns JSON-serializable dict
        with seed, image_edited_grid_base64, image_without_background_grid_base64,
        images_edited_base64, images_without_background_base64 (lists of base64 strings).
        """
        if seed < 0:
            seed = secure_randint(0, 10000)
        set_random_seed(seed)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image = decode_image(image_b64)
        set_random_seed(seed)
        images_edited = list(self._edit_images(image, seed))
        set_random_seed(seed)
        images_with_background = list(img.copy() for img in images_edited)
        images_without_background = self.rmbg.remove_background(images_with_background)
        edited_grid = image_grid(images_edited)
        no_bg_grid = image_grid(images_without_background)
        return {
            "seed": seed,
            "image_edited_grid_base64": to_png_base64(edited_grid),
            "image_without_background_grid_base64": to_png_base64(no_bg_grid),
            "images_edited_base64": [to_png_base64(img) for img in images_edited],
            "images_without_background_base64": [to_png_base64(img) for img in images_without_background],
        }

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Execute full generation pipeline with batch output.

        Args:
            request: Generation request with prompt and settings

        Returns:
            GenerateResponse with generated assets (first candidate GLB + all candidate renders)
        """
        t1 = time.time()

        logger.info(f"Request received | Seed: {request.seed} | Prompt Type: {request.prompt_type.value}")

        # Generate meshes (batch) and get processed images
        meshes, images_edited, images_without_background = await self.generate_meshes(request)

        # Timeline: if texturing finished after the limit, use only 2 candidates for GLB conversion
        elapsed_till_texturing = time.time() - t1
        limit_sec = self.settings.trellis.texturing_time_limit_seconds
        if limit_sec > 0 and elapsed_till_texturing > limit_sec:
            num_glb_candidates = min(2, len(meshes))
            logger.warning(
                f"Texturing took {elapsed_till_texturing:.0f}s (limit {limit_sec}s). "
                f"Using {num_glb_candidates} candidates for GLB conversion (config num_candidates={self.settings.trellis.num_candidates})."
            )
        else:
            num_glb_candidates = len(meshes)

        self._clean_gpu_memory()

        # Base name for output files: use image name when provided, else dev-friendly
        if request.image_name and request.image_name.strip():
            stem = Path(request.image_name).stem
            safe_base = re.sub(r"[^\w\-]", "_", stem).strip("_") or "image"
        else:
            safe_base = "dev"
        output_dir = self.settings.output.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert each mesh to GLB and render as candidate_x.png (up to num_glb_candidates)
        all_glb_bytes = []
        candidate_views = []
        candidate_filenames: list[str] = []

        for i in range(num_glb_candidates):
            mesh = meshes[i]
            meshes[i] = None  # Release reference to save VRAM

            logger.info(f"Processing candidate {i}/{num_glb_candidates - 1}...")
            mesh.simplify()

            elapsed = time.time() - t1
            dynamic_params = self._get_dynamic_glb_params(mesh, request.glbconv_params, elapsed) if self.settings.api.dynamic_params else request.glbconv_params
            glb_bytes = self.convert_mesh_to_glb(mesh, dynamic_params)
            all_glb_bytes.append(glb_bytes)

            # Render and save with name-based or dev filename
            if self.renderer:
                grid_view_bytes = self.renderer.grid_from_glb_bytes(glb_bytes)
                if grid_view_bytes:
                    candidate_filename = f"{safe_base}_candidate_{i}.png"
                    candidate_path = output_dir / candidate_filename
                    with open(candidate_path, "wb") as f:
                        f.write(grid_view_bytes)
                    candidate_filenames.append(candidate_filename)
                    logger.info(f"Rendered {candidate_filename} saved to {candidate_path}")
                    candidate_views.append(grid_view_bytes)
                else:
                    logger.warning(f"Grid view rendering failed for candidate {i}")

            del mesh, glb_bytes
            self._clean_gpu_memory()

        # Release unused mesh references when we capped GLB candidates by time limit
        for j in range(num_glb_candidates, len(meshes)):
            meshes[j] = None

        # VLLM Judge: round-robin — duel every pair, most wins wins; tie-break random (seeded)
        best_idx = 0
        duel_wins: list[int] = [0] * len(candidate_views)
        duel_pair_log: list[dict] = []
        if self.duel_manager and len(candidate_views) > 1:
            gc.collect()
            n = len(candidate_views)
            logger.info(f"VLLM round-robin: {n} candidates, {n * (n - 1) // 2} pairwise duels")
            prompt_image_bytes = base64.b64decode(request.prompt_image)

            for i, j in combinations(range(n), 2):
                pair_seed = request.seed + i * 10007 + j * 10009
                logger.info(f"Duel candidate {i} vs candidate {j} (seed={pair_seed})...")
                winner_idx, issues = await self.duel_manager.run_duel(
                    prompt_image_bytes,
                    candidate_views[i],
                    candidate_views[j],
                    pair_seed,
                )
                # winner_idx -1 => first image (i) wins; 1 => second (j) wins
                if winner_idx == -1:
                    duel_wins[i] += 1
                    w = i
                else:
                    duel_wins[j] += 1
                    w = j
                duel_pair_log.append({"i": i, "j": j, "winner": w, "issues": issues[:200] if issues else ""})
                logger.info(f"  -> Winner: candidate {w} | {issues[:120]}...")

            max_w = max(duel_wins)
            leaders = [k for k in range(n) if duel_wins[k] == max_w]
            if len(leaders) == 1:
                best_idx = leaders[0]
                logger.success(f"Champion: candidate {best_idx} (duel wins: {duel_wins})")
            else:
                rng = random.Random(request.seed)
                best_idx = rng.choice(leaders)
                logger.success(
                    f"Champion: candidate {best_idx} (tie on {max_w} wins among {leaders}, random pick) | all wins: {duel_wins}"
                )
        elif len(candidate_views) == 1:
            duel_wins = [0]

        # One result JSON per input image (per 3D generation run), named by image name
        result_json_path = output_dir / f"{safe_base}_result.json"
        try:
            result_data = {
                "image_name": request.image_name or None,
                "candidate_filenames": candidate_filenames,
                "best_candidate_index": best_idx,
                "duel_wins_per_candidate": duel_wins,
                "duel_pairs": duel_pair_log,
            }
            with open(result_json_path, "w") as f:
                json.dump(result_data, f, indent=2)
            logger.info(f"Result JSON (per input image) written to {result_json_path}")
        except OSError as e:
            logger.warning(f"Failed to write result JSON: {e}")

        glb_trellis_result = TrellisResult(file_bytes=all_glb_bytes[best_idx]) if all_glb_bytes else None
        del all_glb_bytes

        # Save generated files
        image_edited_base64, image_no_bg_base64 = None, None
        if self.settings.output.save_generated_files or self.settings.output.send_generated_files:
            image_edited_base64, image_no_bg_base64 = self.prepare_outputs(
                images_edited,
                images_without_background,
                glb_trellis_result
            )
        del images_edited, images_without_background

        t2 = time.time()
        generation_time = t2 - t1

        logger.success(f"Generation time: {generation_time:.2f}s ({len(candidate_views)} candidates rendered, best={best_idx})")

        # Clean the GPU memory
        self._clean_gpu_memory()

        response = GenerationResponse(
            generation_time=generation_time,
            glb_file_base64=glb_trellis_result.file_bytes if glb_trellis_result else None,
            grid_view_file_base64=candidate_views[best_idx] if candidate_views else None,
            candidate_views=candidate_views if candidate_views else None,
            image_edited_file_base64=image_edited_base64,
            image_without_background_file_base64=image_no_bg_base64
        )

        return response