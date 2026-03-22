import time
from typing import Dict, Tuple
from logger_config import logger
import numpy as np
import torch
import kaolin
from PIL import Image
import trimesh
import trimesh.visual
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from libs.trellis2.representations.mesh.base import MeshWithVoxel
from geometry.mesh.schemas import MeshData, MeshDataWithAttributeGrid, AttributeGrid
from geometry.texturing.dithering import bayer_dither_pattern
from geometry.mesh.utils import sort_mesh, map_vertices_positions, count_boundary_loops
from geometry.mesh.subdivisions import subdivide_egdes
from geometry.mesh.smoothing import taubin_smooth
from geometry.texturing.utils import dilate_attributes, map_mesh_rasterization, rasterize_mesh_data, sample_grid_attributes
from geometry.texturing.enums import AlphaMode
from .params import GLBConverterParams
from .settings import GLBConverterConfig
import cumesh
from libs.cumesh_skippable_errors import skippable_cumesh_fill_holes_error


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError)):
        return True
    return "out of memory" in str(exc).lower()


def _oom_empty_cache() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def _simple_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    fn = F.normalize(fn, dim=-1, eps=1e-12)
    vn = torch.zeros_like(vertices)
    for i in range(3):
        vn.index_add_(0, faces[:, i], fn)
    return F.normalize(vn, dim=-1, eps=1e-12)


DITHER_PATTERN_SIZE = 16
DITHER_PATTERN = bayer_dither_pattern(4096, 4096, DITHER_PATTERN_SIZE)


class GLBConverter:
    """Converter for extracting and texturing meshes to GLB format."""
    DEFAULT_AABB = torch.as_tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32)
    DILATION_KERNEL_SIZE = 5
    
    def __init__(self, settings: GLBConverterConfig):
        """Initialize converter with settings."""
        self.default_params = GLBConverterParams.from_settings(settings)
        self.logger = logger
        self.device = torch.device(f'cuda:{settings.gpu}' if torch.cuda.is_available() else 'cpu')

    def _oom_call_void(self, step: str, fn) -> None:
        """Run fn(); on OOM log, clear CUDA cache, and continue (operation skipped)."""
        try:
            fn()
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, RuntimeError, MemoryError) as e:
            if not _is_oom(e):
                raise
            self.logger.warning(f"OOM skipped ({step}): {e}")
            _oom_empty_cache()

    def _oom_call_val(self, step: str, fn, default):
        """Return fn() or `default` / default() after OOM (no retries)."""
        try:
            return fn()
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, RuntimeError, MemoryError) as e:
            if not _is_oom(e):
                raise
            self.logger.warning(f"OOM in {step}, using fallback: {e}")
            _oom_empty_cache()
            return default() if callable(default) else default

    def _mesh_planar_uv_fallback(self, mesh_data: MeshData) -> MeshData:
        """Cheap UVs + normals when uv_unwrap cannot run (e.g. OOM)."""
        v = mesh_data.vertices
        xy = v[:, :2]
        mn = xy.min(dim=0).values
        mx = xy.max(dim=0).values
        span = (mx - mn).clamp_min(1e-6)
        uvs = ((xy - mn) / span).clamp(0.0, 1.0)
        if mesh_data.vertex_normals is not None:
            vn = mesh_data.vertex_normals
        else:
            vn = _simple_vertex_normals(v, mesh_data.faces)
        self.logger.warning("Using planar UV fallback after OOM or unwrap failure.")
        return MeshData(
            vertices=v,
            faces=mesh_data.faces,
            uvs=uvs,
            vertex_normals=vn,
        )

    def _safe_fill_holes(self, cumesh_mesh: cumesh.CuMesh, max_hole_perimeter: float, step: str) -> None:
        """fill_holes can OOM or hit bad CUDA launch configs; skip and keep topology unchanged."""
        try:
            cumesh_mesh.fill_holes(max_hole_perimeter=max_hole_perimeter)
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, RuntimeError, MemoryError) as e:
            if not skippable_cumesh_fill_holes_error(e):
                raise
            self.logger.warning(f"fill_holes skipped ({step}): {e}")
            _oom_empty_cache()
    
    def convert(self, mesh: MeshWithVoxel, aabb: torch.Tensor = DEFAULT_AABB, params: GLBConverterParams = None) -> trimesh.Trimesh:
        """Convert the given mesh to a textured GLB format."""
        logger.debug(f"Original mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
        
        params = self.default_params.overrided(params)
        logger.debug(f"Using GLB conversion parameters: {params}")

        # 1. Prepare original mesh data with BVH
        original_mesh_data = self._prepare_original_mesh(mesh, aabb)
        
        # 2. Remesh if required (cleanup otherwise)
        if params.remesh:
            mesh_data = self._remesh_mesh(original_mesh_data, params)
        else:
            mesh_data = self._cleanup_mesh(original_mesh_data, params)
            
        # 3. UV unwrap the mesh
        mesh_data = self._uv_unwrap_mesh(mesh_data, params)

        # 4. subdivide unwrapped mesh
        mesh_data = self._subdivide_mesh(mesh_data, original_mesh_data, params)
        
        # 5. Rasterize attributes onto the mesh UVs
        attributes_layout = mesh.layout
        attributes, attributes_layout = self._rasterize_attributes(mesh_data, original_mesh_data, attributes_layout, params)
        
        # 6. Post-process the rasterized attributes into textures
        base_color, orm_texture = self._texture_postprocess(attributes, attributes_layout, params)

        # 7. Create the textured mesh
        textured_mesh = self._create_textured_mesh(mesh_data, base_color, orm_texture, params)

        return textured_mesh


    def _prepare_original_mesh(self, mesh: MeshWithVoxel, aabb: torch.Tensor, compute_vertex_normals: bool = False) -> MeshDataWithAttributeGrid:
        """
        Convert MeshWithVoxel to OriginalMeshData.
        WARNING: This method also fills holes outputing additional faces compared to input one.
        """
        logger.debug(f"Preparing original mesh data")
        start_time = time.time() 

        # Prepare attribute grid
        attrs = AttributeGrid(
            values=mesh.attrs.to(self.device),
            coords=mesh.coords.to(self.device),
            aabb = torch.as_tensor(aabb, dtype=torch.float32, device=self.device),
            voxel_size = torch.as_tensor(mesh.voxel_size, dtype=torch.float32, device=self.device).broadcast_to(3)
        )

        vertices = mesh.vertices.to(self.device)
        faces = mesh.faces.to(self.device)
        vertices_in, faces_in = vertices, faces
        
        cumesh_mesh = cumesh.CuMesh()
        cumesh_mesh.init(vertices, faces)
        self._safe_fill_holes(cumesh_mesh, 3e-2, "prepare_original_mesh")
        logger.debug(f"After filling holes: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

        vertex_normals = None
        vertices, faces = self._oom_call_val(
            "prepare_original_mesh.cumesh_read",
            lambda: cumesh_mesh.read(),
            default=(vertices_in, faces_in),
        )

        if compute_vertex_normals:
            def _vertex_normals():
                cumesh_mesh.compute_vertex_normals()
                return cumesh_mesh.read_vertex_normals()

            vertex_normals = self._oom_call_val("compute_vertex_normals", _vertex_normals, default=None)

        original_mesh_data = MeshDataWithAttributeGrid(vertices=vertices, faces=faces,vertex_normals=vertex_normals, attrs=attrs)
        
        # Build BVH for the current mesh to guide remeshing
        logger.debug(f"Building BVH for current mesh...")
        self._oom_call_void("build_bvh", original_mesh_data.build_bvh)
        logger.debug(f"Done building BVH | Time: {time.time() - start_time:.2f}s")
        
        return original_mesh_data

    def _cleanup_mesh(self, original_mesh_data: MeshDataWithAttributeGrid, params: GLBConverterParams) -> MeshData:
        """Cleanup and optimize the mesh using decimation and remeshing."""
        # Create cumesh from current mesh data
        cumesh_mesh = cumesh.CuMesh()
        cumesh_mesh.init(original_mesh_data.vertices, original_mesh_data.faces)

        if params.smooth_mesh:
            logger.debug(f"Applying Taubin smoothing: iterations={params.smooth_iterations}, lambda={params.smooth_lambda}")

            smoothed_vertices = self._oom_call_val(
                "cleanup_mesh.taubin_smooth",
                lambda: taubin_smooth(
                    original_mesh_data,
                    iterations=params.smooth_iterations,
                    lambda_factor=params.smooth_lambda,
                    mu_factor=-(params.smooth_lambda + 0.01),
                ),
                default=None,
            )
            if smoothed_vertices is not None:
                cumesh_mesh.init(smoothed_vertices, original_mesh_data.faces)

        # Step 1: Aggressive simplification (3x target)
        self._oom_call_void(
            "cleanup_mesh.simplify_initial",
            lambda: cumesh_mesh.simplify(params.decimation_target * 3, verbose=False),
        )
        logger.debug(f"After initial simplification: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")
        
        # Step 2: Clean up topology (duplicates, non-manifolds, isolated parts)
        self._oom_call_void(
            "cleanup_mesh.remove_duplicate_faces",
            lambda: cumesh_mesh.remove_duplicate_faces(),
        )
        self._oom_call_void(
            "cleanup_mesh.repair_non_manifold_edges",
            lambda: cumesh_mesh.repair_non_manifold_edges(),
        )
        self._oom_call_void(
            "cleanup_mesh.remove_small_components",
            lambda: cumesh_mesh.remove_small_connected_components(1e-5),
        )
        self._safe_fill_holes(cumesh_mesh, 3e-2, "cleanup_mesh.after_initial_cleanup")
        logger.debug(f"After initial cleanup: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")
            
        # Step 3: Final simplification to target count
        self._oom_call_void(
            "cleanup_mesh.simplify_final",
            lambda: cumesh_mesh.simplify(params.decimation_target, verbose=False),
        )
        logger.debug(f"After final simplification: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")
        
        # Step 4: Final Cleanup loop
        self._oom_call_void(
            "cleanup_mesh.remove_duplicate_faces_2",
            lambda: cumesh_mesh.remove_duplicate_faces(),
        )
        self._oom_call_void(
            "cleanup_mesh.repair_non_manifold_edges_2",
            lambda: cumesh_mesh.repair_non_manifold_edges(),
        )
        self._oom_call_void(
            "cleanup_mesh.remove_small_components_2",
            lambda: cumesh_mesh.remove_small_connected_components(1e-5),
        )
        self._safe_fill_holes(cumesh_mesh, 3e-2, "cleanup_mesh.final_cleanup")
        logger.debug(f"After final cleanup: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

        hole_pair = self._oom_call_val(
            "cleanup_mesh.count_holes_read", lambda: cumesh_mesh.read(), default=None
        )
        if hole_pair is not None:
            hole_count = count_boundary_loops(*hole_pair)
            logger.debug(f"Holes after cleanup: {hole_count}")

        # Step 5: Unify face orientations
        self._oom_call_void(
            "cleanup_mesh.unify_face_orientations",
            lambda: cumesh_mesh.unify_face_orientations(),
        )

        # Extract cleaned mesh data
        vertices, faces = self._oom_call_val(
            "cleanup_mesh.final_read",
            lambda: cumesh_mesh.read(),
            default=(original_mesh_data.vertices, original_mesh_data.faces),
        )
        return MeshData(
            vertices=vertices,
            faces=faces
        )

    def _remesh_mesh(self, original_mesh_data: MeshDataWithAttributeGrid, params: GLBConverterParams) -> MeshData:
        """Remesh the given mesh to improve quality."""

        # Create cumesh from current mesh data
        logger.debug("Starting remeshing")
        start_time = time.time()

        cumesh_mesh = cumesh.CuMesh()
        cumesh_mesh.init(original_mesh_data.vertices, original_mesh_data.faces)

        if params.smooth_mesh:
            logger.debug(f"Applying Taubin smoothing: iterations={params.smooth_iterations}, lambda={params.smooth_lambda}")

            smoothed_vertices = self._oom_call_val(
                "remesh_mesh.taubin_smooth",
                lambda: taubin_smooth(
                    original_mesh_data,
                    iterations=params.smooth_iterations,
                    lambda_factor=params.smooth_lambda,
                    mu_factor=-(params.smooth_lambda + 0.01),
                ),
                default=None,
            )
            if smoothed_vertices is not None:
                cumesh_mesh.init(smoothed_vertices, original_mesh_data.faces)
            logger.debug(f"Done smoothing | Time: {time.time() - start_time:.2f}s")

        voxel_size = original_mesh_data.attrs.voxel_size
        aabb = original_mesh_data.attrs.aabb
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()

        resolution = grid_size.max().item()
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        
        rv, rf = self._oom_call_val(
            "remesh_mesh.pre_dc_read",
            lambda: cumesh_mesh.read(),
            default=(original_mesh_data.vertices, original_mesh_data.faces),
        )

        dc = self._oom_call_val(
            "remesh_narrow_band_dc",
            lambda: cumesh.remeshing.remesh_narrow_band_dc(
                rv,
                rf,
                center=center,
                scale=(resolution + 3 * params.remesh_band) / resolution * scale,
                resolution=resolution,
                band=params.remesh_band,
                project_back=params.remesh_project,
                verbose=False,
                bvh=original_mesh_data.bvh,
            ),
            default=None,
        )
        if dc is None:
            fb = self._oom_call_val(
                "remesh_mesh.fallback_after_dc_oom",
                lambda: cumesh_mesh.read(),
                default=(original_mesh_data.vertices, original_mesh_data.faces),
            )
            vertices, faces = fb
            logger.debug(f"Remesh skipped after OOM; using {vertices.shape[0]} vertices")
            return MeshData(vertices=vertices, faces=faces)

        vertices, faces = dc
        cumesh_mesh.init(vertices, faces)
        logger.debug(f"After remeshing: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")
        
        self._oom_call_void(
            "remesh_mesh.simplify",
            lambda: cumesh_mesh.simplify(params.decimation_target, verbose=False),
        )
        logger.debug(f"After simplifying: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

        vertices, faces = self._oom_call_val(
            "remesh_mesh.final_read",
            lambda: cumesh_mesh.read(),
            default=(vertices, faces),
        )
        hole_count = count_boundary_loops(vertices, faces)
        logger.debug(f"Holes after remesh: {hole_count}")

        logger.debug(f"Done remeshing | Time: {time.time() - start_time:.2f}s")
        return MeshData(
            vertices=vertices,
            faces=faces
        )

    def _uv_unwrap_mesh(self, mesh_data: MeshData, params: GLBConverterParams) -> MeshData:
        """Perform UV unwrapping on the mesh."""
        # Create cumesh from current mesh data
        logger.debug("Starting UV unwrapping")
        start_time = time.time()
        
        cumesh_mesh = cumesh.CuMesh()
        cumesh_mesh.init(mesh_data.vertices, mesh_data.faces)
        
        xatlas_compute_charts_kwargs = {
            "max_chart_area": 1.0,
            "max_boundary_length": 2.0,
            "max_cost": 10.0,
            "normal_seam_weight": 5.0,
            "normal_deviation_weight": 1.0,
            "fix_winding": True
        }
        
        def _unwrap_full():
            out_vertices, out_faces, out_uvs, out_vmaps = cumesh_mesh.uv_unwrap(
                compute_charts_kwargs={
                    "threshold_cone_half_angle_rad": np.radians(
                        params.mesh_cluster_threshold_cone_half_angle
                    ),
                    "refine_iterations": params.mesh_cluster_refine_iterations,
                    "global_iterations": params.mesh_cluster_global_iterations,
                    "smooth_strength": params.mesh_cluster_smooth_strength,
                },
                xatlas_compute_charts_kwargs=xatlas_compute_charts_kwargs,
                return_vmaps=True,
                verbose=True,
            )
            out_vertices = out_vertices.to(self.device)
            out_faces = out_faces.to(self.device)
            out_uvs = out_uvs.to(self.device)
            out_vmaps = out_vmaps.to(self.device)
            cumesh_mesh.compute_vertex_normals()
            out_normals = cumesh_mesh.read_vertex_normals()[out_vmaps]
            return MeshData(
                vertices=out_vertices,
                faces=out_faces,
                vertex_normals=out_normals,
                uvs=out_uvs,
            )

        result = self._oom_call_val(
            "uv_unwrap",
            _unwrap_full,
            default=lambda: self._mesh_planar_uv_fallback(mesh_data),
        )
        logger.debug(f"Done UV unwrapping | Time: {time.time() - start_time:.2f}s")
        return result

    def _subdivide_mesh(self, mesh_data: MeshData, original_mesh_data: MeshDataWithAttributeGrid, params: GLBConverterParams) -> MeshData:
        """Subdivide mesh with uv data and optionally reproject vertices to original mesh surface."""

        def _subdivide():
            m = subdivide_egdes(mesh_data, iterations=params.subdivisions)
            if params.vertex_reproject > 0.0:
                m = map_vertices_positions(
                    m, original_mesh_data, weight=params.vertex_reproject, inplace=True
                )
            return sort_mesh(m, axes=(2, 1, 0))

        return self._oom_call_val("subdivide_mesh", _subdivide, default=mesh_data)

    def _rasterize_attributes(self, mesh_data: MeshData, original_mesh_data: MeshDataWithAttributeGrid, layout: Dict[str,slice], params: GLBConverterParams) -> Tuple[torch.Tensor, Dict[str,slice]]:
        """Rasterize the given attributes onto the mesh UVs."""
        logger.debug("Sampling attributes(Texture rasterization)")
        start_time = time.time()

        def _rasterize():
            rast_data = rasterize_mesh_data(
                mesh_data, params.texture_size, use_vertex_normals=True
            )
            logger.debug(
                f"Texture baking: sampling {rast_data.positions.shape[0]} valid pixels "
                f"out of {params.texture_size * params.texture_size}"
            )
            logger.debug(
                f"Attribute volume has {original_mesh_data.attrs.values.shape[0]} voxels"
            )
            rast_data = map_mesh_rasterization(
                rast_data, original_mesh_data, flip_vertex_normals=True
            )
            attributes = sample_grid_attributes(rast_data, original_mesh_data.attrs)
            return dilate_attributes(attributes, self.DILATION_KERNEL_SIZE)

        attrs = self._oom_call_val(
            "rasterize_attributes",
            _rasterize,
            default=lambda: torch.zeros(
                params.texture_size,
                params.texture_size,
                original_mesh_data.attrs.values.shape[-1],
                device=self.device,
                dtype=torch.float32,
            ),
        )

        logger.debug(f"Done attribute sampling | Time: {time.time() - start_time:.2f}s")
        
        return attrs, layout
    
    def _texture_postprocess(self, attributes: torch.Tensor, attr_layout: Dict, params: GLBConverterParams) -> Tuple[Image.Image, Image.Image]:
        """Post-process the rasterized attributes into final textures."""
        logger.debug("Finalizing mesh textures")
        start_time = time.time()

        def _finalize():
            base_color = attributes[..., attr_layout['base_color']]
            metallic = attributes[..., attr_layout['metallic']]
            roughness = attributes[..., attr_layout['roughness']]
            alpha = attributes[..., attr_layout['alpha']]
            occlusion_channel = torch.ones_like(metallic)
            alpha = alpha.pow(params.alpha_gamma)
            alpha_mode = params.alpha_mode
            if alpha_mode == AlphaMode.BLEND:
                alpha_mode = AlphaMode.OPAQUE if np.all(alpha == 255) else alpha_mode
            if alpha_mode == AlphaMode.DITHER:
                h, w = alpha.shape[:2]
                dither_pattern = torch.as_tensor(
                    DITHER_PATTERN[:h, :w, None], device=alpha.device
                )
                alpha = (alpha > dither_pattern).float()
                logger.debug(
                    f"Dithered alpha channel has {np.sum(alpha == 0)} transparent pixels "
                    f"out of {alpha.size} total pixels"
                )
            rgba = torch.cat([base_color, alpha], dim=-1).clamp(0, 1)
            orm = torch.cat([occlusion_channel, roughness, metallic], dim=-1).clamp(0, 1)
            base_color_texture = to_pil_image(rgba.permute(2, 0, 1).cpu())
            orm_texture = to_pil_image(orm.permute(2, 0, 1).cpu())
            return base_color_texture, orm_texture

        def _gray_textures():
            s = params.texture_size
            base = Image.new("RGBA", (s, s), (200, 200, 200, 255))
            orm = Image.new("RGB", (s, s), (255, 180, 128))
            return base, orm

        base_color_texture, orm_texture = self._oom_call_val(
            "texture_postprocess", _finalize, default=_gray_textures
        )
        
        logger.debug(f"Done finalizing mesh textures | Time: {time.time() - start_time:.2f}s")
        return base_color_texture, orm_texture

    def _create_textured_mesh(self, mesh_data: MeshData, base_color: Image.Image, orm_texture: Image.Image, params: GLBConverterParams) -> trimesh.Trimesh:
        """Create a textured trimesh mesh from the mesh data and textures."""
        
        logger.debug("Creating textured mesh")
        start_time = time.time()

        alpha_mode = params.alpha_mode
        alpha_mode = AlphaMode.MASK if alpha_mode is AlphaMode.DITHER else alpha_mode

        def _textured_trimesh():
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=base_color,
                baseColorFactor=np.array([1.0, 1.0, 1.0, 1.0]),
                metallicRoughnessTexture=orm_texture,
                roughnessFactor=1.0,
                metallicFactor=1.0,
                alphaMode=alpha_mode.value,
                alphaCutoff=alpha_mode.cutoff,
                doubleSided=bool(not params.remesh),
            )
            vertices_np = mesh_data.vertices.mul(params.rescale).cpu().numpy()
            faces_np = mesh_data.faces.cpu().numpy()
            uvs_np = mesh_data.uvs.cpu().numpy()
            normals_np = mesh_data.vertex_normals.cpu().numpy()
            vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
            normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
            uvs_np[:, 1] = 1 - uvs_np[:, 1]
            return trimesh.Trimesh(
                vertices=vertices_np,
                faces=faces_np,
                vertex_normals=normals_np,
                process=False,
                visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material),
            )

        def _unlit_trimesh():
            v = mesh_data.vertices.mul(params.rescale).detach().cpu().numpy()
            f = mesh_data.faces.detach().cpu().numpy()
            return trimesh.Trimesh(vertices=v, faces=f, process=False)

        def _degenerate_trimesh():
            self.logger.warning("Using degenerate placeholder mesh after repeated OOM during export.")
            return trimesh.Trimesh(
                vertices=np.zeros((3, 3), dtype=np.float64),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
                process=False,
            )

        textured_mesh = None
        for step, builder in (("textured", _textured_trimesh), ("unlit", _unlit_trimesh)):
            try:
                textured_mesh = builder()
                break
            except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, RuntimeError, MemoryError) as e:
                if not _is_oom(e):
                    raise
                self.logger.warning(f"OOM during create_textured_mesh ({step}): {e}")
                _oom_empty_cache()
        if textured_mesh is None:
            textured_mesh = _degenerate_trimesh()
        
        logger.debug(f"Done creating textured mesh | Time: {time.time() - start_time:.2f}s")
                
        return textured_mesh
