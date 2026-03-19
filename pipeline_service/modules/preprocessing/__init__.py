"""
Trellis preprocessing: ``trellis_params_resolve``, ``vllm_trellis_pipeline_router``.

Import from submodules so YAML-only code paths do not load VLM (httpx/openai)::

    from modules.preprocessing.trellis_params_resolve import resolve_trellis_params_after_rmbg
"""
