from pydantic import BaseModel


class JudgeConfig(BaseModel):
    """VLLM Judge configuration"""
    vllm_url: str = "http://localhost:8095/v1"
    vllm_api_key: str = "local"
    vllm_model_name: str = "THUDM/GLM-4.1V-9B-Thinking"
    enabled: bool = True
    # After Qwen: generate this many edits with seeds base, base+1, … and pick best via vLLM (identity).
    qwen_edit_candidate_count: int = 3
    pick_best_qwen_edit_via_vllm: bool = True
