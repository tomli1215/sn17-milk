import base64
import asyncio
import io
import json
import re
import time
from typing import Optional, Tuple

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel
from PIL import Image

from logger_config import logger
from modules.utils import set_random_seed
from .prompting import IMAGE_EDIT_SYSTEM_PROMPT, IMAGE_EDIT_USER_PROMPT
from .settings import JudgeConfig


SYSTEM_PROMPT = """
You are a specialized 3D model evaluation system.
Analyze visual quality and prompt adherence with expert precision.
Always respond with valid JSON only."""

USER_PROMPT_IMAGE = """Does each 3D model match the image prompt?

Penalty 0-10:
0 = Perfect match
3 = Minor issues (slight shape differences, missing small details)
5 = Moderate issues (wrong style, significant details missing)
7 = Major issues (wrong category but related, e.g. chair vs stool)
10 = Completely wrong object

Output: {"penalty_1": <0-10>, "penalty_2": <0-10>, "issues": "<brief>"}"""


class JudgeResponse(BaseModel):
    penalty_1: int
    penalty_2: int
    issues: str


class DuelManager:
    def __init__(self, settings: JudgeConfig):
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url=settings.vllm_url,
            api_key=settings.vllm_api_key,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            ),
        )

    async def _call_vllm(
        self,
        prompt_b64: str,
        img1_b64: str,
        img2_b64: str,
        seed: int,
        *,
        log_penalty_map: str = "",
    ) -> JudgeResponse:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image prompt to generate 3D model:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{prompt_b64}"},
                    },
                    {"type": "text", "text": "First 3D model (4 different views):"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img1_b64}"},
                    },
                    {"type": "text", "text": "Second 3D model (4 different views):"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img2_b64}"},
                    },
                    {"type": "text", "text": USER_PROMPT_IMAGE},
                ],
            },
        ]

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "judge-response",
                "schema": JudgeResponse.model_json_schema(),
            },
        }

        try:
            completion = await self.client.chat.completions.create(
                model=self.settings.vllm_model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=32,
                response_format=response_format,
                seed=seed,
            )

            choice = completion.choices[0]
            finish_reason = choice.finish_reason
            if finish_reason == "length":
                logger.warning("vLLM response was truncated due to max_tokens limit")

            content = choice.message.content
            if not content:
                raise ValueError("Empty response from vLLM")
            content = content.strip()
            pmap = f" | {log_penalty_map}" if log_penalty_map else ""
            logger.info(f"vLLM mesh judge{pmap} (seed={seed}): {content}")

            # Parse penalties via regex (robust to truncated JSON)
            penalty_1_match = re.search(r'"penalty_1":\s*(\d+)', content)
            penalty_2_match = re.search(r'"penalty_2":\s*(\d+)', content)
            penalty_1 = int(penalty_1_match.group(1)) if penalty_1_match else 5
            penalty_2 = int(penalty_2_match.group(1)) if penalty_2_match else 5

            issues = ""
            if finish_reason != "length":
                try:
                    issues = json.loads(content)["issues"]
                except (json.JSONDecodeError, KeyError):
                    issues = "Incomplete JSON"

            return JudgeResponse(
                penalty_1=penalty_1, penalty_2=penalty_2, issues=issues
            )

        except Exception as e:
            logger.error(f"vLLM call failed: {e}")
            return JudgeResponse(penalty_1=5, penalty_2=5, issues="vLLM call failed")

    async def run_duel(
        self,
        prompt_bytes: bytes,
        img1_bytes: bytes,
        img2_bytes: bytes,
        seed: int,
        *,
        candidate_indices: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, str]:
        """
        Run a position-balanced duel between two candidate images.

        Args:
            prompt_bytes: Original prompt image as PNG bytes
            img1_bytes: First candidate rendered grid as PNG bytes
            img2_bytes: Second candidate rendered grid as PNG bytes
            seed: Random seed for reproducibility
            candidate_indices: If set ``(i, j)``, logs map penalties to those GLB candidate indices.

        Returns:
            Tuple of (winner_idx, issues):
                winner_idx: -1 if img1 wins, 1 if img2 wins
                issues: Human-readable issue summary
        """
        if not img1_bytes or not img2_bytes:
            logger.error("Invalid image bytes provided to judge")
            return -1, "Invalid input — defaulting to first candidate"

        prompt_b64 = base64.b64encode(prompt_bytes).decode("utf-8")
        render1_b64 = base64.b64encode(img1_bytes).decode("utf-8")
        render2_b64 = base64.b64encode(img2_bytes).decode("utf-8")

        ci = candidate_indices
        map_direct = (
            f"penalty_1→candidate_{ci[0]} penalty_2→candidate_{ci[1]}"
            if ci
            else "penalty_1→img1(first_mesh) penalty_2→img2(second_mesh)"
        )
        map_swapped = (
            f"penalty_1→candidate_{ci[1]} penalty_2→candidate_{ci[0]}"
            if ci
            else "penalty_1→img2(second_mesh) penalty_2→img1(first_mesh)"
        )

        logger.info("Running position-balanced VLLM mesh duel...")
        res_direct, res_swapped = await asyncio.gather(
            self._call_vllm(
                prompt_b64, render1_b64, render2_b64, seed, log_penalty_map=map_direct
            ),
            self._call_vllm(
                prompt_b64, render2_b64, render1_b64, seed, log_penalty_map=map_swapped
            ),
        )

        score1 = (res_direct.penalty_1 + res_swapped.penalty_2) / 2
        score2 = (res_swapped.penalty_1 + res_direct.penalty_2) / 2
        issues = f"Direct: {res_direct.issues} | Swapped: {res_swapped.issues}"

        if ci:
            i, j = ci
            w = i if score1 < score2 else j
            logger.info(
                f"Mesh duel [candidate {i} vs {j}] — candidate {i} score = "
                f"({res_direct.penalty_1} + {res_swapped.penalty_2}) / 2 = {score1:.2f}  |  "
                f"candidate {j} score = ({res_swapped.penalty_1} + {res_direct.penalty_2}) / 2 = {score2:.2f}  "
                f"| lower wins  |  winner: candidate {w}"
            )
        else:
            wside = "img1" if score1 < score2 else "img2"
            logger.info(
                f"Mesh duel — side img1 score = ({res_direct.penalty_1} + {res_swapped.penalty_2}) / 2 = {score1:.2f}  "
                f"| side img2 score = ({res_swapped.penalty_1} + {res_direct.penalty_2}) / 2 = {score2:.2f}  "
                f"| lower wins  |  winner: {wside}"
            )

        if score1 < score2:
            return -1, issues  # Image 1 wins (lower penalty = better)
        else:
            return 1, issues  # Image 2 wins

    @staticmethod
    def _pil_to_png_b64(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    async def _call_vllm_image_edit(
        self,
        original_b64: str,
        edited1_b64: str,
        edited2_b64: str,
        seed: int,
        *,
        log_slot_order: str = "",
        log_penalty_map: str = "",
    ) -> JudgeResponse:
        messages = [
            {"role": "system", "content": IMAGE_EDIT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Original image:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{original_b64}"},
                    },
                    {"type": "text", "text": "First edited image:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{edited1_b64}"},
                    },
                    {"type": "text", "text": "Second edited image:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{edited2_b64}"},
                    },
                    {"type": "text", "text": IMAGE_EDIT_USER_PROMPT},
                ],
            },
        ]

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "judge-response",
                "schema": JudgeResponse.model_json_schema(),
            },
        }

        set_random_seed(seed)

        try:
            completion = await self.client.chat.completions.create(
                model=self.settings.vllm_model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=32,
                response_format=response_format,
                seed=seed,
            )

            choice = completion.choices[0]
            finish_reason = choice.finish_reason
            if finish_reason == "length":
                logger.warning("vLLM image-edit judge response was truncated due to max_tokens limit")

            content = choice.message.content
            if not content:
                raise ValueError("Empty response from vLLM")

            content = content.strip()
            slot = f" [{log_slot_order}]" if log_slot_order else ""
            pmap = f" [{log_penalty_map}]" if log_penalty_map else ""
            logger.info(f"vLLM image-edit judge{slot}{pmap} (seed={seed}): {content}")

            penalty_1_match = re.search(r'"penalty_1":\s*(\d+)', content)
            penalty_2_match = re.search(r'"penalty_2":\s*(\d+)', content)
            penalty_1 = int(penalty_1_match.group(1)) if penalty_1_match else 5
            penalty_2 = int(penalty_2_match.group(1)) if penalty_2_match else 5

            issues = ""
            if finish_reason != "length":
                try:
                    issues = json.loads(content)["issues"]
                except (json.JSONDecodeError, KeyError):
                    issues = "Incomplete JSON"

            return JudgeResponse(
                penalty_1=penalty_1, penalty_2=penalty_2, issues=issues
            )

        except Exception as e:
            logger.error(f"vLLM image-edit judge call failed: {e}")
            return JudgeResponse(penalty_1=5, penalty_2=5, issues="vLLM image-edit judge call failed")

    async def run_image_duel(
        self,
        original: Image.Image,
        edited1: Image.Image,
        edited2: Image.Image,
        seed: int,
        *,
        candidate_indices: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, str]:
        """
        Position-balanced duel between two edited images for identity preservation vs original.

        Args:
            candidate_indices: If ``(a, b)``, ``edited1`` is Qwen candidate ``a`` and ``edited2`` is candidate ``b``
                (for clear score logs).

        Returns:
            winner_idx: -1 if edited1 wins, 1 if edited2 wins.
        """
        orig_b64 = self._pil_to_png_b64(original)
        e1_b64 = self._pil_to_png_b64(edited1)
        e2_b64 = self._pil_to_png_b64(edited2)

        ci = candidate_indices
        map_direct = (
            f"penalty_1→candidate_{ci[0]} penalty_2→candidate_{ci[1]}"
            if ci
            else "penalty_1→edited1 penalty_2→edited2"
        )
        map_swapped = (
            f"penalty_1→candidate_{ci[1]} penalty_2→candidate_{ci[0]}"
            if ci
            else "penalty_1→edited2 penalty_2→edited1"
        )

        duel_label = f" [candidate {ci[0]} vs {ci[1]}]" if ci else ""
        logger.info(f"Running position-balanced VLLM image-edit duel{duel_label}...")
        res_direct, res_swapped = await asyncio.gather(
            self._call_vllm_image_edit(
                orig_b64,
                e1_b64,
                e2_b64,
                seed,
                log_slot_order="slot1=edited1 slot2=edited2",
                log_penalty_map=map_direct,
            ),
            self._call_vllm_image_edit(
                orig_b64,
                e2_b64,
                e1_b64,
                seed,
                log_slot_order="slot1=edited2 slot2=edited1",
                log_penalty_map=map_swapped,
            ),
        )

        score1 = (res_direct.penalty_1 + res_swapped.penalty_2) / 2
        score2 = (res_swapped.penalty_1 + res_direct.penalty_2) / 2
        issues = (
            f"| Direct: {res_direct.issues} | Swapped: {res_swapped.issues}"
            if res_direct.issues or res_swapped.issues
            else ""
        )

        winner = -1 if score1 < score2 else 1

        if ci:
            a, b = ci
            win_idx = a if winner == -1 else b
            logger.info(
                f"Qwen-edit duel scores — candidate {a} = (direct.p1={res_direct.penalty_1} + "
                f"swapped.p2={res_swapped.penalty_2}) / 2 = {score1:.2f}  |  "
                f"candidate {b} = (swapped.p1={res_swapped.penalty_1} + "
                f"direct.p2={res_direct.penalty_2}) / 2 = {score2:.2f}  |  "
                f"lower is better  |  winner: candidate {win_idx}"
            )
        else:
            wname = "edited1" if winner == -1 else "edited2"
            logger.info(
                f"Qwen-edit duel scores — edited1 = (direct.p1={res_direct.penalty_1} + "
                f"swapped.p2={res_swapped.penalty_2}) / 2 = {score1:.2f}  |  "
                f"edited2 = (swapped.p1={res_swapped.penalty_1} + "
                f"direct.p2={res_direct.penalty_2}) / 2 = {score2:.2f}  |  winner: {wname}"
            )

        return winner, issues

    async def judge_edited_images(
        self,
        representatives: list[Image.Image],
        original: Image.Image,
        seed: int,
    ) -> int:
        """
        Tournament: pick the best among several edited images (same role as v1.2 ImageJudgeInput).

        Uses the first image in each candidate set as the representative (first view in multiview).

        Returns:
            Index of the winning candidate in ``representatives``.
        """
        if len(representatives) < 2:
            logger.warning("Less than 2 edited-image candidates to judge; using index 0")
            return 0

        t1 = time.time()
        logger.info(f"Judging {len(representatives)} Qwen edit candidates for identity preservation (vLLM)")

        orig = original.copy()
        ref = representatives[0]
        if orig.size != ref.size:
            orig = orig.resize(ref.size, Image.Resampling.LANCZOS)

        best_idx = 0
        for i in range(1, len(representatives)):
            winner, issues = await self.run_image_duel(
                orig,
                representatives[best_idx],
                representatives[i],
                seed,
                candidate_indices=(best_idx, i),
            )
            w = i if winner == 1 else best_idx
            logger.info(
                f"Qwen edit bracket step [{best_idx} vs {i}] → advancing candidate {w} {issues}"
            )
            if winner == 1:
                best_idx = i

        logger.success(
            f"Qwen edit judging took {time.time() - t1:.2f}s | Winner candidate index: {best_idx}"
        )
        return best_idx
