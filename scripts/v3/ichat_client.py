# -*- coding: utf-8 -*-
"""ichat_client.py — 通用 OpenAI 兼容多模态聊天客户端 (LLM 调用统一封装).

本模块为数据合成脚本 (generate_v5 / generate_v6 / generate_*_latent_cot 等)
统一提供:
  - 标准 OpenAI / OpenAI 兼容网关 (vLLM / OneAPI / FastChat 等) 的 chat 调用
  - 多模态 (图像 + 文本) 消息组装
  - 指数退避重试
  - JSON 输出鲁棒解析
  - 长跑任务的中间结果原子落盘

公开 API (调用方依赖, 保持向后兼容):
  - create_openai_client()                       -> openai.OpenAI
  - encode_image_to_base64(path)                 -> str | None
  - call_chat_with_image(client, model, ...)     -> str (assistant content) | None
  - parse_gpt_response(raw)                      -> dict | None
  - build_auth_config_from_env_or_args(...)      -> dict | None  (兼容用, 已不再需要鉴权)
  - _save_intermediate(records, output)          -> None

环境变量:
  OPENAI_API_KEY    : OpenAI / 兼容服务的 API Key (必需)
  OPENAI_BASE_URL   : OpenAI 兼容服务 base_url (可选, 默认 https://api.openai.com/v1)
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ichat_client")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                      datefmt="%H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 300


# ============================================================
# Auth (保留函数签名以兼容旧调用; 标准 OpenAI 走 OPENAI_API_KEY)
# ============================================================

def build_auth_config_from_env_or_args(
    source: Optional[str] = None,
    appid: Optional[str] = None,
    appkey: Optional[str] = None,
    rtx: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """兼容性占位: 现版本不再需要 HMAC 鉴权, 直接读取 OPENAI_API_KEY.

    返回 dict (字段对旧调用方透明), 失败返回 None.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("[auth] missing OPENAI_API_KEY in environment")
        return None
    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    cfg = {
        "api_key":  api_key,
        "base_url": base_url,
        # 兼容旧 key (调用方读 cfg.get("rtx") 等不会崩)
        "source":   source or "",
        "appid":    appid or "",
        "appkey":   appkey or "",
        "rtx":      rtx or "",
    }
    logger.info("[auth] OK base_url=%s", base_url)
    return cfg


# ============================================================
# OpenAI client
# ============================================================

def create_openai_client():
    """创建 openai.OpenAI 客户端. api_key/base_url 来自环境变量."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai package not installed. pip install 'openai>=1.10'"
        ) from e
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url, timeout=DEFAULT_TIMEOUT)


# ============================================================
# 图像编码
# ============================================================

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """读图 -> base64 字符串 (不带 data: 前缀). 失败返回 None."""
    if not image_path or not os.path.exists(image_path):
        logger.warning("[encode_image] not found: %s", image_path)
        return None
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    except Exception as e:
        logger.warning("[encode_image] %s: %s", image_path, e)
        return None


def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    if not mt or not mt.startswith("image/"):
        return "image/jpeg"
    return mt


# ============================================================
# 主入口: call_chat_with_image
# ============================================================

_RETRYABLE_PATTERNS = (
    "timeout", "timed out", "connection", "rate", "429",
    "500", "502", "503", "504", "overloaded", "temporarily",
    "internal server error", "bad gateway", "service unavailable",
    "gateway timeout",
)


def _is_retryable(err: Exception) -> bool:
    s = (str(err) or "").lower()
    return any(p in s for p in _RETRYABLE_PATTERNS)


def call_chat_with_image(
    *,
    client,
    model: str,
    system_prompt: str,
    user_text: str,
    image_base64: Optional[str],
    auth_config: Optional[Dict[str, str]] = None,
    max_retries: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    response_format_json: bool = False,
    image_detail: str = "high",
    image_mime: str = "image/jpeg",
    extra_user_images: Optional[List[str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[str]:
    """单次 chat.completions.create 调用 (system + user[image+text]).

    返回 assistant message content (str), 失败返回 None. 已带:
      - 指数退避重试 (上限 max_retries)
      - response_format JSON 强约束 (兼容服务需支持 type=json_object)

    auth_config 现为可选 (向后兼容), 若提供且包含 base_url/api_key, 不会改变 client.
    """
    user_content: List[Dict[str, Any]] = []
    if image_base64:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{image_mime};base64,{image_base64}",
                "detail": image_detail,
            },
        })
    if extra_user_images:
        for p in extra_user_images:
            b64 = encode_image_to_base64(p)
            if not b64:
                continue
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{_guess_mime(p)};base64,{b64}",
                    "detail": image_detail,
                },
            })
    user_content.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model":       model,
                "messages":    messages,
                "temperature": temperature,
                "max_tokens":  max_tokens,
                "timeout":     timeout,
            }
            if response_format_json:
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            try:
                content = resp.choices[0].message.content
            except Exception as e:
                last_err = e
                logger.warning("[chat] attempt %d: malformed resp: %s", attempt, e)
                content = None
            if content:
                return content
            last_err = RuntimeError("empty assistant content")
        except Exception as e:
            last_err = e
            retryable = _is_retryable(e)
            logger.warning(
                "[chat] attempt %d/%d %s: %s",
                attempt, max_retries,
                "RETRYABLE" if retryable else "NON-RETRYABLE",
                str(e)[:300],
            )
            if not retryable and attempt > 1:
                break

        if attempt < max_retries:
            backoff = min(60.0, 2.0 * (2 ** (attempt - 1))) + random.uniform(0, 1.5)
            time.sleep(backoff)

    logger.error("[chat] all %d retries exhausted; last_err=%s",
                 max_retries, str(last_err)[:300] if last_err else "?")
    return None


# ============================================================
# 解析 LLM 输出为 dict
# ============================================================

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$", re.MULTILINE)


def parse_gpt_response(raw: str) -> Optional[Dict[str, Any]]:
    """把 LLM raw 文本鲁棒地 parse 成 dict.

    流程:
      1) strip + 去 ```json fence
      2) json.loads 整段
      3) 失败则正则抓最外层 { ... } 再 loads
      4) 仍失败则尝试简单修复 (尾部多余逗号)
      5) 全失败返回 None
    """
    if not raw or not isinstance(raw, str):
        return None
    txt = raw.strip()
    txt = _FENCE_RE.sub("", txt).strip()
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        sub = m.group(0)
        try:
            obj = json.loads(sub)
            return obj if isinstance(obj, dict) else None
        except Exception:
            try:
                fixed = re.sub(r",(\s*[}\]])", r"\1", sub)
                obj = json.loads(fixed)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass
    logger.warning("[parse] cannot parse json (len=%d, head=%r)", len(txt), txt[:120])
    return None


# ============================================================
# 中间结果落盘
# ============================================================

def _save_intermediate(records: List[Dict], output_path: str) -> None:
    """原子写中间快照: list[dict] -> output_path."""
    if not output_path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        tmp = output_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        os.replace(tmp, output_path)
        logger.info("[save] %s (%d records)", output_path, len(records))
    except Exception as e:
        logger.warning("[save] failed %s: %s", output_path, e)


# ============================================================
# 自检
# ============================================================

def _selfcheck():
    cfg = build_auth_config_from_env_or_args()
    if not cfg:
        print("[selfcheck] auth NOT configured; set OPENAI_API_KEY (and optionally OPENAI_BASE_URL)")
        return 1
    try:
        c = create_openai_client()
        print("[selfcheck] OpenAI client OK ->", c.base_url)
    except Exception as e:
        print("[selfcheck] OpenAI client FAIL:", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
