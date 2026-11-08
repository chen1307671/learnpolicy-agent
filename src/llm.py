"""可选的大模型解释生成适配器（OpenAI 兼容接口）。

设计原则：
1. **只生成解释，不参与决策**：不改风险概率、不改干预排序、不改判定。
2. **默认关闭**：未设置 LLM_API_KEY 时返回模板文本，离线可跑。
3. **失败降级**：超时/网络错误/返回异常一律回退模板，不抛异常。
4. **Key 只读环境变量**，绝不写入代码或仓库。

环境变量：
    LLM_API_KEY   必填，缺失即视为未启用
    LLM_BASE_URL  必填，如 https://open.bigmodel.cn/api/paas/v4
    LLM_MODEL     必填，如 glm-4-flash
    LLM_TIMEOUT   可选，默认 8 秒
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

TEMPLATE_TAIL = "（本段为规则模板生成，未调用大模型。）"


def enabled() -> bool:
    return bool(os.environ.get("LLM_API_KEY") and os.environ.get("LLM_BASE_URL")
                and os.environ.get("LLM_MODEL"))


def _template(evidence: dict) -> str:
    """无 Key 或调用失败时的确定性文本，内容全部来自结构化证据。"""
    p = evidence.get("p_dropout")
    lvl = evidence.get("risk_level", "")
    feat = evidence.get("top_feature", "")
    delta = evidence.get("delta")
    n = evidence.get("cohort_n")
    rate = evidence.get("cohort_rate")
    parts = [f"该生辍学风险 {p:.1%}（{lvl}）。"]
    if feat:
        parts.append(f"主要可干预项为「{feat}」。")
    if delta is not None:
        parts.append(f"有界扰动后风险变化 {delta:+.1%}。")
    if n and rate is not None:
        parts.append(f"同专业同水平 {n} 人的实际辍学率为 {rate:.1%}。")
    return "".join(parts) + TEMPLATE_TAIL


def _prompt(evidence: dict) -> str:
    return (
        "你是学业支持助手。只根据下列已核验的结构化证据，用两句话向辅导员说明风险与下一步，"
        "不得引入任何未给出的数字，不得下结论式处分建议。\n"
        f"证据：{json.dumps(evidence, ensure_ascii=False)}"
    )


def generate(evidence: dict) -> dict:
    """返回 {source, text, latency_ms, error?}；source ∈ api / template / fallback。"""
    import time
    t0 = time.perf_counter()
    if not enabled():
        return {"source": "template", "text": _template(evidence),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    base = os.environ["LLM_BASE_URL"].rstrip("/")
    payload = json.dumps({
        "model": os.environ["LLM_MODEL"],
        "messages": [{"role": "user", "content": _prompt(evidence)}],
        "temperature": 0.2,
        "max_tokens": 220,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions", data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['LLM_API_KEY']}"})
    try:
        timeout = float(os.environ.get("LLM_TIMEOUT", "8"))
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8"))
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        if not text:
            raise ValueError("empty completion")
        return {"source": "api", "text": text,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "model": os.environ["LLM_MODEL"]}
    except Exception as exc:  # noqa: BLE001 - 任何异常都必须降级，不能让演示挂掉
        return {"source": "fallback", "text": _template(evidence),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
