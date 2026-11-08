"""AAEW：拒答感知预警的评估核心（纯函数，无 I/O）。

设计目标
--------
1. 可比性：不同数据集正例率差异大（10.2% vs 24.4%），只比"错误率"会误导。
   因此同时给出 **标准错误率** 与 **平衡错误率**（0.5·(FNR+FPR)），后者对患病率不敏感。
2. 有不确定性：AURC 与 risk@覆盖率 给自助法 95% CI，并与"随机弃权"基线做配对检验。
3. 排序无偏：置信度并列（ties）时按并列块做线性插值，消除样本顺序带来的随机差异。
4. 退化输入有定义：样本过少、单一类别、概率非有限，返回 status 而不抛异常。
5. 可扩展：大样本提供直方图近似路径，并如实返回与精确解的实测偏差。

约定
----
- p：正类（辍学）概率；y：0/1 真值；conf = max(p, 1-p)
- coverage：已回答样本比例；risk：已回答样本中的错误率
- AURC：风险-覆盖曲线下面积（固定 101 点网格，梯形积分）
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRID_POINTS = 101
MIN_N = 30


@dataclass
class AAEWResult:
    status: str
    n: int
    prevalence: float = float("nan")
    base_error: float = float("nan")
    base_balanced_error: float = float("nan")
    aurc: float = float("nan")
    aurc_balanced: float = float("nan")
    aurc_oracle: float = float("nan")
    e_aurc: float = float("nan")
    aurc_ci95: tuple = (float("nan"), float("nan"))
    risk_at: dict = field(default_factory=dict)
    balanced_risk_at: dict = field(default_factory=dict)
    vs_random_p: float = float("nan")
    group_aurc: dict = field(default_factory=dict)
    band: dict = field(default_factory=dict)
    fast_path: bool = False
    curve: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def _status(p: np.ndarray, y: np.ndarray) -> str:
    if p.shape != y.shape:
        return "shape_mismatch"
    if p.size < MIN_N:
        return "too_few_samples"
    if np.unique(y).size < 2:
        return "single_class"
    if not np.all(np.isfinite(p)):
        return "non_finite_probability"
    return "ok"


def _block_bounds(conf_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回每个位置所属并列块的 [start, end)，end 为开区间。"""
    n = len(conf_sorted)
    start = np.zeros(n, dtype=int)
    end = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and conf_sorted[j + 1] == conf_sorted[i]:
            j += 1
        start[i:j + 1] = i
        end[i:j + 1] = j + 1
        i = j + 1
    return start, end


def risk_curve_from_pred(conf: np.ndarray, y: np.ndarray, pred: np.ndarray,
                         grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """标准风险与平衡风险随覆盖率变化；并列块内按块均值线性插值。"""
    order = np.argsort(-conf, kind="mergesort")
    cs, ys, ps = conf[order], y[order], pred[order]
    n = len(y)
    err = (ps != ys).astype(float)
    fn = ((ys == 1) & (ps == 0)).astype(float)
    fp = ((ys == 0) & (ps == 1)).astype(float)
    cum_e, cum_fn, cum_fp = np.cumsum(err), np.cumsum(fn), np.cumsum(fp)
    blk_s, blk_e = _block_bounds(cs)

    pos_total = max(1.0, float((ys == 1).sum()))
    neg_total = max(1.0, float((ys == 0).sum()))

    risk = np.empty_like(grid)
    bal = np.empty_like(grid)
    for gi, cov in enumerate(grid):
        k = float(cov) * n
        j = min(n - 1, max(0, int(np.ceil(k)) - 1))
        s, e = int(blk_s[j]), int(blk_e[j])
        before = float(cum_e[s - 1]) if s > 0 else 0.0
        before_fn = float(cum_fn[s - 1]) if s > 0 else 0.0
        before_fp = float(cum_fp[s - 1]) if s > 0 else 0.0
        size = e - s
        take = min(max(k - s, 0.0), float(size))
        rate_e = (float(cum_e[e - 1]) - before) / size
        rate_fn = (float(cum_fn[e - 1]) - before_fn) / size
        rate_fp = (float(cum_fp[e - 1]) - before_fp) / size
        kk = max(s + take, 1e-9)
        risk[gi] = (before + rate_e * take) / kk
        bal[gi] = 0.5 * ((before_fn + rate_fn * take) / pos_total
                         + (before_fp + rate_fp * take) / neg_total) * (n / kk)
    return risk, bal


def fit_temperature(p: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> float:
    """在给定样本上拟合温度 T（最小化 NLL）。T>1 表示原概率过度自信。"""
    p = np.clip(np.asarray(p, dtype=float).ravel(), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float).ravel()
    logit = np.log(p / (1 - p))
    cand = np.linspace(0.5, 5.0, 181) if grid is None else grid
    best_t, best_nll = 1.0, float("inf")
    for T in cand:
        q = 1.0 / (1.0 + np.exp(-logit / T))
        q = np.clip(q, 1e-6, 1 - 1e-6)
        nll = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
        if nll < best_nll:
            best_t, best_nll = float(T), nll
    return round(best_t, 4)


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float).ravel(), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-logit / T))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """期望校准误差：按置信度分箱，|acc - conf| 的样本加权平均。"""
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total, n = 0.0, len(y)
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= edges[i + 1])
        if not m.any():
            continue
        total += abs((pred[m] == y[m]).mean() - conf[m].mean()) * m.sum() / n
    return round(float(total), 4)


def _trapz(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def evaluate(p, y, groups=None, n_boot: int = 300, seed: int = 20261108,
             grid: np.ndarray | None = None, band: bool = True,
             fast: str = "auto", n_fast: int = 20000) -> AAEWResult:
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    st = _status(p, y)
    if st != "ok":
        return AAEWResult(status=st, n=int(p.size), notes=[st])

    pred = (p >= 0.5).astype(float)
    conf = np.maximum(p, 1 - p)
    g = np.linspace(0, 1, GRID_POINTS) if grid is None else grid

    risk, bal = risk_curve_from_pred(conf, y, pred, g)
    aurc, aurc_bal = _trapz(g, risk), _trapz(g, bal)
    # 完美排序（先答对的）
    _, oracle_bal = risk_curve_from_pred(1.0 - (pred != y), y, pred, g)
    aurc_or = _trapz(g, oracle_bal)

    base_err = float((pred != y).mean())
    base_bal = 0.5 * (((pred == 0) & (y == 1)).sum() / max(1.0, (y == 1).sum())
                      + ((pred == 1) & (y == 0)).sum() / max(1.0, (y == 0).sum()))

    rng = np.random.default_rng(seed)
    fast_used, boot_pool = False, np.arange(len(y))
    if fast is True or (fast == "auto" and len(y) > n_fast):
        # 大样本：自助只在 n_fast 规模的子样本上进行，点估计仍用全量
        boot_pool = rng.choice(len(y), size=min(int(n_fast), len(y)), replace=False)
        fast_used = True
    boots, diffs, curves = [], [], []
    for _ in range(n_boot):
        b = boot_pool[rng.integers(0, len(boot_pool), len(boot_pool))]
        if np.unique(y[b]).size < 2:
            continue
        pb, yb = p[b], y[b]
        pr = (pb >= 0.5).astype(float)
        rb, bb = risk_curve_from_pred(np.maximum(pb, 1 - pb), yb, pr, g)
        ab = _trapz(g, rb)
        boots.append(ab)
        if band:
            curves.append((rb, bb))
        diffs.append(base_err - ab)          # >0 表示 AAEW 优于随机弃权
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (float("nan"),) * 2
    p_val = float(np.mean(np.asarray(diffs) <= 0)) if diffs else float("nan")

    def at(c: float, arr: np.ndarray) -> float:
        return round(float(np.interp(c, g, arr)), 4)

    band_out = {}
    if band and curves:
        arr = np.vstack([c[0] for c in curves])
        arrb = np.vstack([c[1] for c in curves])
        band_out = {"grid": [round(float(v), 4) for v in g],
                    "risk_lower": [round(float(v), 4) for v in np.percentile(arr, 2.5, axis=0)],
                    "risk_upper": [round(float(v), 4) for v in np.percentile(arr, 97.5, axis=0)],
                    "balanced_lower": [round(float(v), 4) for v in np.percentile(arrb, 2.5, axis=0)],
                    "balanced_upper": [round(float(v), 4) for v in np.percentile(arrb, 97.5, axis=0)],
                    "n_boot": len(curves)}

    group_aurc = {}
    if groups is not None:
        groups = np.asarray(groups).ravel()
        for gu in np.unique(groups):
            m = groups == gu
            if m.sum() < MIN_N or np.unique(y[m]).size < 2:
                continue
            rg, _ = risk_curve_from_pred(conf[m], y[m], pred[m], g)
            group_aurc[str(gu)] = round(_trapz(g, rg), 4)

    return AAEWResult(
        status="ok", n=int(len(y)), prevalence=round(float(y.mean()), 4),
        base_error=round(base_err, 4), base_balanced_error=round(base_bal, 4),
        aurc=round(aurc, 4), aurc_balanced=round(aurc_bal, 4),
        aurc_oracle=round(aurc_or, 4), e_aurc=round(aurc_bal - aurc_or, 4),
        aurc_ci95=(round(ci[0], 4), round(ci[1], 4)),
        risk_at={f"{int(c*100)}%": at(c, risk) for c in (0.25, 0.5, 0.8, 1.0)},
        balanced_risk_at={f"{int(c*100)}%": at(c, bal) for c in (0.25, 0.5, 0.8, 1.0)},
        vs_random_p=round(p_val, 4), group_aurc=group_aurc, band=band_out,
        fast_path=fast_used,
        curve={"grid": [round(float(v), 4) for v in g],
               "risk": [round(float(v), 4) for v in risk],
               "balanced_risk": [round(float(v), 4) for v in bal]},
    )


def aurc_approx(p, y, bins: int = 512) -> tuple[float, float]:
    """直方图近似（O(n+bins)），返回 (近似 AURC, 与精确解偏差)。"""
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    pred = (p >= 0.5).astype(float)
    conf = np.maximum(p, 1 - p)
    err = (pred != y).astype(float)
    cnt, edges = np.histogram(conf, bins=bins)
    e_sum, _ = np.histogram(conf, bins=bins, weights=err)
    order = np.argsort(edges[:-1])[::-1]
    cnt, e_sum = cnt[order].astype(float), e_sum[order]
    cum_c, cum_e = np.cumsum(cnt), np.cumsum(e_sum)
    with np.errstate(invalid="ignore", divide="ignore"):
        risk = cum_e / cum_c
    cov = cum_c / max(1.0, cum_c[-1])
    approx = _trapz(cov, risk)
    exact = evaluate(p, y, n_boot=0).aurc
    return float(approx), float(abs(approx - exact))
