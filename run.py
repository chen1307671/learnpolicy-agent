"""学策 LearnPolicy Agent · 自检 / 评测 / 演示入口

    python run.py --selftest   数据校验 + 模型评测 + 消融 + Agent 基线对比 + 三条提示词
    python run.py --serve      启动可登录 Web 演示
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from core import PROMPTS, build  # noqa: E402
from server import init_state, serve  # noqa: E402

OUT = HERE / "reports"
OUT.mkdir(exist_ok=True)


def group_of(name: str) -> str:
    if name.startswith("Curricular units 1st sem"):
        return "第一学期学业表现"
    if name.startswith("Curricular units 2nd sem"):
        return "第二学期学业表现"
    if name in ("Unemployment rate", "Inflation rate", "GDP"):
        return "宏观经济"
    if name in ("Debtor", "Tuition fees up to date", "Scholarship holder",
                "Daytime/evening attendance"):
        return "就读与资助状态"
    if name in ("Marital status", "Gender", "Age at enrollment", "Nacionality",
                "Displaced", "International", "Educational special needs",
                "Mother's qualification", "Father's qualification",
                "Mother's occupation", "Father's occupation"):
        return "人口与家庭背景"
    return "入学与专业信息"


def bootstrap_ci(d, m, n_boot=300):
    X, yb = d.X[d.test_idx], d.y_bin[d.test_idx]
    p = m.p_dropout(X)
    rng = np.random.default_rng(20261107)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yb), len(yb))
        if len(set(yb[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(yb[idx], p[idx]))
    return {"n_boot": n_boot, "auc_dropout_mean": round(float(np.mean(vals)), 4),
            "ci95_low": round(float(np.percentile(vals, 2.5)), 4),
            "ci95_high": round(float(np.percentile(vals, 97.5)), 4),
            "method": "行级 bootstrap（不考虑专业聚类）"}


def group_bootstrap(d, m, n_boot=300):
    """按 Course 整组重采样的 bootstrap：考虑专业内聚类，比行级更保守。"""
    te = d.test_idx
    p = m.p_dropout(d.X[te])
    yb = d.y_bin[te]
    g = d.groups[te]
    uniq = np.unique(g)
    rng = np.random.default_rng(20261107)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(g == x)[0] for x in pick])
        if len(set(yb[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(yb[idx], p[idx]))
    return {"n_boot": n_boot, "n_groups": int(len(uniq)),
            "auc_dropout_mean": round(float(np.mean(vals)), 4),
            "ci95_low": round(float(np.percentile(vals, 2.5)), 4),
            "ci95_high": round(float(np.percentile(vals, 97.5)), 4),
            "method": "按 Course 整组重采样"}


def _fit_lr_auc(X, y, yb, tr, te, seed=20261107):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xtr, Xte = X[tr], X[te]
    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
    lr.fit(sc.transform(Xtr), y[tr])
    col = list(lr.classes_).index("Dropout")
    return round(float(roc_auc_score(yb[te], lr.predict_proba(sc.transform(Xte))[:, col])), 4)


def early_warning(d):
    """同一划分、同一模型族下，比较不同信息可用时点的判别力（回应时间泄漏质疑）。"""
    names = d.feature_names
    def mask(pred):
        return np.array([pred(k) for k in names])
    sets = {
        "入学信息（无学期成绩）": mask(lambda k: not k.startswith("Curricular units")),
        "入学信息 + 第一学期": mask(lambda k: not k.startswith("Curricular units 2nd sem")),
        "全部特征（含第二学期）": mask(lambda k: True),
    }
    out = {k: _fit_lr_auc(d.X[:, v], d.y, d.y_bin, d.train_idx, d.test_idx)
           for k, v in sets.items()}
    return {"auc_by_feature_set": out,
            "n_features": {k: int(v.sum()) for k, v in sets.items()},
            "note": "第二学期特征属学年末信息；同时点对比用于说明早预警口径的上限，不替代主指标。"}


def _balanced_group_test(groups, frac, seed):
    """按专业整体入样、并按样本量凑到目标比例的分组切分（复跑稳定性用）。"""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    target = len(groups) * frac
    picked, acc = [], 0
    for g in uniq:
        if acc >= target:
            break
        n = int((groups == g).sum())
        if acc + n <= target * 1.35 or len(picked) < 3:
            picked.append(g)
            acc += n
    if not picked:
        picked = [uniq[0]]
    return np.isin(groups, picked)


def split_robustness(d, n_seeds=5):
    aucs, fracs = [], []
    for s in range(n_seeds):
        te = _balanced_group_test(d.groups, 0.30, 1000 + s)
        tr = ~te
        aucs.append(_fit_lr_auc(d.X, d.y, d.y_bin, tr, te, seed=20261107 + s))
        fracs.append(round(float(te.mean()), 3))
    return {"n_seeds": n_seeds, "split": "样本量均衡的分组切分（按专业整体入样）",
            "auc_mean": round(float(np.mean(aucs)), 4),
            "auc_std": round(float(np.std(aucs)), 4),
            "auc_min": min(aucs), "auc_max": max(aucs),
            "test_row_fraction": fracs, "aucs": aucs,
            "note": "更换种子重跑分组划分，检验指标是否依赖单次抽签；主指标仍用固定划分以便复现。"}


def ablation(d, m, n_boot=1):
    """按真实字段语义分组做特征消融；每组只重训 LR（与基线同口径）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    base = m.evaluate()["auc_dropout_lr"]
    groups = sorted({group_of(k) for k in d.feature_names})
    out = {}
    for g in groups:
        keep = [i for i, k in enumerate(d.feature_names) if group_of(k) != g]
        X = d.X[:, keep]
        sc = StandardScaler().fit(X[d.train_idx])
        Xtr, Xte = sc.transform(X[d.train_idx]), sc.transform(X[d.test_idx])
        lr = LogisticRegression(max_iter=2000, random_state=20261107)
        lr.fit(Xtr, d.y[d.train_idx])
        col = list(lr.classes_).index("Dropout")
        auc = round(float(roc_auc_score(
            d.y_bin[d.test_idx], lr.predict_proba(Xte)[:, col])), 4)
        out[g] = {"auc_without": auc, "auc_drop": round(base - auc, 4)}
    return out, groups


class LearnSubset:
    """保持 LearnData 接口的特征子集视图，用于消融，不改动原始数据。"""
    def __init__(self, base, keep):
        self.rows = base.rows
        self.feature_names = [base.feature_names[i] for i in keep]
        self.X = base.X[:, keep]
        self.y, self.y_bin, self.groups = base.y, base.y_bin, base.groups
        self.train_idx, self.valid_idx, self.test_idx = \
            base.train_idx, base.valid_idx, base.test_idx
        self.bounds = {k: (float(np.percentile(self.X[:, i], 5)),
                           float(np.percentile(self.X[:, i], 95)))
                       for i, k in enumerate(self.feature_names)}


def agent_baselines(d, agent, pol, sample_n=36):
    rng = np.random.default_rng(20261107)
    ids = rng.choice(d.test_idx, size=min(sample_n, len(d.test_idx)), replace=False)
    gain_cache = {int(rid): pol._gains(int(rid)) for rid in ids}
    out = {}
    for mode in ("policy", "fixed", "random"):
        steps, converged, acts, gains, plans = [], 0, [], [], []
        for rid in ids:
            r = agent.run(int(rid), mode=mode)
            steps.append(r["planning"]["steps"])
            converged += 1 if r["planning"]["converged"] else 0
            acts.extend(r["planning"]["actions"])
            g = gain_cache[int(rid)]
            gains.append(sum(g.get(a, 0.0) for a in r["planning"]["actions"]
                             if a in g))
            plans.append(tuple(r["planning"]["actions"]))
        per_step = [gg / ss for gg, ss in zip(gains, steps) if ss]
        out[mode] = {"avg_steps": round(float(np.mean(steps)), 2),
                     "convergence_rate": round(converged / len(ids), 3),
                     "avg_information_gain": round(float(np.mean(gains)), 4),
                     "gain_per_step": round(float(np.mean(per_step)), 4),
                     "distinct_actions": len(set(acts)),
                     "distinct_plans": len(set(plans)),
                     "action_distribution": {a: acts.count(a) for a in set(acts)}}
    return out


def selftest() -> int:
    d, m, iv, pol, agent = build()
    metrics = m.evaluate()
    assert metrics["group_overlap"] == 0, "分组划分泄漏"
    assert len(d.rows) == 4424 and d.X.shape[1] == 36

    ci = bootstrap_ci(d, m)
    gci = group_bootstrap(d, m)
    ew = early_warning(d)
    srb = split_robustness(d)
    abl, groups = ablation(d, m)
    baselines = agent_baselines(d, agent, pol)

    outputs = []
    for i, p in enumerate(PROMPTS, 1):
        rid = int(d.test_idx[(i * 37) % len(d.test_idx)])
        r = agent.run(rid, p)
        outputs.append({
            "index": i, "prompt": p, "row_id": rid,
            "course": r["student"]["course"],
            "actual_target": r["student"]["actual_target"],
            "p_dropout": r["student"]["p_dropout"],
            "risk_level": r["student"]["risk_level"],
            "top_signal": r["model_evidence"]["top_signals"][0],
            "counterfactual": r["counterfactual_top"],
            "institutional_alternative": r["institutional_alternative"],
            "cohort_backtest": r["cohort_backtest"],
            "cohort_p_value": r["cohort_backtest"].get("p_value"),
            "cohort_direction": r["cohort_backtest"].get("direction"),
            "recommendation": r["recommendation"],
            "recommendation_verdict": r["recommendation_verdict"],
            "planning": r["planning"], "trace_steps": len(r["trace"]),
            "full_result": r,
        })
        assert r["trace"] and r["counterfactual_top"], f"提示词 {i} 输出不完整"

    report = {
        "selftest": "PASS",
        "data_contract": {
            "source": "UCI Machine Learning Repository #697",
            "url": "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success",
            "download": "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip",
            "doi": "10.24432/C5MC89", "license": "CC BY 4.0 International",
            "retrieved_at": "2026-11-07T15:59:52+08:00",
            "rows": len(d.rows), "features": d.X.shape[1],
            "target_distribution": {k: int(v) for k, v in
                                    zip(*np.unique(d.y, return_counts=True))},
            "courses": int(len(set(d.groups))), "missing_cells": 0,
            "data_csv_sha256": d.source_sha256, "zip_sha256": d.zip_sha256,
            "data_kind": "REAL_INSTITUTIONAL_RECORDS", "no_synthetic_data": True,
        },
        "model_metrics": metrics, "bootstrap_ci": ci,
        "bootstrap_group_ci": gci,
        "early_warning": ew,
        "split_robustness": srb,
        "feature_groups": groups, "ablation": abl,
        "planner": {"train_accuracy": pol.train_accuracy,
                    "holdout_accuracy": pol.holdout_accuracy,
                    "holdout_majority": pol.holdout_majority,
                    "n_holdout_states": pol.n_holdout_states,
                    "majority_baseline": pol.majority_baseline,
                    "label_distribution": pol.label_distribution,
                    "n_training_states": pol.n_training_states,
                    "actions": pol.ACTIONS},
        "agent_baselines": baselines,
        "prompts": outputs,
        "ai_components": ["trained 3-class LR + HGB + Platt calibration",
                          "learned DecisionTree tool-selection policy",
                          "counterfactual intervention engine",
                          "real-cohort backtest", "multi-step agent with memory/reflection"],
    }
    # 全局特征重要性（来自真实训练的 LR 系数；特征已标准化，故 |coef| 可比较），用于图表
    try:
        coef = m.lr.coef_[m.drop_col]
        imp = sorted(zip(d.feature_names, np.abs(coef)), key=lambda x: -x[1])[:12]
        report["feature_importance"] = [
            {"feature": k, "importance": round(float(v), 4)} for k, v in imp]
    except Exception:
        report["feature_importance"] = []

    (OUT / "selftest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    (OUT / "prompt_outputs.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[selftest] PASS  数据 {len(d.rows)}×{d.X.shape[1]} / 课程 {len(set(d.groups))} 组")
    print(f"[model] AUC(macro OVR)={metrics['auc_macro_ovr_calibrated']} "
          f"AUC(dropout LR/HGB)={metrics['auc_dropout_lr']}/{metrics['auc_dropout_hgb']} "
          f"Brier={metrics['brier_dropout']} F1={metrics['f1_macro']} 组重叠={metrics['group_overlap']}")
    print(f"[CI] {ci['ci95_low']} ~ {ci['ci95_high']}")
    print(f"[CI-group] {gci['ci95_low']} ~ {gci['ci95_high']}（{gci['n_groups']} 个专业整组重采样）")
    print(f"[early] " + " | ".join(f"{k}={v}" for k, v in ew["auc_by_feature_set"].items()))
    print(f"[split-robust] {srb['n_seeds']} 次均衡分组：mean={srb['auc_mean']} "
          f"std={srb['auc_std']} [{srb['auc_min']},{srb['auc_max']}]")
    print(f"[planner] holdout={pol.holdout_accuracy} vs 多数类={pol.holdout_majority} "
          f"| in-sample={pol.train_accuracy} n_states={pol.n_training_states}")
    print(f"[agent] " + " | ".join(
        f"{k}: 步{v['avg_steps']} 总增益{v['avg_information_gain']} "
        f"每步{v['gain_per_step']} 收敛{v['convergence_rate']}"
        for k, v in baselines.items()))
    for o in outputs:
        cf = o["counterfactual"]
        print(f"  提示词{o['index']}: {o['risk_level']}风险 p={o['p_dropout']} "
              f"干预={cf['feature'] if cf else 'N/A'} Δ={cf['delta'] if cf else 'N/A'} "
              f"队列方向={o['cohort_direction']} p={o['cohort_p_value']} "
              f"判定={o['recommendation_verdict']} 步数={o['planning']['steps']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8767")))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                    help="监听地址；云平台通常注入 HOST=0.0.0.0 与 PORT")
    a = ap.parse_args()
    if a.serve:
        def _warmup() -> None:
            d, m, iv, pol, agent = build()
            init_state(d, m, agent)

        serve(a.port, a.host, warmup=_warmup)
        return 0
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
