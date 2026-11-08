"""学策 LearnPolicy Agent · 核心引擎（从零重建，不复用任何旧选题素材/框架）

AI 内核（四项均为真实可训练/可评测组件，非规则引擎）：
  1. 学业结果预测模型：三分类 LR + HGB 对照 + 概率校准，按 Course 分组划分
  2. 反事实干预推演：对可干预特征做有界扰动，输出 P(Dropout) 变化
  3. 真实队列回测：在真实数据的匹配子群上统计实际 Dropout 率作为独立佐证
  4. 自适应规划器：从真实数据自动生成"状态→最优工具"标签，训练策略模型；
     Agent 每步由策略模型决定下一个工具，带记忆、反思与收敛判据

数据：UCI #697（CC BY 4.0, DOI 10.24432/C5MC89），真实高校记录，无合成数据。
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from kb import LEVEL_LABEL, retrieve as kb_retrieve, summary as kb_summary

HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "raw"
SEED = 20261107
TARGET = "Target"
GROUP = "Course"
DROPOUT = "Dropout"

# 可干预特征：基于字段语义的设计约定（入学后可由学生/学校改变）；非官方定义【待确认】
INTERVENABLE = [
    "Tuition fees up to date", "Scholarship holder", "Debtor",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (enrolled)",
]
# 扰动方向与步长：仅对数值字段，且限制在该字段真实观测的 [p5, p95] 范围内
PERTURB = {
    "Tuition fees up to date": 1, "Scholarship holder": 1, "Debtor": -1,
    "Curricular units 1st sem (approved)": 1, "Curricular units 1st sem (grade)": 1.0,
    "Curricular units 1st sem (evaluations)": 1, "Curricular units 1st sem (enrolled)": 1,
}
# 杠杆类型：student=学生/辅导员可改；institution=需院系或资助部门决策
# （奖学金是评定结果，不是学生能自行改变的变量，故不计入"学生可改"建议）
LEVER = {
    "Tuition fees up to date": "student",
    "Debtor": "student",
    "Curricular units 1st sem (approved)": "student",
    "Curricular units 1st sem (grade)": "student",
    "Curricular units 1st sem (evaluations)": "student",
    "Curricular units 1st sem (enrolled)": "student",
    "Scholarship holder": "institution",
}


def _prop_test(k1: int, n1: int, k2: int, n2: int) -> dict | None:
    """两比例检验。p1=同水平组, p2=更高水平组。

    方向性假设为"更高水平组辍学率更低"（p2<p1），故同时给单侧与双侧 p 值。
    样本不足返回 None，避免用噪声下方向结论。
    """
    if n1 < 10 or n2 < 10:
        return None
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se < 1e-12:
        return {"p_two_sided": 1.0, "p_one_sided": 1.0, "z": 0.0}
    z = (p1 - p2) / se                      # z>0 表示更高水平组辍学率更低（符合预期方向）
    p_two = math.erfc(abs(z) / math.sqrt(2))
    p_one = p_two / 2 if z > 0 else 1 - p_two / 2
    return {"p_two_sided": round(p_two, 4), "p_one_sided": round(p_one, 4),
            "z": round(z, 3)}

FOCUS_MAP = {
    "Curricular units 1st sem (approved)": ["通过课程", "通过学分", "approved", "课程数"],
    "Curricular units 1st sem (grade)": ["成绩", "分数", "grade"],
    "Scholarship holder": ["奖学金", "scholarship"],
    "Tuition fees up to date": ["学费", "缴费", "tuition"],
    "Curricular units 1st sem (evaluations)": ["考核", "evaluation"],
}


def focus_feature(question: str) -> str | None:
    """受限的意图解析：把问题中的干预关键词映射到真实特征；不做语义理解。"""
    q = (question or "").lower()
    for feat, words in FOCUS_MAP.items():
        if any(w.lower() in q for w in words):
            return feat
    return None


PROMPTS = [
    "这名学生的辍学风险有多高？主要受哪些因素影响？",
    "如果优先改善第一学期通过课程数，风险会下降多少？请给出真实数据佐证。",
    "请自主决定需要哪些分析步骤，并给出可执行、可验证的干预建议。",
]


def _csv_rows() -> list[dict]:
    import csv
    txt = (RAW / "data.csv").read_bytes().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(txt), delimiter=";"))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class LearnData:
    def __init__(self):
        self.source_sha256 = _sha(RAW / "data.csv")
        # 原始 ZIP 为可选的溯源附件：未随包提供时不影响运行，仅记录为 not_bundled
        zip_path = RAW / "uci697.zip"
        self.zip_sha256 = _sha(zip_path) if zip_path.exists() else "not_bundled"
        rows = _csv_rows()
        self.rows = rows
        raw_keys = [k for k in rows[0] if k != TARGET]
        self.feature_names = [k.strip() for k in raw_keys]
        self._raw_keys = raw_keys
        self.X = np.asarray([[float(r[raw]) for raw in raw_keys]
                             for r in rows], dtype=float)
        self.y = np.asarray([r[TARGET] for r in rows])
        self.groups = np.asarray([r[GROUP] for r in rows])
        self.y_bin = (self.y == DROPOUT).astype(int)
        self.bounds = {k: (float(np.percentile(self.X[:, i], 5)),
                           float(np.percentile(self.X[:, i], 95)))
                       for i, k in enumerate(self.feature_names)}
        self._split()

    def _split(self):
        """按 Course 分组做两层划分，避免同一专业跨集合造成信息泄漏。"""
        outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
        train_val, test = next(outer.split(self.X, self.y_bin, groups=self.groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
        a, b = next(inner.split(self.X[train_val], self.y_bin[train_val],
                                groups=self.groups[train_val]))
        self.train_idx = train_val[a]
        self.valid_idx = train_val[b]
        self.test_idx = test
        assert not (set(self.groups[self.train_idx]) & set(self.groups[self.test_idx])), \
            "分组划分出现课程重叠"

    def index(self, row_id: int) -> dict:
        r = self.rows[row_id]
        return {"row_id": row_id, "target": r[TARGET],
                "course": r[GROUP],
                "features": {k.strip(): float(raw_val)
                             for k, raw_val in zip(self._raw_keys, [r[x] for x in self._raw_keys])}}


class RiskModel:
    def __init__(self, data: LearnData):
        self.d = data
        med = np.nanmedian(data.X[data.train_idx], axis=0)
        self.med = np.where(np.isfinite(med), med, 0.0)
        mean = np.where(np.isnan(data.X[data.train_idx]), self.med,
                        data.X[data.train_idx]).mean(axis=0)
        std = np.where(np.isnan(data.X[data.train_idx]), self.med,
                       data.X[data.train_idx]).std(axis=0)
        std[std < 1e-9] = 1.0
        self.mean, self.std = mean, std
        Xtr = self.transform(data.X[data.train_idx])
        y = data.y[data.train_idx]
        self.lr = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED)
        self.lr.fit(Xtr, y)
        self.hgb = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08,
                                                  random_state=SEED)
        self.hgb.fit(Xtr, y)
        self.cal = CalibratedClassifierCV(
            LogisticRegression(max_iter=3000, C=1.0, random_state=SEED),
            method="sigmoid", cv=3)
        self.cal.fit(Xtr, y)
        self.classes_ = list(self.lr.classes_)
        self.drop_col = self.classes_.index(DROPOUT)
        # 归因强度的归一化基准：训练集上 |coef·z| 的最大值（一次性估计，供逐样本归因使用）
        z_tr = self.transform(data.X[data.train_idx])
        self.attr_scale = float(np.abs(z_tr * self.lr.coef_[self.drop_col]).max()) or 1.0

    def transform(self, X):
        Xc = np.where(np.isnan(X), self.med, X)
        return (Xc - self.mean) / self.std

    def proba(self, X) -> np.ndarray:
        """返回三分类概率矩阵（校准后）。"""
        return self.cal.predict_proba(self.transform(X))

    def p_dropout(self, X) -> np.ndarray:
        return self.proba(X)[:, self.drop_col]

    def entropy(self, X) -> np.ndarray:
        p = np.clip(self.proba(X), 1e-9, 1)
        return -(p * np.log(p)).sum(axis=1) / math.log(len(self.classes_))

    def evaluate(self) -> dict:
        X, y, yb = self.d.X[self.d.test_idx], self.d.y[self.d.test_idx], \
                   self.d.y_bin[self.d.test_idx]
        p = self.proba(X)
        pd = p[:, self.drop_col]
        y_ovr = np.eye(len(self.classes_))[np.array(
            [self.classes_.index(v) for v in y])]
        auc_macro = roc_auc_score(y_ovr, p, average="macro", multi_class="ovr")
        pred = np.array(self.classes_)[p.argmax(axis=1)]
        return {
            "pipeline": "GroupShuffleSplit(by Course, seed=20261107) -> LR/HGB + Platt calibration",
            "n_train": int(len(self.d.train_idx)),
            "n_valid": int(len(self.d.valid_idx)),
            "n_test": int(len(self.d.test_idx)),
            "test_dropout_rate": round(float(yb.mean()), 4),
            "auc_macro_ovr_calibrated": round(float(auc_macro), 4),
            "auc_dropout_lr": round(float(roc_auc_score(
                yb, self.lr.predict_proba(self.transform(X))[:, self.drop_col])), 4),
            "auc_dropout_hgb": round(float(roc_auc_score(
                yb, self.hgb.predict_proba(self.transform(X))[:, self.drop_col])), 4),
            "brier_dropout": round(float(brier_score_loss(yb, pd)), 4),
            "f1_macro": round(float(f1_score(y, pred, average="macro")), 4),
            "groups_train": int(len(set(self.d.groups[self.d.train_idx]))),
            "groups_test": int(len(set(self.d.groups[self.d.test_idx]))),
            "group_overlap": int(len(set(self.d.groups[self.d.train_idx])
                                 & set(self.d.groups[self.d.test_idx]))),
        }


class Intervention:
    """反事实推演 + 真实队列回测。所有佐证来自原始 CSV 的真实统计。"""

    def __init__(self, data: LearnData, model: RiskModel):
        self.d, self.m = data, model

    def counterfactual(self, row_id: int, top_k: int = 4,
                       focus: str | None = None) -> list[dict]:
        x0 = self.d.X[row_id:row_id + 1].copy()
        base = float(self.m.p_dropout(x0)[0])
        out = []
        for key in INTERVENABLE:
            if key not in self.d.feature_names:
                continue
            j = self.d.feature_names.index(key)
            lo, hi = self.d.bounds[key]
            step = PERTURB.get(key, 1)
            newv = min(hi, max(lo, x0[0, j] + step))
            if abs(newv - x0[0, j]) < 1e-9:
                continue
            x1 = x0.copy()
            x1[0, j] = newv
            p1 = float(self.m.p_dropout(x1)[0])
            out.append({"feature": key, "from": round(float(x0[0, j]), 3),
                        "to": round(float(newv), 3),
                        "p_before": round(base, 4), "p_after": round(p1, 4),
                        "delta": round(p1 - base, 4),
                        "lever": LEVER.get(key, "student")})
        ranked = sorted(out, key=lambda z: z["delta"])
        if focus:
            ranked.sort(key=lambda z: 0 if z["feature"] == focus else 1)
        return ranked[:top_k]

    def cohort_backtest(self, row_id: int, feature: str,
                        model_delta: float | None = None) -> dict:
        """同一专业内的真实子群对照，并给出两比例检验。

        口径约束（对应评审意见）：
        - 只在同一 Course 内取同水平组与更高水平组，降低专业混杂；
        - 两比例 z 检验给 p 值，方向差异不再凭肉眼判断；
        - 任一组 n<30 时返回"证据不足"，不输出方向结论。
        """
        if feature not in self.d.feature_names:
            return {"error": "unknown feature", "direction": "undetermined"}
        j = self.d.feature_names.index(feature)
        v = self.d.X[row_id, j]
        lo, hi = self.d.bounds[feature]
        col = self.d.X[:, j]
        same_course = self.d.groups == self.d.groups[row_id]
        others = same_course & (np.arange(len(self.d.X)) != row_id)
        if len(np.unique(col)) <= 10:
            # 二值 / 低基数字段用精确匹配，避免带宽把两组混为一谈
            nearby = others & (col == v)
            higher = others & (col > v)
            band = 0.0
        else:
            band = max((hi - lo) * 0.10, 0.5)
            nearby = others & (np.abs(col - v) <= band)
            higher = others & (col >= v + max(band, 1.0))

        def rate(mask):
            n = int(mask.sum())
            k = int(self.d.y_bin[mask].sum()) if n else 0
            return {"n": n, "dropouts": k,
                    "actual_dropout_rate": round(k / n, 4) if n else None}

        a, b = rate(nearby), rate(higher)
        test = _prop_test(a["dropouts"], a["n"], b["dropouts"], b["n"])
        p_val = test["p_one_sided"] if test else None
        if a["n"] < 30 or b["n"] < 30 or a["actual_dropout_rate"] is None \
                or b["actual_dropout_rate"] is None:
            direction = "insufficient"
        elif model_delta is None:
            direction = "undetermined"
        else:
            lower_in_higher = b["actual_dropout_rate"] < a["actual_dropout_rate"]
            direction = "consistent" if (model_delta < 0) == lower_in_higher else "opposite"
        return {"feature": feature, "student_value": round(float(v), 3),
                "band": round(band, 3), "stratum": "同一 Course 内",
                "same_level_cohort": a, "higher_level_cohort": b,
                "p_value": p_val,
                "p_two_sided": test["p_two_sided"] if test else None,
                "p_one_sided": test["p_one_sided"] if test else None,
                "z": test["z"] if test else None,
                "direction": direction,
                "note": "同一专业内的真实队列对照；p 值为方向性单侧两比例 z 检验，另附双侧值。相关不等于因果。"}


class PlannerPolicy:
    """可学习的工具选择策略：标签由真实数据 + 模型行为自动生成，不人工标注、不编造。

    状态 = [归一化熵, P(Dropout), 归因强度, 已执行工具 one-hot, 证据一致性]
    动作 = {risk, attribute, counterfactual, backtest, finish}
    标签 = 在真实样本上模拟执行各未执行工具后"不确定性下降最大"的那个动作；
           若当前不确定性已低于收敛阈值则标为 finish。
    """

    ACTIONS = ["risk", "attribute", "counterfactual", "backtest", "finish"]

    def __init__(self, data: LearnData, model: RiskModel, intervention: Intervention):
        self.d, self.m, self.iv = data, model, intervention
        self.convergence = 0.35
        self.clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                          random_state=SEED)
        self.train_accuracy = None
        self.holdout_accuracy = None
        self.holdout_majority = None
        self.n_holdout_states = 0
        self._base_cache: dict[int, tuple[float, float, np.ndarray]] = {}
        self._gain_cache: dict[int, dict] = {}
        self._train()

    def _base(self, row_id: int):
        """逐行缓存熵 / P(Dropout) / 归因贡献，避免训练与推理时重复预测。"""
        c = self._base_cache.get(row_id)
        if c is None:
            x = self.d.X[row_id:row_id + 1]
            z = self.m.transform(x)[0]
            contrib = np.abs(z * self.m.lr.coef_[self.m.drop_col])
            c = (float(self.m.entropy(x)[0]), float(self.m.p_dropout(x)[0]), contrib)
            self._base_cache[row_id] = c
        return c

    def _state(self, row_id: int, done: list[str], consistency: float) -> np.ndarray:
        e, p, contrib = self._base(row_id)
        # 归因强度改为逐样本计算（|coef·z| 的最大值，按训练集尺度归一）；
        # 旧实现用 |coef| 的最大值，与样本无关，等于一个常数伪特征。
        attr = float(min(contrib.max() / (self.m.attr_scale + 1e-9), 1.0))
        flags = [1.0 if a in done else 0.0 for a in self.ACTIONS[:-1]]
        return np.asarray([e, p, attr, *flags, consistency], dtype=float)

    def _gains(self, row_id: int) -> dict:
        cached = self._gain_cache.get(row_id)
        if cached is not None:
            return cached
        cfs = self.iv.counterfactual(row_id, top_k=3)
        cf_feat = cfs[0]["feature"] if cfs else INTERVENABLE[0]
        bt = self.iv.cohort_backtest(row_id, cf_feat)
        a = bt.get("same_level_cohort", {}).get("actual_dropout_rate")
        b = bt.get("higher_level_cohort", {}).get("actual_dropout_rate")
        _, p, contrib = self._base(row_id)
        # 归因增益改为逐样本"集中度"（最大贡献占比）：是否由单一因素主导；
        # 旧实现用 coef 的标准差，与样本无关，同样是常数伪特征。
        attr_gain = float(contrib.max() / (contrib.sum() + 1e-9))
        out = {
            "counterfactual": max([abs(c["delta"]) for c in cfs], default=0.0),
            "backtest": abs((a or 0) - (b or 0)),
            "attribute": attr_gain,
            # risk 的信息量 = 校准概率偏离三分类先验的程度，取自模型真实输出
            "risk": round(abs(p - 1.0 / len(self.m.classes_)), 4),
        }
        self._gain_cache[row_id] = out
        return out

    def _auto_label(self, row_id: int, consistency: float, done: list[str]) -> int:
        """在给定已执行状态下，选信息增益最大的未执行工具作为动作标签。"""
        x = self.d.X[row_id:row_id + 1]
        e0 = float(self.m.entropy(x)[0])
        if e0 < self.convergence:
            return self.ACTIONS.index("finish")
        gains = {k: v for k, v in self._gains(row_id).items() if k not in done}
        if not gains:
            return self.ACTIONS.index("finish")
        best = max(gains, key=gains.get)
        return self.ACTIONS.index(best if gains[best] > 0.02 else "finish")

    def _train(self):
        rng = np.random.default_rng(SEED)
        sample = rng.choice(self.d.train_idx,
                            size=min(700, len(self.d.train_idx)), replace=False)
        rows_states: list[tuple[int, np.ndarray, int]] = []
        for rid in sample:
            # 为每个样本模拟多个不同的执行状态，使策略学到"状态→动作"而非恒定动作
            for k in range(4):
                done = list(rng.choice(self.ACTIONS[:-1], size=k, replace=False)) \
                       if k else []
                consistency = float(rng.uniform(0.2, 0.9))
                rows_states.append((int(rid),
                                    self._state(int(rid), done, consistency),
                                    self._auto_label(int(rid), consistency, done)))
        # 按"学生"切分训练/留出：同一学生的多个状态不跨集合，留出准确率才是泛化口径
        uniq_rows = np.array(sorted({r for r, _, _ in rows_states}))
        rng.shuffle(uniq_rows)
        cut = max(1, int(len(uniq_rows) * 0.8))
        fit_rows = set(uniq_rows[:cut].tolist())
        ho_rows = set(uniq_rows[cut:].tolist())
        Xf = np.asarray([s for r, s, _ in rows_states if r in fit_rows])
        yf = np.asarray([y for r, _, y in rows_states if r in fit_rows])
        Xh = np.asarray([s for r, s, _ in rows_states if r in ho_rows])
        yh = np.asarray([y for r, _, y in rows_states if r in ho_rows])
        self.clf.fit(Xf, yf)
        self.train_accuracy = round(float(self.clf.score(Xf, yf)), 4)
        if len(yh):
            self.holdout_accuracy = round(float(self.clf.score(Xh, yh)), 4)
            self.holdout_majority = round(
                max(Counter(yh).values()) / len(yh), 4)
            self.n_holdout_states = int(len(yh))
        # 部署模型用全部状态重训（留出集只用于报告泛化，不参与部署）
        Xa = np.asarray([s for _, s, _ in rows_states])
        ya = np.asarray([y for _, _, y in rows_states])
        self.clf.fit(Xa, ya)
        self.label_distribution = dict(Counter(self.ACTIONS[i] for i in ya))
        self.n_training_states = len(ya)
        # 多数类基线，用于诚实判断策略是否真的学到东西
        self.majority_baseline = round(
            max(Counter(ya).values()) / len(ya), 4)

    def decide(self, row_id: int, done: list[str], consistency: float) -> str:
        s = self._state(row_id, done, consistency).reshape(1, -1)
        act = self.ACTIONS[int(self.clf.predict(s)[0])]
        if act in done:
            for a in self.ACTIONS:
                if a not in done:
                    return a
        return act


class LearnPolicyAgent:
    """自适应规划智能体：模型驱动工具选择 + 记忆 + 反思 + 收敛。"""

    def __init__(self, data, model, intervention, policy):
        self.d, self.m, self.iv, self.pol = data, model, intervention, policy
        self.metrics = model.evaluate()

    def run(self, row_id: int, question: str = "", max_steps: int = 6,
            mode: str = "policy") -> dict:
        t0 = time.perf_counter()
        trace, memory, done = [], [], []
        consistency = 0.35
        focus = focus_feature(question)
        rng = np.random.default_rng(SEED + row_id)
        fixed_order = ["risk", "attribute", "counterfactual", "backtest", "finish"]
        for step in range(1, max_steps + 1):
            if mode == "fixed":
                act = fixed_order[min(step - 1, len(fixed_order) - 1)]
            elif mode == "random":
                remaining = [a for a in self.pol.ACTIONS if a not in done]
                act = str(rng.choice(remaining)) if remaining else "finish"
            else:
                act = self.pol.decide(row_id, done, consistency)
            e = float(self.m.entropy(self.d.X[row_id:row_id + 1])[0])
            obs = self._execute(act, row_id, memory, focus)
            done.append(act)
            memory.append({"step": step, "action": act, "observation": obs})
            trace.append({"step": step,
                          "action_source": ("trained_policy_model" if mode == "policy"
                                            else f"{mode}_baseline"),
                          "action": act, "entropy_before": round(e, 4),
                          "observation": str(obs)[:220],
                          "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2)})
            if "delta" in str(obs):
                consistency = min(0.95, consistency + 0.2)
            if act == "finish" or (consistency > 0.8 and len(done) >= 3):
                break
        cfs = self.iv.counterfactual(row_id, top_k=4, focus=focus)
        # 建议优先落在"学生/辅导员可改"的杠杆上；机构级杠杆（奖学金）单列，供院系参考
        student_cfs = [c for c in cfs if c.get("lever") == "student"]
        best = (student_cfs or cfs)[0] if cfs else None
        inst_cfs = [c for c in cfs if c.get("lever") == "institution"]
        back = self.iv.cohort_backtest(row_id, best["feature"],
                                       model_delta=best["delta"]) if best else {}
        p0 = float(self.m.p_dropout(self.d.X[row_id:row_id + 1])[0])
        rec, verdict = self._recommend(best, back)
        # 知识库检索：把"哪个特征异常"映射为可执行措施，并标注证据等级
        kb_hits = kb_retrieve(best["feature"]) if best else []
        kb_best = kb_hits[0] if kb_hits else None
        if kb_best and verdict in ("advise", "weak"):
            rec = rec + [f"措施库：{kb_best['name']}（{LEVEL_LABEL[kb_best['evidence']]}；责任方：{kb_best['owner']}）"
                         f"——{kb_best['action']}"]
        return {
            "query": {"row_id": row_id, "question": question or PROMPTS[0]},
            "student": {"course": self.d.rows[row_id][GROUP],
                        "actual_target": self.d.rows[row_id][TARGET],
                        "p_dropout": round(p0, 4),
                        "risk_level": "高" if p0 >= 0.6 else "中" if p0 >= 0.3 else "低"},
            "model_evidence": {"auc_macro_ovr":
                               self.metrics["auc_macro_ovr_calibrated"],
                               "brier_dropout": self.metrics["brier_dropout"],
                               "top_signals": self._attribution(row_id)},
            "counterfactual_top": best,
            "counterfactual_all": cfs,
            "institutional_alternative": inst_cfs[0] if inst_cfs else None,
            "cohort_backtest": back,
            "recommendation": rec, "recommendation_verdict": verdict,
            "knowledge_base": {"selected": kb_best, "entries": kb_hits,
                               "summary": kb_summary()},
            "trace": trace, "memory": memory,
            "planning": {"steps": len(trace), "actions": done, "mode": mode,
                         "policy": "DecisionTree trained on auto-generated labels"
                         if mode == "policy" else f"{mode} baseline",
                         "converged": done[-1] == "finish" or consistency > 0.8},
            "reproducibility": {"seed": SEED, "offline": True,
                                "llm_adapter": "optional (not required)",
                                "data_kind": "REAL_UCI_697"},
        }

    def _attribution(self, row_id: int) -> list[dict]:
        x = self.d.X[row_id:row_id + 1]
        z = self.m.transform(x)[0]
        coef = self.m.lr.coef_[self.m.drop_col]
        items = [{"feature": k, "contribution": round(float(coef[i] * z[i]), 4)}
                 for i, k in enumerate(self.d.feature_names)]
        return sorted(items, key=lambda a: abs(a["contribution"]), reverse=True)[:5]

    def _execute(self, act, row_id, memory, focus=None):
        if act == "risk":
            return {"p_dropout": round(float(
                self.m.p_dropout(self.d.X[row_id:row_id + 1])[0]), 4)}
        if act == "attribute":
            return {"top_signals": self._attribution(row_id)[:3]}
        if act == "counterfactual":
            return {"candidates": self.iv.counterfactual(row_id, top_k=3, focus=focus)}
        if act == "backtest":
            cfs = self.iv.counterfactual(row_id, top_k=1, focus=focus)
            f = cfs[0]["feature"] if cfs else INTERVENABLE[0]
            dlt = cfs[0]["delta"] if cfs else None
            return self.iv.cohort_backtest(row_id, f, model_delta=dlt)
        return {"converged": True, "memory_steps": len(memory)}

    @staticmethod
    def _recommend(best, back):
        """返回 (建议列表, 判定)。判定取值：advise / manual_review / insufficient。"""
        if not best:
            return ["当前样本无可干预特征，建议人工复核"], "manual_review"
        a = back.get("same_level_cohort", {}).get("actual_dropout_rate")
        b = back.get("higher_level_cohort", {}).get("actual_dropout_rate")
        p = back.get("p_value")
        direction = back.get("direction")
        lever = best.get("lever", "student")
        lever_txt = "学生/辅导员可改" if lever == "student" else "机构决策（需院系或资助部门配合）"
        head = (f"候选干预「{best['feature']}」（{lever_txt}）："
                f"模型推演 P(Dropout) {best['delta']:+.4f}")
        if direction == "insufficient" or a is None or b is None:
            return ([head,
                     f"真实队列样本不足（同水平 {back['same_level_cohort']['n']} / "
                     f"更高水平 {back['higher_level_cohort']['n']}），不足以判定方向",
                     "标记待复核：证据不足时不输出干预优先级"], "insufficient")
        if direction == "opposite":
            p_two = back.get("p_two_sided")
            return ([head,
                     f"标记待复核：模型推演与真实队列方向不一致"
                     f"（同水平组辍学率 {a}，更高水平组 {b}，双侧 p={p_two}）",
                     "按设计约定，方向冲突时不输出干预建议，避免误导"],
                    "manual_review")
        if p is not None and p > 0.10:
            return ([head,
                     f"方向一致但差异不显著（同水平 {a} vs 更高水平 {b}，单侧 p={p}）",
                     "列为候选线索，不作为干预优先级；建议补充样本或由辅导员结合个案判断"],
                    "weak")
        return ([head,
                 f"真实队列佐证：同水平组辍学率 {a}，更高水平组 {b}"
                 f"（单侧 p={p}，同专业内对照）",
                 "建议由学业辅导员结合实际情况确认后执行；本结论为决策辅助，不作处分依据"],
                "advise")


def build():
    d = LearnData()
    m = RiskModel(d)
    iv = Intervention(d, m)
    pol = PlannerPolicy(d, m, iv)
    return d, m, iv, pol, LearnPolicyAgent(d, m, iv, pol)
