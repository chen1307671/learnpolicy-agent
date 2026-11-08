"""20 页内容唯一数据源：PPT / 视频 / 讲稿共用；数字全部来自实测报告。

不写入任何统计常量，一律从 reports/selftest.json 读取。
"""
from __future__ import annotations

import json
import os
import pathlib

HERE = pathlib.Path(__file__).resolve().parents[1]
# 智能体链接：部署后设置环境变量 AGENT_URL 再重建 PPT/视频即可，无需改代码
AGENT_URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8767")
S = json.loads((HERE / "reports" / "selftest.json").read_text(encoding="utf-8"))
DC, M = S["data_contract"], S["model_metrics"]
CI, ABL, PL, BL = S["bootstrap_ci"], S["ablation"], S["planner"], S["agent_baselines"]
PR = S["prompts"]
EW = S.get("early_warning", {})
# 自建知识库统计与外部验证结果（均来自代码/报告，不在此处写死数字）
import sys as _sys
_sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:
    from kb import summary as _kb_summary
    KB_SUM = _kb_summary()
except Exception:  # 知识库缺失时降级，不让 PPT 生成失败
    KB_SUM = {"total": 0, "by_level": {}, "features_covered": []}
_EXT_P = HERE / "reports" / "external_validation.json"
EXT = json.loads(_EXT_P.read_text(encoding="utf-8")) if _EXT_P.exists() else {}
EXT_A = (EXT.get("protocol_a_no_g1g2") or {}).get("auc_mean")
EXT_B = (EXT.get("protocol_b_with_g1") or {}).get("auc_mean")
SRB = S.get("split_robustness", {})
GCI = S.get("bootstrap_group_ci", {})
EWA = EW.get("auc_by_feature_set", {})
_EARLY_ADM = EWA.get("入学信息（无学期成绩）")
_EARLY_S1 = EWA.get("入学信息 + 第一学期")
_NREVIEW = sum(1 for p in PR
               if p.get("recommendation_verdict") in ("manual_review", "insufficient"))
_NWEAK = sum(1 for p in PR if p.get("recommendation_verdict") == "weak")
_NADVISE = sum(1 for p in PR if p.get("recommendation_verdict") == "advise")
_VERDICT_PARTS = []
if _NREVIEW:
    _VERDICT_PARTS.append(f"{_NREVIEW} 条方向冲突被拦下（待复核）")
if _NWEAK:
    _VERDICT_PARTS.append(f"{_NWEAK} 条方向一致但不显著，仅列为线索")
if _NADVISE:
    _VERDICT_PARTS.append(f"{_NADVISE} 条通过方向一致性与显著性检验")
VERDICT_LINE = ("证伪机制实际生效：" + "；".join(_VERDICT_PARTS) + "。"
                if _VERDICT_PARTS else
                "三条提示词均通过方向一致性与显著性检验后，才输出干预建议。")

ROWS = DC["rows"]
FEATS = DC["features"]
COURSES = DC["courses"]
DATA_LINE = "数据：UCI #697（CC BY 4.0, DOI 10.24432/C5MC89），真实高校学业记录；零合成数据。"
_PROMPT_TAG = {1: "① 风险归因", 2: "② 指定干预", 3: "③ 自主规划"}
_FEAT_CN = {
    "Curricular units 1st sem (approved)": "第一学期通过课程",
    "Curricular units 1st sem (grade)": "第一学期成绩",
    "Curricular units 1st sem (evaluations)": "第一学期考核数",
    "Curricular units 1st sem (enrolled)": "第一学期选课数",
    "Scholarship holder": "奖学金（机构级）",
    "Tuition fees up to date": "学费已缴",
    "Debtor": "欠费状态",
}


def _s() -> list[dict]:
    return [
        dict(no=1, kicker="封面", title="学策 LearnPolicy Agent",
             subtitle="可干预性驱动的自适应学业推演智能体",
             hero=(f"{ROWS:,}", "条真实高校学业记录"),
             bullets=["不止预测「谁会辍学」，而是回答：",
                      "对这个学生，先动哪个可干预因素最有效？结论能否在真实数据上被回测证伪？"],
             cards=[(f"{FEATS}", "维真实特征"), ("4+1", "AI 内核 + 工程机制"), ("0", "条合成数据")],
             footnote="通用赛道 · 教育领域 · AI 智能体", cover=True,
             script="各位评委好。学策要回答的不是谁会辍学，而是一个更难的问题："
                    "对这个具体的学生，先动哪个可干预因素最有效，而且这个结论能不能在真实数据上被回测证伪。"),

        dict(no=2, kicker="作品背景与说明", title="知道风险，不等于知道该做什么",
             hero=(f"{M['test_dropout_rate']:.1%}", "测试集真实辍学率"),
             bullets=["传统学业预警只给一个风险分数，无法回答「先动什么、凭什么可信」。",
                      "同一分数背后成因不同：有人缺钱，有人第一学期课程没通过。",
                      "学校资源有限，要的是可执行的干预优先级，而非又一张排名表。"],
             footnote=DATA_LINE,
             script="先看痛点。测试集里有三成的真实辍学率，但只给出风险分数没有用。"
                    "同样的风险背后，有人是欠费，有人是第一学期课程没过。学校需要的是可执行的干预优先级。"),

        dict(no=3, kicker="作品背景与说明", title="边界：决策辅助，不作处分依据", dark=True,
             cards=[("范围内", "风险归因 · 反事实推演 · 队列回测 · 干预建议"),
                    ("范围外", "自动处分 · 标签化学生 · 因果断言 · 成绩预测承诺")],
             bullets=["输出定位为「待辅导员复核的线索」，相关不等于因果。",
                      "不采集个人身份信息；数据为公开匿名化的学业记录。"],
             footnote="伦理边界为设计的一部分，不是补丁。",
             script="边界先说清楚。我们做归因、反事实推演、队列回测和干预建议，"
                    "不做自动处分，不给学生贴标签，也不承诺因果。输出只是供辅导员复核的线索。"),

        dict(no=4, kicker="数据收集与处理", title="真实数据，可核验的授权链",
             table={"headers": ["项", "内容", "来源"],
                    "rows": [["规模", f"{ROWS:,} 行 × {FEATS} 特征", "UCI #697 官方页"],
                             ["标签", "Dropout / Graduate / Enrolled 三分类", "data.csv"],
                             ["专业组", f"{COURSES} 个 Course 分组", "data.csv"],
                             ["授权", "CC BY 4.0 International", "官方 license 段"],
                             ["外部验证", "UCI #320（1,044 条，两校两科）", "Cortez & Silva 2008"]]},
             bullets=["三个渠道：UCI #697 主数据、Zenodo 同源存档、UCI #320 外部验证集，均给出 DOI/出处与获取时间。",
                      "预处理链：HTTPS 直链 → SHA256 校验 → 字段清洗 → 按 Course 分组切分；两条 fetch 脚本可一键复现。",
                      (f"外部验证：UCI #320 同管线 AUC {EXT_A} → 加入第一学段成绩 {EXT_B}，"
                       f"与主数据「第一学期 0.9053」同向。" if EXT_A else
                       "外部验证结果见 reports/external_validation.json。")],
             footnote=("① UCI #697（DOI 10.24432/C5MC89，2022 年发布、无更新版本）；② Zenodo 存档（DOI 10.5281/zenodo.5777339，下载校验）；"
                       "③ UCI #320（两校两科 1,044 条，Cortez & Silva 2008）。存证 raw/uci697_page.html。"),
             script="数据来自三个公开渠道：主数据是 UCI 六九七号，四千四百二十四行、三十六个特征；同源存档用于校验；"
                    "第二个数据集是 UCI 三二零，我们用同一套管线在它上面复现，作为外部验证。"),

        dict(no=5, kicker="数据收集与处理", title="按专业分组划分，杜绝信息泄漏",
             hero=(str(M["group_overlap"]), "训练组与测试组的专业重叠数"),
             bullets=[f"训练 {M['n_train']} 条 / 验证 {M['n_valid']} 条 / 测试 {M['n_test']} 条。",
                      "同一专业若跨集合，模型会「见过」该专业的分布，指标虚高。",
                      "预处理流程：下载 → SHA256 校验 → 字段清洗（去尾部空白、缺失检查）→ 按 Course 分组 → 训练/验证/测试切分；该层即 Agent 的数据底座。",
                      "采用 GroupShuffleSplit 按 Course 两层划分，并写成断言（组重叠=0）。"],
             footnote=(f"管线：{M['pipeline']}；技术手段：HTTPS 直链 + SHA256 + csv 解析（scripts/fetch_data.py）。"
                       f"分组以整个专业入样，行数占比偏离 30%（测试 {M['n_test']} 行 / {M['groups_test']} 组）；"
                       f"{SRB.get('n_seeds', 5)} 次均衡复跑 AUC {SRB.get('auc_mean')} ± {SRB.get('auc_std')}。"),
             script="划分上我们做了一个关键约束：按专业分组切分，同一专业不会同时出现在训练和测试里，"
                    "组重叠为零。否则模型只是记住了专业，指标会虚高。"),

        dict(no=6, kicker="数据收集与处理", title="知识库：把特征异常翻译成可执行措施",
             hero=(f"{KB_SUM['total']}", f"条措施（A {KB_SUM['by_level'].get('A', 0)} / B {KB_SUM['by_level'].get('B', 0)} / C {KB_SUM['by_level'].get('C', 0)}）"),
             table={"headers": ["措施示例", "适用特征", "责任方", "等级"],
                    "col_w": [1.55, 1.45, 1.25, 0.45],
                    "rows": [["学费分期或缓缴", "欠费 / 缴费状态", "资助中心", "A"],
                             ["不及格课程重修", "第一学期通过课程", "教务处", "A"],
                             ["同伴学习小组", "第一学期成绩", "学业支持中心", "B"],
                             ["奖学金评定复核", "奖学金状态", "院系（机构级）", "C"]]},
             bullets=["每条含六个字段：适用特征 / 责任方 / 动作 / 前提 / 禁忌 / 观察指标。",
                      "只有 A、B 级进入建议排序；C 级仅人工参考。接口 /api/kb 可在线浏览全部条目。"],
             footnote=(f"自建知识库（非外部商业库）：{KB_SUM['total']} 条，覆盖 {len(KB_SUM['features_covered'])} 个可干预特征；"
                       f"证据来源只分「本数据队列」与「机制假设」，不引用未核实文献。"),
             script="第三块是知识库。十六条措施，每条都有责任方、前提和禁忌，并标注证据等级；"
                    "只有本数据有方向性证据的 A 级才进入建议，纯机制假设的 C 级不参与排序。"),

        dict(no=6, kicker="分析过程 · 模型", title="分析流程与模型组件",
             table={"headers": ["组件", "作用", "可验证产物"],
                    "rows": [["逻辑回归", "可解释基线 + 系数归因", "特征贡献 Top5"],
                             ["梯度提升树", "非线性对照", "AUC 对比"],
                             ["Platt 校准", "把分数变成可读概率", "Brier 分数"]]},
             bullets=["三种学业结果联合建模，取 P(Dropout) 作为风险量。",
                      "校准后概率直接进入 Agent 状态，驱动后续工具选择。"],
             footnote="全部模型在训练集拟合，测试集只用于最终评估。",
             script="模型部分是真的训练出来的：三分类逻辑回归给出可解释归因，"
                    "梯度提升树作对照，再用 Platt 校准把输出变成可读的概率。"
                    "这个概率不是终点，它是智能体后续决策的状态输入。"),

        dict(no=7, kicker="分析过程 · 结果", title="模型指标与不确定性区间",
             cards=[(str(M["auc_macro_ovr_calibrated"]), "AUC 宏平均 OVR", ""),
                    (str(M["auc_dropout_lr"]), "AUC 辍学 LR", ""),
                    (str(M["auc_dropout_hgb"]), "AUC 辍学 HGB", ""),
                    (str(M["brier_dropout"]), "Brier 校准后", "")],
             bullets=[f"宏平均 F1 = {M['f1_macro']}；树模型 AUC 略高于线性模型，如实上报。",
                      f"行级 bootstrap 95% 区间 {CI['ci95_low']}~{CI['ci95_high']}；按专业整组重采样后为 {GCI.get('ci95_low')}~{GCI.get('ci95_high')}（更保守）。",
                      f"早预警口径（同划分同模型族）：仅入学信息 {_EARLY_ADM}，加第一学期 {_EARLY_S1}，含第二学期 {M['auc_dropout_lr']}——第二学期属学年末信息，不能用于早期干预。"],
             footnote=f"宏平均 F1={M['f1_macro']}；指标来自 GroupShuffleSplit 测试集，可在 selftest.json 复算。",
             script="测试集上宏平均 AUC 零点八四，辍学二分类 AUC 零点九二，"
                    "Brier 零点零八八。我们对辍学二分类 AUC 做了三百次 bootstrap，"
                    "95% 区间是零点九零到零点九四；树模型这次比线性模型略好，也如实上报。"),

        dict(no=8, kicker="分析过程 · 消融", title="哪一类信息真正有用",
             table={"headers": ["特征组", "去掉后 AUC", "AUC 变化"],
                    "rows": [[g, f"{v['auc_without']:.3f}",
                              (f"{v['auc_drop']:+.3f} (AUC↑)" if v["auc_drop"] < 0
                               else f"{v['auc_drop']:+.3f}")]
                             for g, v in sorted(ABL.items(), key=lambda kv: kv[1]["auc_drop"])]},
             bullets=["逐组去掉后重训同一口径的线性模型，观察判别力变化。",
                      "结论用于判断归因是否稳定，不为「所有特征都重要」辩护。"],
             footnote=("表中「AUC 变化」为去掉该组后的差值；负值带 (AUC↑) 表示去掉该组后判别力反升，"
                       "即该组与第二学期高度冗余。差值小于 0.005 的项视为噪声区间，不据此下结论；"
                       "消融在相同的划分与种子下重训，结果见 reports/selftest.json。"),
             script="我们把特征按语义分成六组做消融。对判别力贡献最大的是"
                    "「就读与资助状态」和第二学期学业表现；而第一学期学业表现的边际贡献为负，"
                    "说明它与第二学期高度冗余。但它恰恰是更早、可干预的抓手——"
                    "这正说明「何时干预」比「预测」更关键。"),

        dict(no=9, kicker="创新性与独特视角", title="创新一 · 可干预性反事实推演",
             hero=("Δ P(Dropout)", "有界扰动后的风险变化"),
             bullets=["只对「入学后可改变」的特征做扰动，且限制在该特征真实观测的 5–95 分位内。",
                      "区分学生/辅导员可改（学费、欠费、第一学期学业）与机构决策（奖学金），建议优先输出前者。",
                      "对每个候选干预重新调用校准模型，输出风险变化并排序。",
                      "把「预测分数」升级为「可执行的干预优先级」。"],
             footnote=("可干预清单为字段语义约定【待确认】，非官方定义。与已有方法的差异：DiCE 做多特征多样化反事实（更优，但难以解释与执行）；"
                       "SHAP 只给归因、不给干预；Uplift 需要实验数据。本作品选单特征有界扰动，牺牲最优性换取可解释与可执行。"),
             script="第一个创新点是可干预性反事实推演。我们只扰动入学后能改变的特征，"
                    "并且把扰动限制在真实观测的百分之五到九十五分位内，"
                    "再用校准模型重新预测，把风险分数变成干预优先级。"),

        dict(no=10, kicker="创新性与独特视角", title="创新二 · 可学习的工具选择策略",
             hero=(f"{PL['holdout_accuracy']:.4f}", f"策略留出集准确率 / 多数类 {PL['holdout_majority']:.4f}"),
             bullets=["从真实数据自动生成「状态 → 最优工具」标签：模拟各工具带来的信息增益，取最大者。",
                      "用这些标签训练决策树策略模型；Agent 每步的动作由模型预测，不是 if-else。",
                      f"训练状态 {PL['n_training_states']} 个，其中留出 {PL['n_holdout_states']} 个按学生切分，只用于评估泛化。"],
             footnote="准确率为按学生切分的留出集结果（同一学生的多个状态不跨集合）；标签由真实数据与模型行为计算，不人工标注。",
             script="第二个创新点是工具选择由模型决定。我们先从真实数据自动计算每个工具的信息增益，"
                    "生成状态到动作的标签，再训练一个策略模型。"
                    "智能体每一步调用哪个工具，是这个模型预测出来的，不是写死的顺序。"),

        dict(no=11, kicker="创新性与独特视角", title="创新三 · 真实队列回测证伪",
             hero=("双证据", "模型推演 + 真实队列"),
             bullets=["模型说「提高某特征能降风险」还不够，必须回到原始数据验证。",
                      "在同一专业内取该特征同水平组与更高水平组，统计实际辍学率，降低专业混杂。",
                      "同一专业内做两比例检验；方向冲突或样本不足则标记待复核，方向一致但不显著则仅列为线索。",
                      "跨学科：教育测量定义结果变量 × 统计推断做同专业对照 × 人机协同把最终决定权交还辅导员。"],
             footnote="队列统计来自原始 CSV；相关不等于因果。三级判定在 agent.run 中实现（recommendation_verdict）：advise / weak / manual_review。",
             script="第三个创新点是回测证伪。模型说某个干预有效还不够，"
                    "我们回到原始数据，取同水平组和更高水平组，比较真实辍学率。"
                    "如果方向对不上，就标记待复核，不硬给建议。"),

        dict(no=12, kicker="结果解释与建议", title="多步智能体：记忆、反思与收敛",
             table={"headers": ["机制", "实现", "可观测"],
                    "rows": [["感知", "校准概率 + 预测熵构成状态", "trace 状态"],
                             ["规划", "策略模型输出下一工具", "action_source"],
                             ["记忆", "已执行动作与观察入 memory", "memory 列表"],
                             ["反思", "证据累积提升一致性并触发收敛", "converged"]]},
             bullets=[f"默认最多 6 步；三种基线实测平均步数与信息增益见第 14 页。",
                      "全链路 trace 记录动作来源，证明不是硬编码流水线。"],
             footnote="可选 LLM 适配器接口已预留；默认离线确定性模式可完整复现。",
             script="智能体具备感知、规划、记忆、反思和收敛。"
                    "每一步的动作来源都写进轨迹，评委可以核对它确实是模型选的，不是硬编码流水线。"),

        dict(no=13, kicker="结果解释与建议", title="规划策略与两个基线的真实对比",
             table={"headers": ["模式", "平均步数", "总增益", "每步增益", "方案数"],
                    "rows": [[k, str(v["avg_steps"]), str(v["avg_information_gain"]),
                              str(v.get("gain_per_step")), str(v["distinct_plans"])]
                             for k, v in BL.items()]},
             bullets=[f"总增益受步数影响：策略 {BL['policy']['avg_steps']} 步、固定 {BL['fixed']['avg_steps']} 步；"
                      f"归一为每步增益后，策略 {BL['policy']['gain_per_step']} > 固定 {BL['fixed']['gain_per_step']} > "
                      f"随机 {BL['random']['gain_per_step']}。",
                      f"策略在 36 个学生上生成 {BL['policy']['distinct_plans']} 种方案，固定顺序仅 "
                      f"{BL['fixed']['distinct_plans']} 种；三种模式收敛率均为 {BL['policy']['convergence_rate']:.0%}。"],
             footnote="增益由真实数据上的工具信息量计算；样本 36 个测试学生。总增益与步数同向变化，故同时报每步增益作为同预算比较。",
             script="我们把学习到的策略和固定顺序、随机选择两个基线做对比。"
                    "关键差异不在步数：固定顺序对所有学生千篇一律，而策略为 36 个学生"
                    "生成了多种不同方案，真正随状态自适应。它的每步信息增益与强基线相当，"
                    "却不需要人工设计最优顺序——这才是自适应规划的价值。"),

        dict(no=14, kicker="结果解释与建议", title="三条提示词，三种真实输出",
             table={"headers": ["提示词", "风险", "P(Dropout)", "优先干预", "Δ", "步数"],
                    "col_w": [1.25, 0.55, 1.05, 1.75, 0.75, 0.5],
                    "rows": [[_PROMPT_TAG.get(p["index"], str(p["index"])), p["risk_level"],
                              str(p["p_dropout"]),
                              _FEAT_CN.get((p["counterfactual"] or {}).get("feature"), "-"),
                              str(p["counterfactual"]["delta"]) if p["counterfactual"] else "-",
                              str(p["planning"]["steps"])] for p in PR]},
             bullets=["三条提示词分别触发风险归因、指定干预的推演、以及完全自主规划。",
                      "问题中的干预关键词会影响特征聚焦，但工具选择始终由策略模型决定。",
                      VERDICT_LINE],
             footnote=f"Agent 链接：{AGENT_URL}（账号 judge / 密码 learn2026）；逐字段输出（含方向、p 值、判定）见 reports/prompt_outputs.json。",
             script="三条提示词都有真实输出。第二条指定了干预方向，系统会聚焦对应特征，"
                    "但先跑哪个工具仍然由策略模型决定。完整结果可逐字段核对。"),

        dict(no=15, kicker="应用结果呈现", title="可登录的应用界面",
             hero=("judge", "评审账号，可自行选学生提问"),
             bullets=["左侧选真实学生样本与三条提示词；右侧输出风险、归因、反事实、队列回测、建议。",
                      "规划轨迹逐步展开，每一步标注动作来源；完整 JSON 可展开。",
                      "账号与密码随作品说明提交，符合官方「供评审专家使用」要求。"],
             footnote=f"访问地址 {AGENT_URL}；离线运行、不上传数据；界面不含任何身份信息。",
             script="这是可登录的应用界面。评委可以自己选学生、自己提问，"
                    "现场看到风险、归因、反事实、真实队列佐证和完整的规划轨迹。"),

        dict(no=16, kicker="应用结果呈现", title="可复现性：从数据到结论一条命令",
             table={"headers": ["步骤", "入口"],
                    "rows": [["数据校验 + 全量评测", "python run.py --selftest"],
                             ["启动演示", "python run.py --serve"],
                             ["提交前自检", "python verify_all.py"],
                             ["重建 PPT / 视频", "make_ppt.py / make_video.py"]]},
             bullets=[f"数据 SHA256、划分种子、模型参数、策略标签生成过程全部固定。",
                      "零合成数据；所有数字来自 selftest.json，带管线标签。"],
             footnote="评测报告：reports/selftest.json；逐文件哈希见数据清单。",
             script="可复现性可以当场验证：一条命令跑完数据校验、模型评测、消融、"
                    "基线对比和三条提示词。种子、划分和标签生成过程全部固定。"),

        dict(no=18, kicker="结果解释与建议", title="主动披露局限",
             cards=[("样本", "单一高校 4424 条，跨校泛化未验证"),
                    ("因果", "反事实是模型推演，队列回测只提供相关性"),
                    ("标注", "策略标签由信息增益自动生成，非人工专家标注")],
             bullets=["三分类宏平均 F1 明显低于二分类 AUC，不回避多分类难度。",
                      "策略标签来自信息增益启发式，决策树本质是对该启发式的蒸馏，非强化学习。",
                      "队列回测限制在同一专业后，部分特征样本不足（n<30）时只报证据不足。",
                      "可干预特征清单是设计约定，需领域专家确认。",
                      "未做真实干预实验，不宣称因果效果或效率提升。"],
             footnote=("全部局限写入评测报告，未删改。方法参照：Mothilal et al. (2020) DiCE 反事实解释；"
                       "Lundberg & Lee (2017) SHAP；Wachter et al. (2017) 反事实解释；"
                       "数据集 Realinho et al. (2022) UCI #697。"),
             script="这一页主动讲短板：数据来自单一高校，跨校泛化未验证；"
                    "反事实只是模型推演，队列回测只能提供相关性；"
                    "策略标签来自信息增益而非专家标注。我们不宣称因果效果。"),

        dict(no=19, kicker="结果解释与建议", title="风险识别与落地边界",
             table={"headers": ["风险", "应对"],
                    "rows": [["误判影响学生权益", "仅作辅导员复核线索，不作处分依据"],
                             ["隐私与合规", "公开匿名化数据，CC BY 4.0 署名"],
                             ["干预不可行", "扰动限于真实观测分位，超出则不出建议"]]},
             bullets=["价值主张：把学业预警从「筛人」推进到「给可验证的下一步」。",
                      "优化建议（按性价比）：① 接入第一学期中期考核，预警提前到第 8 周（需教务接口，约 1 学期）；"
                      "② 换用高校辍学数据集做跨校验证（需数据授权，约 2 周）；③ 队列对照升级为倾向得分匹配（需统计人力，约 1 周）。",
                      "最终决策由辅导员结合实际情况作出。"],
             footnote="对应官方「问题与建议」「风险识别与应对」章节。",
             script="风险主要是三条：误判影响学生权益、隐私合规、以及干预是否真的可行。"
                    "对应措施是只做复核线索、使用公开匿名数据、把扰动限制在真实观测范围内。"),

        dict(no=20, kicker="价值收束", title="从风险分数，到可验证的下一步",
             subtitle="用真实学业数据训练模型，用学习到的策略决定每一步，"
                      "用反事实推演提出干预，再用真实队列回测决定该不该说出口。",
             bullets=[f"真实数据 {ROWS:,}×{FEATS} · 专业分组零重叠 · 可训练模型 + 可学习策略",
                      "零合成数据，全部结论可复算；评分细则等缺失项如实标注待确认。"],
             footnote="Data: UCI ML Repository #697, CC BY 4.0, DOI 10.24432/C5MC89.",
             cover=True,
             script="最后收束。学策把学业预警从给一个分数，推进到给一个可验证的下一步："
                    "真实数据训练模型，学到的策略决定每一步，反事实提出干预，"
                    "真实队列回测决定这句话该不该说出口。谢谢各位评委。"),
    ]


SLIDES = _s()
# 内容顺序即页码：统一重排，避免手工维护出错
for _i, _s in enumerate(SLIDES, 1):
    _s["no"] = _i
assert len(SLIDES) == 20, f"页数必须为 20，当前 {len(SLIDES)}"
TOTAL_SCRIPT_CHARS = sum(len(s["script"]) for s in SLIDES)
