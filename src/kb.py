"""干预措施知识库（自建，12 条）。

定位：把"某个可干预特征异常"映射为**可执行、有责任方、有前提与禁忌**的措施，
供智能体在给出建议时检索引用。它不是外部商业知识库，也不冒充行业标准；
每条的 `evidence` 明确标注证据来源，只有"本数据"级别才允许进入优先级建议。

证据等级：
- A：本数据队列方向一致（有直接证据）
- B：本数据仅相关性，方向不稳（弱证据，只作线索）
- C：无本数据证据，仅机制假设（不作优先级，仅供人工参考）
"""
from __future__ import annotations

KB: list[dict] = [
    {
        "id": "KB-01", "name": "学费分期或缓缴",
        "features": ["Tuition fees up to date", "Debtor"],
        "owner": "学生资助中心 / 财务处",
        "action": "对欠费学生开放分期缴纳或短期缓缴，避免因缴费状态影响注册与考试资格。",
        "precondition": "学生确有缴费困难，且学籍状态正常。",
        "contraindication": "已进入退学流程或存在恶意欠费记录时不适用。",
        "signal": "缴费状态由欠费转为已缴；次月缺勤率下降。",
        "evidence": "A", "evidence_note": "本数据：缴费状态属'就读与资助状态'组，去掉该组 AUC 下降 0.0162（全组最大）。",
    },
    {
        "id": "KB-02", "name": "紧急助学金 / 困难补助",
        "features": ["Debtor", "Scholarship holder"],
        "owner": "学生资助中心",
        "action": "按家庭经济困难认定结果发放一次性补助或临时困难救助，覆盖当期生活与学习支出。",
        "precondition": "已完成困难认定，或可先认定后补材料。",
        "contraindication": "未做困难认定时不得以模型分数替代认定程序。",
        "signal": "欠费清零；勤工助学/补助到账记录。",
        "evidence": "B", "evidence_note": "本数据：资助状态与辍学相关，但无法区分'补助导致留存'与'低风险者更易获资助'。",
    },
    {
        "id": "KB-03", "name": "奖学金评定复核（机构级）",
        "features": ["Scholarship holder"],
        "owner": "院系奖助工作组",
        "action": "复核评奖口径与名额分配，对边缘学生评估是否有可调整空间。",
        "precondition": "属机构决策，需院系集体审议。",
        "contraindication": "不得为个别学生单独开口子；不得作为模型输出直接执行。",
        "signal": "评奖结果调整记录（需公示程序）。",
        "evidence": "C", "evidence_note": "奖学金是评定结果而非学生可自行改变的变量，仅作机构级参考。",
    },
    {
        "id": "KB-04", "name": "不及格课程重修 / 补考计划",
        "features": ["Curricular units 1st sem (approved)"],
        "owner": "教务处 / 开课学院",
        "action": "为第一学期未通过课程制定重修或补考时间表，优先安排先修链上的关键课程。",
        "precondition": "课程有补考或重修通道。",
        "contraindication": "若挂科集中在无法重修的实践环节，需改走学业辅导。",
        "signal": "第二学期通过课程数上升；重修报名记录。",
        "evidence": "A", "evidence_note": "本数据：第一学期通过课程数是可干预且与结果强相关的字段；同专业队列方向一致者方可输出。",
    },
    {
        "id": "KB-05", "name": "学期选课减负",
        "features": ["Curricular units 1st sem (enrolled)"],
        "owner": "学业导师 / 教务处",
        "action": "在规则允许范围内下调当期学分上限，把精力集中到核心课程。",
        "precondition": "学生已出现多门不及格或多头挂科。",
        "contraindication": "会影响毕业进度或奖助资格时需先做学业规划评估。",
        "signal": "当期学分下降、通过率上升。",
        "evidence": "B", "evidence_note": "本数据：选课数与通过数为同向指标，减负是否有效无法在观测数据中识别。",
    },
    {
        "id": "KB-06", "name": "同伴学习小组 / 朋辈辅导",
        "features": ["Curricular units 1st sem (grade)", "Curricular units 1st sem (approved)"],
        "owner": "学院学业支持中心",
        "action": "按课程组建 4–6 人学习小组，配一名高年级朋辈辅导员，每周固定一次。",
        "precondition": "有稳定的朋辈辅导员来源。",
        "contraindication": "学生明确拒绝集体活动时改为一对一。",
        "signal": "小组出勤；阶段测验均分。",
        "evidence": "B", "evidence_note": "本数据：成绩与留存相关，但辅导效果需干预实验验证。",
    },
    {
        "id": "KB-07", "name": "期中预警面谈",
        "features": ["Curricular units 1st sem (evaluations)", "Curricular units 1st sem (grade)"],
        "owner": "辅导员",
        "action": "在期中考核后 2 周内完成一次结构化面谈，确认困难类型（经济/学业/心理/家庭）并转介。",
        "precondition": "已有期中考核数据。",
        "contraindication": "不得以面谈记录作为处分依据。",
        "signal": "面谈完成率；转介单。",
        "evidence": "A", "evidence_note": "本数据：第一学期考核与成绩是可用信号，面谈本身属流程动作，效果需跟踪。",
    },
    {
        "id": "KB-08", "name": "学业导师一对一",
        "features": ["Curricular units 1st sem (grade)", "Curricular units 1st sem (approved)"],
        "owner": "院系学业导师",
        "action": "每两周一次 30 分钟学业复盘，明确下一阶段 3 个可完成目标。",
        "precondition": "导师带生数在可承受范围内（建议 ≤20）。",
        "contraindication": "导师负荷已满时优先覆盖高风险学生。",
        "signal": "目标完成率；GPA 变化。",
        "evidence": "B", "evidence_note": "本数据：无法区分导师制效应与生源差异。",
    },
    {
        "id": "KB-09", "name": "心理与健康转介",
        "features": ["Curricular units 1st sem (evaluations)", "Curricular units 1st sem (grade)"],
        "owner": "心理健康中心",
        "action": "出现持续缺考、成绩骤降或主动求助时，转介心理咨询或校医院。",
        "precondition": "学生同意，或存在自伤/伤人风险时的强制流程。",
        "contraindication": "模型分数不能作为转介的唯一理由。",
        "signal": "转介完成；复学状态。",
        "evidence": "C", "evidence_note": "本数据无心理健康字段，纯机制假设，不得进入优先级排序。",
    },
    {
        "id": "KB-10", "name": "出勤与作业跟踪",
        "features": ["Curricular units 1st sem (evaluations)", "Curricular units 1st sem (enrolled)"],
        "owner": "任课教师 / 辅导员",
        "action": "对连续两周缺勤或作业缺交达阈值的学生，触发一次提醒与记录。",
        "precondition": "有可用的考勤与作业数据。",
        "contraindication": "不得公开点名或排名。",
        "signal": "缺勤次数下降；作业提交率。",
        "evidence": "B", "evidence_note": "本数据无考勤字段，属间接推断。",
    },
    {
        "id": "KB-11", "name": "专业适配咨询（机构级）",
        "features": ["Curricular units 1st sem (approved)"],
        "owner": "院系教学委员会",
        "action": "对连续两学期核心课大面积不及格者，评估转专业或调整培养方案的可行性。",
        "precondition": "学生本人提出或同意。",
        "contraindication": "不得作为劝退手段。",
        "signal": "转专业申请与后续留存。",
        "evidence": "C", "evidence_note": "本数据无法识别专业适配效应。",
    },
    {
        "id": "KB-12", "name": "家校沟通（需授权）",
        "features": ["Curricular units 1st sem (grade)", "Tuition fees up to date"],
        "owner": "辅导员",
        "action": "在学生书面同意前提下，与家长沟通经济与学业困难，协调支持方案。",
        "precondition": "学生已成年并书面同意，或未成年的法定监护情形。",
        "contraindication": "未经同意不得向家长披露成绩与心理信息。",
        "signal": "支持方案落地情况。",
        "evidence": "C", "evidence_note": "涉及个人信息保护，仅在合规前提下使用。",
    },
    {
        "id": "KB-13", "name": "学业契约（learning contract）",
        "features": ["Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)"],
        "owner": "辅导员 + 学业导师",
        "action": "与学生共同签署一页纸的学期目标与检查节点（每 4 周一次），明确未达标时的支持动作。",
        "precondition": "学生自愿签署，目标须可测量（如通过门数）。",
        "contraindication": "不得作为处分或评奖的附加条件。",
        "signal": "节点达成率；目标课程通过数。",
        "evidence": "B", "evidence_note": "本数据无法识别契约效应；机制上把模糊承诺转为可检查节点。",
    },
    {
        "id": "KB-14", "name": "补考冲刺班",
        "features": ["Curricular units 1st sem (approved)", "Curricular units 1st sem (evaluations)"],
        "owner": "开课学院 / 教务处",
        "action": "在补考前 2–3 周开设集中辅导，按挂科门类分班，配历年题与答疑。",
        "precondition": "补考安排已确定，且有师资。",
        "contraindication": "挂科面过宽（>4 门）时先做减负与学业规划，不宜只办冲刺班。",
        "signal": "补考通过率；参加率。",
        "evidence": "B", "evidence_note": "本数据：通过课程数与留存相关，冲刺班效果需前后对照才能识别。",
    },
    {
        "id": "KB-15", "name": "勤工助学时长调整",
        "features": ["Debtor", "Curricular units 1st sem (grade)"],
        "owner": "学生资助中心 / 用工部门",
        "action": "对每周工时超过阈值的学生下调工时或调整为弹性岗位，避免挤占学习时间。",
        "precondition": "学生有勤工助学岗位记录。",
        "contraindication": "下调工时会影响其生活费来源时，须同时补足资助。",
        "signal": "周工时下降；缺勤与作业提交率。",
        "evidence": "C", "evidence_note": "本数据无工时字段，纯机制假设，不进入优先级排序。",
    },
    {
        "id": "KB-16", "name": "学习空间与设备支持",
        "features": ["Curricular units 1st sem (grade)"],
        "owner": "图书馆 / 信息中心",
        "action": "提供晚间自习位、借用笔记本或网络补贴，解决无设备/无安静环境的问题。",
        "precondition": "学生提出或面谈中识别出环境障碍。",
        "contraindication": "不得收集与学习无关的个人设备信息。",
        "signal": "借用记录；自习时长。",
        "evidence": "C", "evidence_note": "本数据无相关字段，属机制假设。",
    },
]

LEVEL_LABEL = {"A": "A 级（本数据有方向性证据）",
               "B": "B 级（本数据仅相关性）",
               "C": "C 级（机制假设，不作优先级）"}


def retrieve(feature: str, limit: int = 3) -> list[dict]:
    """按特征检索措施；A/B 级优先，C 级排后。"""
    hits = [e for e in KB if feature in e["features"]]
    order = {"A": 0, "B": 1, "C": 2}
    return sorted(hits, key=lambda e: order.get(e["evidence"], 3))[:limit]


def for_lever(lever: str) -> list[dict]:
    """按杠杆类型（student / institution）筛选可执行措施。"""
    if lever == "student":
        return [e for e in KB if "机构级" not in e["name"]]
    return [e for e in KB if "机构级" in e["name"]]


def summary() -> dict:
    from collections import Counter
    return {"total": len(KB),
            "by_level": dict(Counter(e["evidence"] for e in KB)),
            "features_covered": sorted({f for e in KB for f in e["features"]})}
