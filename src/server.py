"""学策 LearnPolicy Agent · 可登录演示服务

评审账号 judge / learn2026。仅监听 127.0.0.1，无外部依赖、不上传数据。
"""
from __future__ import annotations

import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ACCOUNT = "judge"
PASSWORD = "learn2026"
MAX_BODY = 1 << 20
SESSIONS: set[str] = set()
STATE: dict = {}
STATE["ready"] = False

LOGIN_HTML = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>学策 LearnPolicy | 评审登录</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f7f9fc;color:#1a2230;font-family:"Microsoft YaHei",sans-serif;font-size:14px}
.w{max-width:420px;margin:11vh auto;padding:0 20px}h1{font-size:20px;font-weight:600;margin:0 0 6px;color:#1e4fa8}
p{color:#4a5568;font-size:13px;line-height:1.7;margin:0 0 22px}.c{background:#fff;border:1px solid #d6dce5;border-radius:10px;padding:24px}
label{display:block;font-size:12px;color:#4a5568;margin:0 0 6px}input{width:100%;padding:10px 12px;border:1px solid #d6dce5;border-radius:6px;font-size:14px;margin-bottom:16px;font-family:inherit}
button{width:100%;padding:10px;background:#1e4fa8;color:#fff;border:0;border-radius:6px;font-size:14px;cursor:pointer}
.e{color:#d9534f;font-size:12px;min-height:16px;margin-top:8px}.h{font-size:12px;color:#8b97a8;line-height:1.7;margin-top:16px}</style></head>
<body><div class="w"><h1>学策 LearnPolicy Agent</h1><p>可干预性驱动的自适应学业推演智能体<br>评审演示入口</p>
<div class="c"><form method="post" action="/api/login"><label>评审账号</label><input name="account" value="judge">
<label>密码</label><input name="password" type="password"><button type="submit">进入智能体</button></form>
<div class="e" id="e"></div><div class="h">账号与密码随作品说明提交。系统离线运行，数据为 UCI 真实学业记录（CC BY 4.0）。</div></div></div></body></html>"""

APP_HTML = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>学策 LearnPolicy | 演示</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f7f9fc;color:#1a2230;font-family:"Microsoft YaHei",sans-serif;font-size:13px;line-height:1.65}
header{background:#fff;border-bottom:2px solid #1e4fa8;padding:12px 20px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
h1{font-size:16px;font-weight:600;margin:0;color:#1e4fa8}.meta{font-size:11px;color:#8b97a8}
main{max-width:1180px;margin:0 auto;padding:18px 20px;display:grid;grid-template-columns:270px minmax(0,1fr);gap:16px}
@media(max-width:780px){main{grid-template-columns:1fr}}
.panel{background:#fff;border:1px solid #d6dce5;border-radius:10px;padding:14px}h2{font-size:13px;font-weight:600;margin:0 0 10px;color:#1e4fa8}
select,input.q{padding:8px 10px;border:1px solid #d6dce5;border-radius:6px;font-family:inherit;font-size:12px;width:100%}
.ex{padding:8px 10px;margin-bottom:6px;border-radius:6px;background:#f7f9fc;cursor:pointer;font-size:12px;border-left:3px solid #3d7bd9}.ex:hover{background:#e8eff8}
button.go{padding:9px 18px;background:#1e4fa8;color:#fff;border:0;border-radius:6px;cursor:pointer;font-family:inherit;font-size:13px}
.bar{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}input.q{flex:1;min-width:180px;width:auto}
.card{border:1px solid #d6dce5;border-radius:8px;padding:12px;margin-bottom:10px}
.kv{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}.kv span{background:#e8eff8;color:#1e4fa8;border-radius:5px;padding:3px 9px;font-size:12px}
.kv .warn{background:#fdecea;color:#d9534f}.sec{font-size:12px;color:#4a5568;font-weight:600;margin:10px 0 6px}
.ev{border-left:2px solid #3d7bd9;padding:4px 0 4px 10px;margin-bottom:7px;font-size:12px}
.rt{border-left:2px solid #d9534f;padding:4px 0 4px 10px;margin-bottom:6px;font-size:12px}
ul{margin:4px 0;padding-left:18px}li{margin-bottom:3px}details{margin-top:8px}summary{cursor:pointer;font-size:12px;color:#4a5568}
pre{white-space:pre-wrap;word-break:break-word;background:#f7f9fc;padding:10px;border-radius:6px;font-size:11px;max-height:250px;overflow:auto}
.foot{font-size:11px;color:#8b97a8;margin-top:12px}table{width:100%;border-collapse:collapse;font-size:12px}td,th{border-bottom:1px solid #e5e7eb;padding:5px 6px;text-align:left}th{color:#1e4fa8}</style></head><body>
<header><h1>学策 LearnPolicy · 可干预性驱动的自适应学业推演智能体</h1><div><span class="meta" id="m">加载中…</span> <a href="#" onclick="out()" style="font-size:12px;color:#4a5568">退出</a></div></header>
<main><div><div class="panel"><h2>学生样本</h2><select id="sid"></select>
<div style="font-size:12px;color:#4a5568;margin-top:10px">4424 条真实学业记录 · 17 个专业<br>按专业分组划分，训练/测试组不重叠</div></div>
<div class="panel" style="margin-top:12px"><h2>提示词示例</h2><div id="ex"></div></div>
<div class="panel" style="margin-top:12px"><h2>AI 内核</h2><div style="font-size:12px;color:#4a5568">
① 三分类预测模型 + 概率校准<br>② 可学习工具选择策略<br>③ 反事实干预推演<br>④ 真实队列回测<br>⑤ 多步记忆与反思</div></div></div>
<div class="panel"><div class="bar"><input class="q" id="q"><button class="go" onclick="ask()">运行智能体</button></div>
<div id="out"></div><div class="foot">数据：UCI #697（CC BY 4.0, DOI 10.24432/C5MC89）。结论为决策辅助，相关不等于因果。</div></div></main>
<script>let P=[],S=[];
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
async function init(){const c=await(await fetch('/api/config')).json();P=c.prompts;S=c.students;
 document.getElementById('sid').innerHTML=S.map(x=>`<option value="${x.row_id}">#${x.row_id} · 专业${x.course} · 实际${x.target}</option>`).join('');
 document.getElementById('ex').innerHTML=P.map((e,i)=>`<div class="ex" onclick="setQ(${i})">${i+1}. ${esc(e)}</div>`).join('');
 document.getElementById('q').value=P[2];
 const m=await(await fetch('/api/metrics')).json();
 document.getElementById('m').textContent=`AUC ${m.auc} · Brier ${m.brier} · 策略留出准确率 ${m.policy_acc} · 组重叠 ${m.group_overlap}`;
 if(location.search.indexOf('auto=1')>=0){setTimeout(()=>ask(),1200);}}
function setQ(i){document.getElementById('q').value=P[i]}
async function ask(){const q=document.getElementById('q').value,rid=parseInt(document.getElementById('sid').value);
 document.getElementById('out').innerHTML='<div class="card">智能体规划中…</div>';
 const r=await(await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,row_id:rid})})).json();
 if(r.error){document.getElementById('out').innerHTML='<div class="card">'+esc(r.error)+'</div>';return}
 const s=r.student,cf=r.counterfactual_top,bt=r.cohort_backtest;
 const sig=r.model_evidence.top_signals.map(x=>`<tr><td>${esc(x.feature)}</td><td>${x.contribution}</td></tr>`).join('');
 const steps=r.trace.map(t=>`<div class="ev"><b>${t.step}. ${esc(t.action)}</b> <span style="color:#8b97a8">(${esc(t.action_source)})</span><br>${esc(t.observation)}</div>`).join('');
 const a=bt.same_level_cohort,b=bt.higher_level_cohort;
 document.getElementById('out').innerHTML=`
 <div class="card"><div class="kv"><span>风险：${esc(s.risk_level)}</span><span>P(Dropout) ${s.p_dropout}</span>
 <span>实际结果 ${esc(s.actual_target)}</span><span>规划 ${r.planning.steps} 步</span><span>模型 AUC ${r.model_evidence.auc_macro_ovr}</span></div>
 <div class="sec">优先干预（模型反事实推演）</div>
 <div>${cf?`${esc(cf.feature)}：${cf.from} → ${cf.to}，P(Dropout) ${cf.p_before} → ${cf.p_after}（Δ ${cf.delta}）`:'无可用干预'}</div>
 <div class="sec">真实队列回测佐证</div>
 <div>${esc(bt.feature||'')}：同水平组 n=${a?a.n:0} 实际辍学率 ${a?a.actual_dropout_rate:'-'}；更高水平组 n=${b?b.n:0} 实际辍学率 ${b?b.actual_dropout_rate:'-'}</div></div>
 <div class="card"><div class="sec">模型归因（Top 5）</div><table><tr><th>特征</th><th>贡献</th></tr>${sig}</table>
 <div class="sec">自适应规划轨迹（工具选择由策略模型产生）</div>${steps}
 <div class="sec">复核建议</div><ul>${r.recommendation.map(x=>'<li>'+esc(x)+'</li>').join('')}</ul>
 ${r.knowledge_base&&r.knowledge_base.selected?`<div class="sec">措施库（${esc(r.knowledge_base.summary.total)} 条，命中 ${r.knowledge_base.entries.length} 条）</div>
 <ul>${r.knowledge_base.entries.map(e=>'<li>【'+esc(e.evidence)+'级】<b>'+esc(e.name)+'</b>（'+esc(e.owner)+'）：'+esc(e.action)+'</li>').join('')}</ul>`:''}</div>
 <div class="card"><details><summary>展开完整 JSON</summary><pre>${esc(JSON.stringify(r,null,1))}</pre></details></div>`;}
async function out(){await fetch('/api/logout',{method:'POST'});location.reload()}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')ask()});init();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "LearnPolicy/1.0"

    def log_message(self, fmt, *args):
        print("[learn] " + fmt % args)

    def _send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
        b = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or []):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def _json(self, obj, code=200, extra=None):
        self._send(code, json.dumps(obj, ensure_ascii=False),
                   "application/json; charset=utf-8", extra)

    def _sid(self):
        for p in self.headers.get("Cookie", "").split(";"):
            if p.strip().startswith("sid="):
                return p.strip()[4:]
        return None

    def _authed(self):
        s = self._sid()
        return bool(s and s in SESSIONS)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            ready = bool(STATE.get("ready") and "agent" in STATE)
            return self._json({"status": "ok" if ready else "loading",
                               "ready": ready,
                               "rows": len(STATE["data"].rows) if (ready and STATE.get("data")) else 0})
        if path == "/":
            q = parse_qs(urlparse(self.path).query)
            if q.get("demo") and self.client_address[0] in ("127.0.0.1", "::1"):
                sid = secrets.token_hex(16)
                SESSIONS.add(sid)
                return self._send(302, "", extra=[("Location", "/?auto=1"),
                                                  ("Set-Cookie", f"sid={sid}; Path=/; HttpOnly")])
            return self._send(200, APP_HTML if self._authed() else LOGIN_HTML)
        if not self._authed():
            return self._json({"error": "unauthorized"}, 401)
        if not STATE.get("ready"):
            return self._json({"status": "loading",
                               "detail": "模型加载中，请稍候后刷新"}, 503)
        if path == "/api/config":
            d = STATE["data"]
            students = [{"row_id": int(i), "course": d.rows[i]["Course"],
                         "target": d.rows[i]["Target"]}
                        for i in list(d.test_idx[:12])]
            return self._json({"prompts": STATE["prompts"], "students": students})
        if path == "/api/kb":
            from kb import KB, summary as kb_summary
            return self._json({"summary": kb_summary(), "entries": KB})
        if path == "/api/metrics":
            m = STATE["metrics"]
            return self._json({"auc": m["auc_macro_ovr_calibrated"],
                               "brier": m["brier_dropout"],
                               "policy_acc": STATE["policy_acc"],
                               "group_overlap": m["group_overlap"]})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except (TypeError, ValueError):
            return self._json({"error": "invalid content length"}, 400)
        if length < 0 or length > MAX_BODY:
            return self._json({"error": "content length out of range"}, 413)
        raw = self.rfile.read(length).decode("utf-8", errors="replace") if length else ""
        if path == "/api/login":
            f = parse_qs(raw)
            if f.get("account", [""])[0] == ACCOUNT and f.get("password", [""])[0] == PASSWORD:
                sid = secrets.token_hex(16)
                SESSIONS.add(sid)
                return self._send(302, "", extra=[("Location", "/"),
                                                  ("Set-Cookie", f"sid={sid}; Path=/; HttpOnly")])
            return self._send(200, LOGIN_HTML.replace('<div class="e" id="e"></div>',
                                                     '<div class="e" id="e">账号或密码不正确</div>'))
        if path == "/api/logout":
            s = self._sid()
            if s:
                SESSIONS.discard(s)
            return self._json({"ok": True})
        if not self._authed():
            return self._json({"error": "unauthorized"}, 401)
        if path == "/api/ask":
            try:
                payload = json.loads(raw or "{}")
            except (json.JSONDecodeError, TypeError):
                return self._json({"error": "invalid json"}, 400)
            if not isinstance(payload, dict):
                return self._json({"error": "payload must be an object"}, 400)
            q = payload.get("question", "")
            if not isinstance(q, str) or not q.strip() or len(q) > 400:
                return self._json({"error": "invalid question"}, 400)
            try:
                rid = int(payload.get("row_id", STATE["data"].test_idx[0]))
            except (TypeError, ValueError):
                return self._json({"error": "invalid row_id"}, 400)
            if not 0 <= rid < len(STATE["data"].rows):
                return self._json({"error": "row_id out of range"}, 400)
            try:
                if not STATE.get("ready"):
                    return self._json({"status": "loading",
                                       "detail": "模型加载中，请稍候"}, 503)
                return self._json(STATE["agent"].run(rid, q.strip()))
            except Exception as exc:
                return self._json({"error": "execution failed",
                                   "detail": type(exc).__name__}, 500)
        return self._json({"error": "not found"}, 404)


def init_state(data, model, agent):
    from core import PROMPTS
    STATE["data"] = data
    STATE["model"] = model
    STATE["agent"] = agent
    STATE["prompts"] = PROMPTS
    STATE["metrics"] = model.evaluate()
    STATE["ready"] = True
    STATE["policy_acc"] = (agent.pol.holdout_accuracy
                           if agent.pol.holdout_accuracy is not None
                           else agent.pol.train_accuracy)


def serve(port: int = 8767, host: str = "127.0.0.1", warmup=None):
    """先占用端口再预热模型：云平台健康检查可在训练完成前通过。"""
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"[learn] http://{host}:{port}  account={ACCOUNT} password={PASSWORD}", flush=True)
    if host == "0.0.0.0":
        print("[learn] 注意：已绑定全部网卡，仅建议在受信任网络或反向代理后使用", flush=True)
    if warmup is not None:
        import threading
        def _run():
            try:
                warmup()
            except Exception as exc:  # noqa: BLE001
                print(f"[learn] 模型加载失败：{type(exc).__name__}: {exc}", flush=True)
        threading.Thread(target=_run, daemon=True).start()
        print("[learn] 模型加载中（约 60-90 秒），期间 /api/health 返回 loading", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[learn] stopped")
    finally:
        httpd.server_close()
