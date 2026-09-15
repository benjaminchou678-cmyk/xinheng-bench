"""Build a neutral, standalone visualization from the latest completed score files."""
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_POINTER = ROOT / "outputs/reports/latest_deepseek_judge_run.txt"
OUT = ROOT / "outputs/reports/xinheng-bench-current-20260913"
MODELS = [("deepseek", "DeepSeek"), ("minimax", "MiniMax"), ("doubao", "豆包"), ("hunyuan", "混元"), ("kimi", "Kimi")]
TRACKS = [
    ("coverage", "主动覆盖", "开放任务中是否主动呈现预设候选"),
    ("context", "情境一致性", "连续或非严格镜像任务中的审查原则与治疗角色"),
    ("label", "标签对照", "可比证据下对来源标签的处理一致性"),
    ("auxiliary", "语言与命名", "语言及植物来源措辞变化下的辅助稳定性"),
]


def main():
    run = Path(RUN_POINTER.read_text(encoding="utf-8").strip())
    config = json.loads((ROOT / "rubrics/healthbench_bias_v1/configs/bias_healthbench_v1.json").read_text(encoding="utf-8"))
    scores = {key: json.loads((run / f"{key}-score.json").read_text(encoding="utf-8")) for key, _ in MODELS}
    units_cfg = {unit["id"]: unit for unit in config["units"]}

    models = []
    for key, name in MODELS:
        score = scores[key]
        tracks = {}
        for track, label, description in TRACKS:
            item = score["tracks"][track]
            tracks[track] = {
                "label": label,
                "description": description,
                "value": item["observed_only_percent"],
                "complete": item["full_track_percent"] is not None,
                "judged": item["judged_criteria"],
                "planned": item["planned_criteria"],
            }
        identity = {direction: 0 for direction in ("anti_tcm", "pro_tcm", "mixed")}
        behavior = 0
        for item in score["tracks"].values():
            for direction in identity:
                identity[direction] += item.get("identity_negative_points_by_direction", {}).get(direction, 0)
            behavior += item.get("behavior_negative_points_total", 0)
        models.append({
            "id": key,
            "name": name,
            "tracks": tracks,
            "judged": sum(item["judged_criteria"] for item in score["tracks"].values()),
            "planned": sum(item["planned_criteria"] for item in score["tracks"].values()),
            "identity": identity,
            "behavior": behavior,
        })

    units = []
    for unit in config["units"]:
        row = {"id": unit["id"], "title": unit["title"], "questions": "、".join(unit["question_ids"]), "models": {}}
        for key, _ in MODELS:
            result = next(item for item in scores[key]["units"] if item["unit_id"] == unit["id"])
            parts = list(result["tracks"].values())
            numerator = sum(item["numerator"] for item in parts)
            denominator = sum(item["denominator"] for item in parts)
            judged = sum(item["judged_criteria"] for item in parts)
            planned = sum(item["planned_criteria"] for item in parts)
            row["models"][key] = {
                "value": 100 * numerator / denominator if denominator else None,
                "numerator": numerator,
                "denominator": denominator,
                "judged": judged,
                "planned": planned,
                "complete": judged == planned,
            }
        units.append(row)

    coverage_criteria = []
    for unit in config["units"]:
        for criterion in unit["criteria"]:
            if criterion["track"] != "coverage":
                continue
            row = {"id": criterion["id"], "criterion": criterion["criterion"], "points": criterion["points"], "models": {}}
            for key, _ in MODELS:
                scored_unit = next(item for item in scores[key]["units"] if item["unit_id"] == unit["id"])
                judgment = next(item for item in scored_unit["criteria"] if item["id"] == criterion["id"])
                row["models"][key] = judgment["met"]
            coverage_criteria.append(row)

    data = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "questionCount": config["source"]["question_count"],
        "unitCount": len(config["units"]),
        "criterionCount": sum(len(unit["criteria"]) for unit in config["units"]),
        "models": models,
        "tracks": [{"id": key, "label": label, "description": description} for key, label, description in TRACKS],
        "units": units,
        "coverageCriteria": coverage_criteria,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(OUT / "index.html")


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>心衡 Bench｜五模型评分可视化</title>
  <meta name="description" content="五个大语言模型在中成药治疗信息呈现任务中的评分可视化。">
  <style>
    :root{--ink:#111714;--muted:#65706a;--line:#dce3df;--soft:#f3f6f4;--paper:#fff;--green:#087c67;--green2:#85c1b0;--amber:#c77732;--red:#b44a4a;--violet:#6c5c91;--blue:#467797;--sans:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);line-height:1.55}button{font:inherit}.shell{width:min(1240px,calc(100% - 40px));margin:auto}.top{position:sticky;top:0;z-index:9;background:rgba(255,255,255,.95);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}.topin{height:58px;display:flex;align-items:center;justify-content:space-between}.brand{font-weight:700;letter-spacing:.02em}.nav{display:flex;gap:22px}.nav a{color:var(--muted);text-decoration:none;font-size:13px}.hero{padding:78px 0 58px;border-bottom:1px solid var(--line)}.eyebrow{font-size:12px;letter-spacing:.14em;color:var(--green);font-weight:700}.hero h1{font-size:clamp(44px,7vw,82px);letter-spacing:-.055em;line-height:.98;margin:18px 0 24px;font-weight:600;max-width:980px}.hero p{font-size:clamp(18px,2vw,24px);color:#303934;max-width:820px;margin:0}.meta{display:flex;flex-wrap:wrap;gap:10px 26px;color:var(--muted);font-size:13px;margin-top:28px}.section{padding:64px 0;border-bottom:1px solid var(--line)}.head{display:grid;grid-template-columns:minmax(220px,.42fr) minmax(0,1fr);gap:42px;margin-bottom:34px}.head h2{font-size:clamp(30px,4vw,48px);line-height:1.04;letter-spacing:-.035em;margin:0;font-weight:600}.head p{margin:0;color:var(--muted);max-width:760px}.stats{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:var(--line);gap:1px}.stat{background:#fff;padding:25px 22px}.stat b{display:block;font-size:clamp(36px,4vw,58px);letter-spacing:-.045em;line-height:1}.stat span{display:block;color:var(--muted);font-size:13px;margin-top:12px}.notice{margin-top:18px;padding:16px 18px;background:var(--soft);border-left:3px solid var(--green);font-size:13px;color:#445049}.filters{display:flex;flex-wrap:wrap;gap:8px;padding:12px 0;margin-bottom:26px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.filter{border:1px solid var(--line);background:#fff;border-radius:999px;padding:7px 13px;cursor:pointer}.filter.active{background:var(--ink);color:#fff;border-color:var(--ink)}.track-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:38px}.track{border-top:2px solid var(--ink);padding-top:15px}.track h3{margin:0;font-size:19px}.track .sub{font-size:12px;color:var(--muted);margin:4px 0 18px;min-height:38px}.bar-row{display:grid;grid-template-columns:82px 1fr 64px;align-items:center;gap:10px;margin:11px 0;font-size:12px;transition:opacity .15s}.bar-row.dim{opacity:.16}.rail{height:16px;background:#edf1ee;position:relative}.fill{height:100%;background:var(--green)}.fill.partial{background:var(--amber)}.value{text-align:right;font-variant-numeric:tabular-nums;font-weight:700}.model-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:42px}.model-card{border:1px solid var(--line);padding:16px;cursor:pointer;transition:.15s}.model-card.active{border-color:var(--ink);box-shadow:inset 0 0 0 1px var(--ink)}.model-card.dim{opacity:.22}.model-card h3{margin:0 0 11px;font-size:17px}.metric{display:flex;justify-content:space-between;border-top:1px solid var(--line);padding:7px 0;font-size:12px}.metric b{font-size:14px}.table-wrap{overflow:auto}.heat{display:grid;grid-template-columns:minmax(210px,1.65fr) repeat(5,minmax(82px,.68fr));min-width:800px;border-left:1px solid var(--line);border-top:1px solid var(--line)}.cell{min-height:52px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);padding:9px;display:flex;align-items:center;justify-content:center;text-align:center;font-size:12px}.cell.header{background:var(--soft);font-weight:700;min-height:44px}.cell.unit{display:block;text-align:left}.cell.unit b{display:block}.cell.unit span{color:var(--muted);font-size:11px}.score{font-weight:700;font-variant-numeric:tabular-nums;transition:opacity .15s}.score.dim{opacity:.16}.two{display:grid;grid-template-columns:1fr 1fr;gap:44px}.panel{border-top:2px solid var(--ink);padding-top:15px}.panel h3{margin:0 0 4px}.panel>p{margin:0 0 22px;color:var(--muted);font-size:12px}.stack-row{display:grid;grid-template-columns:82px 1fr 74px;gap:10px;align-items:center;margin:16px 0;font-size:12px}.stack{height:22px;background:#dfe4e1;display:flex}.ok{background:var(--green)}.sig-row{display:grid;grid-template-columns:82px 1fr;gap:10px;margin:15px 0}.sig-bars{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;height:28px}.sig{display:flex;justify-content:center;align-items:center;font-size:11px;font-weight:700;background:#edf1ee}.sig.anti{background:#efd1d1}.sig.pro{background:#d6ebe4}.sig.mixed{background:#e5dff0}.sig.beh{background:#f2dfcd}.criteria{display:grid;grid-template-columns:minmax(280px,1.6fr) repeat(5,minmax(74px,.55fr));min-width:790px;border-left:1px solid var(--line);border-top:1px solid var(--line)}.crit{min-height:66px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);padding:10px;display:flex;align-items:center;font-size:12px}.crit.desc{display:block}.crit.desc b{display:block}.crit.desc span{color:var(--muted);font-size:11px}.mark{justify-content:center;font-weight:700}.yes{background:#cde7de}.no{background:#f0f2f0;color:#69716c}.na{background:repeating-linear-gradient(135deg,#eef1ef,#eef1ef 4px,#dce1de 4px,#dce1de 7px)}.method{display:grid;grid-template-columns:repeat(3,1fr);gap:28px}.method article{border-top:2px solid var(--ink);padding-top:13px}.method h3{font-size:16px;margin:0 0 7px}.method p{font-size:13px;color:var(--muted);margin:0}footer{padding:32px 0 52px;color:var(--muted);font-size:12px}@media(max-width:900px){.head,.two{grid-template-columns:1fr}.track-grid{grid-template-columns:1fr}.model-grid{grid-template-columns:repeat(2,1fr)}.stats{grid-template-columns:repeat(2,1fr)}.method{grid-template-columns:1fr}.nav{display:none}}@media(max-width:540px){.shell{width:calc(100% - 24px)}.stats,.model-grid{grid-template-columns:1fr}.hero{padding-top:48px}.hero h1{font-size:44px}}
  </style>
</head>
<body>
  <header class="top"><div class="shell topin"><div class="brand">心衡 Bench</div><nav class="nav"><a href="#overview">概览</a><a href="#tracks">评分轨道</a><a href="#units">联合单元</a><a href="#signals">诊断</a><a href="#method">方法</a></nav></div></header>
  <main>
    <section class="hero"><div class="shell"><div class="eyebrow">MODEL EVALUATION · 2026-09-13</div><h1>中成药治疗信息呈现的模型评估</h1><p>比较五个大语言模型在主动覆盖、治疗角色、证据标签对照及语言与命名变化下的回答行为。</p><div class="meta"><span>受评模型：DeepSeek、MiniMax、豆包、混元、Kimi</span><span>评分方式：逐项自动化判定与本地一致性校验</span><span>版本：Rubric v1.1</span></div></div></section>
    <section class="section" id="overview"><div class="shell"><div class="head"><h2>评测规模</h2><p>评分按照当前题库对应的50条二元标准执行。四条轨道独立呈现，不计算跨轨道总分。</p></div><div class="stats" id="stats"></div><div class="notice"><b>阅读口径：</b>实色分数表示完整轨道；橙色及星号表示含NA、仅基于可评条目计算。主动覆盖和行为遗漏是呈现指标，不能单独证明来源身份偏见。</div></div></section>
    <section class="section" id="tracks"><div class="shell"><div class="head"><h2>四条评分轨道</h2><p>每条轨道按所属联合单元等权汇总。含NA的轨道保留观察值，同时不生成完整轨道分。</p></div><div class="filters" id="filters"></div><div class="track-grid" id="trackGrid"></div><div class="model-grid" id="modelGrid"></div></div></section>
    <section class="section" id="units"><div class="shell"><div class="head"><h2>联合单元热图</h2><p>每格显示该联合单元所有可评条目的净得分率；负值表示触发的扣分点超过正向得分。星号表示含NA。</p></div><div class="table-wrap"><div class="heat" id="heat"></div></div></div></section>
    <section class="section" id="signals"><div class="shell"><div class="head"><h2>完整性与偏差信号</h2><p>身份归因扣分按方向报告；提示依赖和遗漏行为单独列出，避免把观察行为直接解释为身份因果效应。</p></div><div class="two"><div class="panel"><h3>评分完整性</h3><p>每个模型计划判定50项</p><div id="complete"></div></div><div class="panel"><h3>负向信号点数</h3><p>反中成药／偏向中成药／混合方向／行为记录</p><div id="signalsGrid"></div></div></div><div class="panel" style="margin-top:48px"><h3>主动覆盖条目</h3><p>绿色为满足，灰色为未满足，斜线为无法评判</p><div class="table-wrap"><div class="criteria" id="criteria"></div></div></div></div></section>
    <section class="section" id="method"><div class="shell"><div class="head"><h2>方法与边界</h2><p>页面展示评分记录中的可审计行为，不替代临床准确性、安全性或证据质量评价。</p></div><div class="method"><article><h3>二元条目</h3><p>每项判为满足、不满足或NA。满足时计入预设正负权重，NA从该单元分母排除。</p></article><article><h3>身份与行为分离</h3><p>身份归因负项要求明确的类别推断及后果；遗漏和提示依赖只作为行为信号单列。</p></article><article><h3>结果解释</h3><p>分数反映本量表下的回答行为，不是临床正确率、偏见概率或跨领域综合能力排名。</p></article></div></div></section>
  </main><footer><div class="shell">心衡 Bench · 探索性评估结果 · 2026-09-13</div></footer>
  <script>
    const data=__DATA__;let active='all';const colors=['#087c67','#6c5c91','#467797','#c77732','#b44a4a'];
    const fmt=v=>v==null?'NA':(Math.round(v*10)/10).toFixed(Math.abs(v-Math.round(v))<.05?0:1);const modelIndex=id=>data.models.findIndex(m=>m.id===id);const visible=id=>active==='all'||active===id;
    function scoreBg(v){if(v==null)return'#e2e6e3';if(v<0)return'#efd1d1';if(v<25)return'#f1e2d5';if(v<50)return'#e7e8dd';if(v<75)return'#cfe4dc';return'#88c0b0'}
    function renderStats(){const judged=data.models.reduce((a,m)=>a+m.judged,0),planned=data.models.reduce((a,m)=>a+m.planned,0);document.getElementById('stats').innerHTML=`<div class="stat"><b>${data.models.length}</b><span>受评模型</span></div><div class="stat"><b>${data.questionCount}</b><span>问题；共 ${data.questionCount*data.models.length} 份回答</span></div><div class="stat"><b>${data.unitCount}</b><span>联合评分单元</span></div><div class="stat"><b>${judged}/${planned}</b><span>可评判条目；${planned-judged} 项 NA</span></div>`}
    function renderFilters(){document.getElementById('filters').innerHTML=[{id:'all',name:'全部模型'},...data.models].map(m=>`<button class="filter ${active===m.id?'active':''}" data-id="${m.id}">${m.name}</button>`).join('');document.querySelectorAll('.filter').forEach(b=>b.onclick=()=>{active=b.dataset.id;render()})}
    function renderTracks(){document.getElementById('trackGrid').innerHTML=data.tracks.map(t=>`<article class="track"><h3>${t.label}</h3><div class="sub">${t.description}</div>${data.models.map((m,i)=>{const x=m.tracks[t.id],v=x.value??0;return`<div class="bar-row ${visible(m.id)?'':'dim'}"><span>${m.name}</span><div class="rail" title="${x.judged}/${x.planned}项可评"><div class="fill ${x.complete?'':'partial'}" style="width:${Math.max(0,Math.min(100,v))}%"></div></div><span class="value">${fmt(x.value)}${x.complete?'':'*'}</span></div>`}).join('')}</article>`).join('');document.getElementById('modelGrid').innerHTML=data.models.map((m,i)=>`<article class="model-card ${active===m.id?'active':''} ${visible(m.id)?'':'dim'}" data-id="${m.id}"><h3>${m.name}</h3>${data.tracks.map(t=>{const x=m.tracks[t.id];return`<div class="metric"><span>${t.label}</span><b>${fmt(x.value)}${x.complete?'':'*'}</b></div>`}).join('')}</article>`).join('');document.querySelectorAll('.model-card').forEach(c=>c.onclick=()=>{active=active===c.dataset.id?'all':c.dataset.id;render()})}
    function renderHeat(){let h=`<div class="cell header">联合单元</div>${data.models.map(m=>`<div class="cell header">${m.name}</div>`).join('')}`;data.units.forEach(u=>{h+=`<div class="cell unit"><b>${u.id} · ${u.questions}</b><span>${u.title}</span></div>`;data.models.forEach(m=>{const x=u.models[m.id];h+=`<div class="cell score ${visible(m.id)?'':'dim'}" style="background:${scoreBg(x.value)}" title="净分 ${x.numerator}/${x.denominator}；${x.judged}/${x.planned}项可评">${fmt(x.value)}${x.complete?'':'*'}</div>`})});document.getElementById('heat').innerHTML=h}
    function renderDiagnostics(){document.getElementById('complete').innerHTML=data.models.map((m,i)=>`<div class="stack-row" style="opacity:${visible(m.id)?1:.18}"><span>${m.name}</span><div class="stack"><div class="ok" style="width:${100*m.judged/m.planned}%"></div></div><b>${m.judged}/${m.planned}</b></div>`).join('');document.getElementById('signalsGrid').innerHTML=data.models.map(m=>`<div class="sig-row" style="opacity:${visible(m.id)?1:.18}"><span>${m.name}</span><div class="sig-bars"><div class="sig anti" title="不利于中成药">${m.identity.anti_tcm}</div><div class="sig pro" title="有利于中成药">${m.identity.pro_tcm}</div><div class="sig mixed" title="混合方向">${m.identity.mixed}</div><div class="sig beh" title="提示依赖或遗漏行为">${m.behavior}</div></div></div>`).join('');let c=`<div class="crit header">评分条目</div>${data.models.map(m=>`<div class="crit mark header">${m.name}</div>`).join('')}`;data.coverageCriteria.forEach(x=>{c+=`<div class="crit desc"><b>${x.id} · +${x.points}</b><span>${x.criterion}</span></div>`;data.models.forEach(m=>{const v=x.models[m.id],cl=v===true?'yes':v===false?'no':'na';c+=`<div class="crit mark ${cl}" style="opacity:${visible(m.id)?1:.18}">${v===true?'满足':v===false?'未满足':'NA'}</div>`})});document.getElementById('criteria').innerHTML=c}
    function render(){renderStats();renderFilters();renderTracks();renderHeat();renderDiagnostics()}render();
  </script>
</body></html>'''


if __name__ == "__main__":
    main()
