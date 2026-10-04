"""Build a single-scene browser review of tracks, VLM candidates, and final events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Original MP4 file")
    parser.add_argument("--events", required=True, help="Final clip JSON")
    parser.add_argument("--tracks", required=True, help="Per-frame tracking JSONL")
    parser.add_argument("--candidates", required=True, help="Candidates sent to the VLM")
    parser.add_argument("--review", help="VLM review JSON; uses candidate IDs")
    parser.add_argument("--ballots", help="Optional first-stage binary VLM window decisions")
    parser.add_argument("--proposals", help="Optional two-stage localized proposal JSON")
    parser.add_argument("--reference", help="GT events JSON")
    parser.add_argument("--output", required=True, help="Destination HTML")
    args = parser.parse_args()

    video = Path(args.input).resolve()
    if video.suffix.lower() not in VIDEO_SUFFIXES or not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    clip = read_json(Path(args.events))
    if clip["clip_id"] != video.stem:
        raise ValueError(f"Video {video.stem} does not match clip {clip['clip_id']}")
    tracks = [json.loads(line) for line in Path(args.tracks).read_text(encoding="utf-8").splitlines()]
    candidates = read_json(Path(args.candidates))
    reviews = read_json(Path(args.review)) if args.review else []
    review_by_id = {row["candidate_id"]: row for row in reviews}
    for candidate in candidates:
        review = review_by_id.get(candidate["candidate_id"])
        if review:
            candidate["decision"] = review.get("decision", "unknown")
            candidate["reason"] = review.get("reason", "")
            candidate["event_types"] = [event["type"] for event in review.get("events", [])]
    proposals = read_json(Path(args.proposals)) if args.proposals else []
    ballots = read_json(Path(args.ballots)) if args.ballots else []
    reference = read_json(Path(args.reference)).get(clip["clip_id"], {}) if args.reference else {}
    payload = {"source": video.as_uri(), "clip": clip, "tracks": tracks,
               "candidates": candidates, "ballots": ballots, "proposals": proposals,
               "reference": reference.get("interactions", [])}
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    template = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Scene pipeline review</title>
<style>
body{font:15px Segoe UI,Arial,sans-serif;color:#1d2730;background:#f4f6f8;margin:20px}
main{max-width:1250px;margin:auto}h1{font-size:24px;margin:0 0 5px}.note{color:#56626c}
.controls,.transport{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:12px 0}
label{white-space:nowrap}button,select{padding:6px 9px}input[type=range]{flex:1;min-width:250px}
#frame{width:100%;background:#111;max-height:72vh;object-fit:contain}#video{display:none}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:white;padding:12px;border:1px solid #dce2e8;border-radius:8px;max-height:340px;overflow:auto}
.item{padding:7px;border-bottom:1px solid #e5e9ed;cursor:pointer}.item:hover,.selected{background:#e9f3ff}.muted{color:#66727b}
.track{color:#b100a0}.candidate{color:#087c89}.proposal{color:#1567b0}.alg{color:#a5167d}.gt{color:#9d7300}
@media(max-width:750px){.grid{grid-template-columns:1fr}}
</style></head><body><main><h1 id="title"></h1>
<p class="note">Choose the layers to display. Click a candidate or event to jump to its start. The video remains on your computer.</p>
<div class="controls">
<label><input id="showTracks" type="checkbox" checked> All tracks</label>
<label><input id="showCandidates" type="checkbox" checked> VLM candidates</label>
<label><input id="showBallots" type="checkbox" checked> Binary VLM windows</label>
<label><input id="showProposals" type="checkbox" checked> Localized proposals</label>
<label><input id="showAlg" type="checkbox" checked> Final ALG events</label>
<label><input id="showGT" type="checkbox" checked> GT events</label>
<label>Focus candidate <select id="focus"><option value="">All candidates</option></select></label>
</div>
<video id="video" preload="metadata" playsinline></video><canvas id="frame"></canvas>
<div class="transport"><button id="back">◀ frame</button><button id="play">▶ play</button><button id="pause">⏸ pause</button><button id="forward">frame ▶</button><input id="seek" type="range" min="0" max="1" step="0.001" value="0"><strong id="clock">0.00 s</strong></div>
<p id="active" class="note"></p>
<div class="grid"><section class="panel"><h2>Original candidates</h2><div id="candidates"></div></section>
<section class="panel"><h2>Binary VLM windows</h2><div id="ballots"></div></section>
<section class="panel"><h2>Final ALG events</h2><div id="alg"></div><h2>Ground truth</h2><div id="gt"></div></section>
<section class="panel"><h2>Localized proposals</h2><div id="proposals"></div></section>
<section class="panel"><h2>Tracking at current frame</h2><div id="tracking"></div></section></div>
</main><script>const data=__DATA__;
const video=document.getElementById('video'),canvas=document.getElementById('frame'),ctx=canvas.getContext('2d');
const seek=document.getElementById('seek'),focus=document.getElementById('focus');let lastFrame=-1;
const esc=s=>String(s??'').replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));
const fmt=n=>Number(n||0).toFixed(2)+'s';const span=e=>(e.spans||[]).map(s=>fmt(s.start_s)+'–'+fmt(s.end_s)).join(', ');
const active=(e,t)=>e.start_s<=t&&t<e.end_s;const eventActive=(e,t)=>(e.spans||[]).some(s=>active(s,t));
const checks=['showTracks','showCandidates','showBallots','showProposals','showAlg','showGT'];const on=id=>document.getElementById(id).checked;
document.getElementById('title').textContent='Pipeline review — '+data.clip.clip_id;
video.src=data.source;video.addEventListener('loadedmetadata',()=>{seek.max=video.duration;draw()});
video.addEventListener('seeked',draw);video.addEventListener('timeupdate',()=>{seek.value=video.currentTime;document.getElementById('clock').textContent=fmt(video.currentTime);draw()});
video.addEventListener('play',()=>requestAnimationFrame(loop));function loop(){draw();if(!video.paused)requestAnimationFrame(loop)}
document.getElementById('play').onclick=()=>video.play();document.getElementById('pause').onclick=()=>video.pause();seek.oninput=()=>{video.currentTime=Number(seek.value)};
for(const id of checks)document.getElementById(id).onchange=draw;focus.onchange=draw;
function nearestRow(t){let best=null;for(const row of data.tracks){if(!best||Math.abs(row.timestamp_s-t)<Math.abs(best.timestamp_s-t))best=row}return best||{objects:[]}}
function step(n){video.pause();let i=data.tracks.findIndex(r=>r.timestamp_s>=video.currentTime);if(i<0)i=data.tracks.length-1;video.currentTime=data.tracks[Math.max(0,Math.min(data.tracks.length-1,i+n))]?.timestamp_s??video.currentTime}
document.getElementById('back').onclick=()=>step(-1);document.getElementById('forward').onclick=()=>step(1);
function jump(t){video.pause();video.currentTime=t;draw()}
function list(target,items,render,start){const box=document.getElementById(target);box.replaceChildren();if(!items.length){box.textContent='None';return}for(const item of items){let div=document.createElement('div');div.className='item';div.innerHTML=render(item);div.onclick=()=>jump(start(item));box.append(div)}}
data.candidates.forEach((c,i)=>{const o=document.createElement('option');o.value=String(i);o.textContent=`${c.candidate_id}: ${c.person_id} → ${c.vehicle_id} (${fmt(c.start_s)}–${fmt(c.end_s)})`;focus.append(o)});
list('candidates',data.candidates,c=>`<b class="candidate">${esc(c.candidate_id)}</b> ${esc(c.person_id)} → ${esc(c.vehicle_id)} · ${fmt(c.start_s)}–${fmt(c.end_s)}<br>${c.decision?'VLM: '+esc(c.decision)+' '+esc((c.event_types||[]).join(', ')):'Input to localization'}<br><span class="muted">${esc(c.reason||'')}</span>`,c=>c.start_s);
list('ballots',data.ballots,b=>`<b class="candidate">${esc(b.decision)}</b> ${esc(b.person_id)} → ${esc(b.vehicle_id)} · ${fmt(b.start_s)}–${fmt(b.end_s)}<br><span class="muted">${esc(b.reason||'')}</span>`,b=>b.start_s);
list('proposals',data.proposals,p=>`<b class="proposal">${esc(p.person_id)} → ${esc(p.vehicle_id)}</b> · ${fmt(p.start_s)}–${fmt(p.end_s)} · ${p.positive_windows} positive windows<br>Second pass: ${esc(p.decision?.decision||'unknown')} ${esc((p.decision?.events||[]).map(e=>e.type).join(', '))}<br><span class="muted">${esc(p.decision?.reason||'')}</span>`,p=>p.start_s);
list('alg',data.clip.interactions,e=>`<b class="alg">${esc(e.type)}</b> · ${span(e)} · ${esc(e.persons?.[0]?.person_id)} → ${esc(e.vehicle?.vehicle_id)}`,e=>e.spans[0].start_s);
list('gt',data.reference,e=>`<b class="gt">${esc(e.type)}</b> · ${span(e)}`,e=>e.spans[0].start_s);
function rect(box,color,label,width=3){if(!box)return;const [x1,y1,x2,y2]=box;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.font='17px Segoe UI';ctx.fillStyle=color;ctx.fillText(label,x1,Math.max(18,y1-4))}
function draw(){if(!video.videoWidth)return;const t=video.currentTime,row=nearestRow(t),objects=row.objects||[];
canvas.width=video.videoWidth;canvas.height=video.videoHeight;ctx.drawImage(video,0,0);
const selected=focus.value===''?null:data.candidates[Number(focus.value)];const candidates=data.candidates.filter(c=>active(c,t)&&(!selected||c===selected));
const ballots=data.ballots.filter(b=>active(b,t)&&(!selected||(b.person_id===selected.person_id&&b.vehicle_id===selected.vehicle_id)));
const proposals=data.proposals.filter(p=>active(p,t));const events=data.clip.interactions.filter(e=>eventActive(e,t));const gt=data.reference.filter(e=>eventActive(e,t));
if(on('showTracks'))for(const o of objects)rect(o.bbox,o.kind==='person'?'#ff40d0':'#34baff',o.id,2);
if(on('showCandidates'))for(const c of candidates)for(const id of [c.person_id,c.vehicle_id]){const o=objects.find(o=>o.id===id);if(o)rect(o.bbox,'#00dbb4',id+' CAND '+c.candidate_id,4)}
if(on('showBallots'))for(const b of ballots.filter(b=>b.decision==='interaction'))for(const id of [b.person_id,b.vehicle_id]){const o=objects.find(o=>o.id===id);if(o)rect(o.bbox,'#7cec43',id+' VOTE+',3)}
if(on('showProposals'))for(const p of proposals)for(const id of [p.person_id,p.vehicle_id]){const o=objects.find(o=>o.id===id);if(o)rect(o.bbox,'#397bff',id+' LOCAL',4)}
if(on('showAlg'))for(const e of events)for(const id of [...e.persons.map(p=>p.person_id),e.vehicle.vehicle_id]){const o=objects.find(o=>o.id===id);if(o)rect(o.bbox,'#ffab32',id+' ALG '+e.type,5)}
let labels=[];if(on('showCandidates'))labels.push(...candidates.map(c=>'CAND '+c.candidate_id));if(on('showBallots'))labels.push(...ballots.map(b=>'VLM WINDOW '+b.decision));if(on('showProposals'))labels.push(...proposals.map(p=>'LOCAL '+(p.decision?.decision||'?')));if(on('showAlg'))labels.push(...events.map(e=>'ALG '+e.type));if(on('showGT'))labels.push(...gt.map(e=>'GT '+e.type));
if(labels.length){ctx.fillStyle='rgba(0,0,0,.72)';ctx.fillRect(0,0,canvas.width,labels.length*26+8);ctx.font='18px Segoe UI';labels.forEach((s,i)=>{ctx.fillStyle=s.startsWith('GT')?'#ffe069':'#fff';ctx.fillText(s,8,23+i*26)})}
document.getElementById('active').textContent=`At ${fmt(t)}: ${candidates.length} candidate(s), ${ballots.length} binary VLM window(s), ${proposals.length} localized proposal(s), ${events.length} ALG event(s), ${gt.length} GT event(s).`;
document.getElementById('tracking').textContent=objects.map(o=>`${o.id} (${o.kind}, ${Number(o.confidence).toFixed(2)})`).join(' · ')||'No active tracks';
}
</script></body></html>'''
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace("__DATA__", data), encoding="utf-8")
    print(destination.resolve())


if __name__ == "__main__":
    main()
