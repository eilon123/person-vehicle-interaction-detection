"""Create a lightweight browser-based live reviewer over the original videos."""
import argparse
import html
import json
from pathlib import Path

from person_vehicle.io import read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original video directory")
    parser.add_argument("--events", required=True, help="Directory containing clip JSON files")
    parser.add_argument("--output", required=True, help="HTML destination")
    parser.add_argument("--reference", default=str(Path(__file__).resolve().parents[1] / "annotations" / "manual" / "events.json"),
                        help="GT events.json; defaults to the canonical annotation-tool output")
    parser.add_argument("--clip", help="Include only this clip ID")
    parser.add_argument("--tracks", help="Directory containing per-frame track JSONL files; defaults next to events")
    parser.add_argument("--descriptions", help="Directory containing per-clip VLM-description JSON files")
    args = parser.parse_args()
    input_dir, event_dir, destination = Path(args.input), Path(args.events), Path(args.output)
    track_dir = Path(args.tracks) if args.tracks else event_dir.parent / "tracks"
    description_dir = Path(args.descriptions) if args.descriptions else event_dir.parent / "vlm_descriptions"
    clips = []
    reference = read_json(args.reference) if args.reference else {}
    for event_path in sorted(event_dir.glob("*.json")):
        clip = read_json(event_path)
        if args.clip and clip.get("clip_id") != args.clip:
            continue
        source = next((p for p in input_dir.glob(f"{clip['clip_id']}.*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}), None)
        if not source or clip.get("status") != "ok":
            continue
        track_path = track_dir / f"{clip['clip_id']}.jsonl"
        track_rows = [json.loads(line) for line in track_path.read_text(encoding="utf-8").splitlines()] if track_path.exists() else []
        description_path = description_dir / f"{clip['clip_id']}.json"
        description_doc = read_json(description_path) if description_path.exists() else {"descriptions": []}
        clips.append({"clip_id": clip["clip_id"], "source": source.resolve().as_uri(),
                      "duration_s": clip["duration_s"], "interactions": clip["interactions"],
                      "reference": reference.get(clip["clip_id"], {}).get("interactions", []),
                      "descriptions": description_doc.get("descriptions", []),
                      "tracks": track_rows})
    payload = json.dumps(clips, ensure_ascii=False).replace("</", "<\\/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    title = destination.stem.replace("_", " ")
    page = f'''<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:15px Segoe UI,Arial;margin:24px;background:#f4f6f8;color:#18212b}}main{{max-width:1100px;margin:auto}}video{{width:100%;background:#111}}.bar{{display:flex;gap:8px;align-items:center;margin:10px 0;flex-wrap:wrap}}button,select{{padding:7px 10px}}input[type=range]{{flex:1;min-width:280px}}.event{{padding:7px 10px;border-left:5px solid #c020c0;background:#fff;margin:5px 0;cursor:pointer}}.gt{{border-color:#e0a400}}.hint{{color:#5b6670}}#time{{font-variant-numeric:tabular-nums}}canvas{{width:100%;max-height:70vh;background:#111}}#video{{display:none}}.timeline-card{{background:#fff;border:1px solid #d5dbe2;border-radius:8px;padding:12px;margin:14px 0}}.timeline-head,.timeline-row{{display:grid;grid-template-columns:90px 1fr;gap:8px}}.timeline-head{{margin-bottom:5px}}.ticks{{position:relative;height:20px;color:#68737e;font-size:11px}}.tick{{position:absolute;transform:translateX(-50%);white-space:nowrap}}.timeline-row{{align-items:start;margin:7px 0}}.timeline-label{{font-weight:650;padding-top:5px}}.timeline-track{{position:relative;min-height:29px;background:repeating-linear-gradient(to right,#eef1f4 0,#eef1f4 1px,transparent 1px,transparent 10%);border:1px solid #d9dee4;border-radius:4px;overflow:hidden;cursor:pointer}}.segment{{position:absolute;height:22px;top:3px;border-radius:4px;color:#fff;font-size:11px;line-height:22px;padding:0 5px;box-sizing:border-box;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;cursor:pointer;min-width:3px}}.segment.alg{{background:#b225b8}}.segment.gt-segment{{background:#cf8c00}}.playhead{{position:absolute;top:0;bottom:0;width:2px;background:#e22;z-index:5;pointer-events:none}}.legend{{display:flex;gap:16px;font-size:12px;color:#5b6670}}.swatch{{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:4px}}</style>
<main><h1>Live person–vehicle review</h1><p class="hint">Original videos are used directly. Drag the bar, pause, or step one frame at a time.</p>
<div class="bar"><select id="clip"></select><button onclick="step(-1)">◀ frame</button><button onclick="step(1)">frame ▶</button><span id="time">0.00 s</span></div>
<video id="video" controls preload="metadata"></video><canvas id="view"></canvas><div class="bar"><input id="seek" type="range" min="0" max="1" step="0.001" value="0"><button onclick="video.currentTime=0">⏮ start</button><button onclick="video.play()">▶ play</button><button onclick="video.pause()">⏸ pause</button></div><p class="hint">Magenta boxes and labels show algorithm tracks; yellow labels show GT intervals and actions.</p>
<section class="timeline-card"><h2>Interaction timeline</h2><div class="legend"><span><i class="swatch" style="background:#b225b8"></i>Algorithm</span><span><i class="swatch" style="background:#cf8c00"></i>Ground truth</span></div><div class="timeline-head"><div></div><div id="ticks" class="ticks"></div></div><div id="timeline"></div><p class="hint">Click the timeline to seek. The red line shows the current video time.</p></section>
<section class="timeline-card"><h2>Active VLM description</h2><div id="active-description" class="hint">No active VLM description.</div></section>
<h2>Algorithm events</h2><div id="events"></div><h2>VLM descriptions</h2><div id="descriptions"></div><h2>Ground truth</h2><div id="gt"></div></main>
<script>const clips={payload};let video=document.querySelector('#video'),seek=document.querySelector('#seek'),select=document.querySelector('#clip'),canvas=document.querySelector('#view'),ctx=canvas.getContext('2d'),tracks={{}};
for(const c of clips){{let o=document.createElement('option');o.textContent=c.clip_id;select.append(o)}}
function fmt(x){{return Number(x).toFixed(2)+' s'}} function spans(e){{return (e.spans||[]).map(s=>fmt(s.start_s)+'–'+fmt(s.end_s)).join(', ')}}
function renderList(id,list,cls){{let box=document.querySelector(id);box.innerHTML=''; if(!list.length){{box.innerHTML='<p class="hint">None</p>';return}} for(const e of list){{let d=document.createElement('div');d.className='event '+(cls||'');d.textContent=e.event_id+' · '+e.type+' · '+spans(e)+' · '+(e.persons?.[0]?.person_id||'')+' → '+(e.vehicle?.vehicle_id||'');d.onclick=()=>{{let s=e.spans[0];video.currentTime=s.start_s;video.pause()}};box.append(d)}}}}
function renderDescriptions(list){{let box=document.querySelector('#descriptions');box.innerHTML='';if(!list.length){{box.innerHTML='<p class="hint">None</p>';return}}for(const d of list){{let x=document.createElement('div');x.className='event';let head=document.createElement('b');head.textContent=fmt(d.start_s)+'–'+fmt(d.end_s)+' · '+(d.final_type||'interaction')+' · '+((d.person_ids||[d.person_id]).filter(Boolean).join(', '))+' → '+(d.vehicle_id||'');let body=document.createElement('span');body.className='hint';body.textContent=d.description||'';x.append(head,document.createElement('br'),body);x.onclick=()=>{{video.currentTime=Number(d.start_s)||0;video.pause()}};box.append(x)}}}}
function updateDescription(c){{let active=(c.descriptions||[]).filter(d=>Number(d.start_s)<=video.currentTime&&video.currentTime<Number(d.end_s));document.querySelector('#active-description').textContent=active.length?active.map(d=>d.description).join(' | '):'No active VLM description.'}}
function timelineRow(label,list,kind,duration){{let row=document.createElement('div');row.className='timeline-row';let name=document.createElement('div');name.className='timeline-label';name.textContent=label;let track=document.createElement('div');track.className='timeline-track';let items=list.flatMap(e=>(e.spans||[]).map(s=>({{e,s}}))).sort((a,b)=>a.s.start_s-b.s.start_s),lanes=[];for(const item of items){{let lane=lanes.findIndex(end=>end<=item.s.start_s);if(lane<0){{lane=lanes.length;lanes.push(0)}}lanes[lane]=item.s.end_s;let seg=document.createElement('div');seg.className='segment '+kind;seg.style.left=(100*item.s.start_s/duration)+'%';seg.style.width=(100*Math.max(0,item.s.end_s-item.s.start_s)/duration)+'%';seg.style.top=(3+lane*25)+'px';let person=item.e.persons?.map(p=>p.person_id).join(', ')||'',vehicle=item.e.vehicle?.vehicle_id||'';seg.textContent=item.e.type;seg.title=item.e.event_id+' | '+item.e.type+' | '+fmt(item.s.start_s)+' - '+fmt(item.s.end_s)+' | '+person+' -> '+vehicle;seg.onclick=ev=>{{ev.stopPropagation();video.currentTime=item.s.start_s;video.pause()}};track.append(seg)}}track.style.height=Math.max(29,7+lanes.length*25)+'px';track.onclick=ev=>{{let r=track.getBoundingClientRect();video.currentTime=Math.max(0,Math.min(duration,(ev.clientX-r.left)/r.width*duration));video.pause()}};let head=document.createElement('div');head.className='playhead';track.append(head);row.append(name,track);return row}}
function updatePlayhead(){{let c=clips[select.selectedIndex];if(!c)return;let pct=100*Math.max(0,Math.min(1,video.currentTime/Math.max(c.duration_s,.001)));document.querySelectorAll('.playhead').forEach(x=>x.style.left=pct+'%')}}
function renderTimeline(c){{let duration=Math.max(Number(c.duration_s)||.001,.001),ticks=document.querySelector('#ticks'),box=document.querySelector('#timeline');ticks.innerHTML='';box.innerHTML='';for(let i=0;i<=10;i++){{let tick=document.createElement('span');tick.className='tick';tick.style.left=(i*10)+'%';tick.textContent=fmt(duration*i/10);ticks.append(tick)}}box.append(timelineRow('Algorithm',c.interactions,'alg',duration),timelineRow('GT',c.reference,'gt-segment',duration));updatePlayhead()}}
function load(){{let c=clips[select.selectedIndex];video.src=c.source;video.load();tracks=c.tracks||[];renderList('#events',c.interactions,'');renderDescriptions(c.descriptions||[]);renderList('#gt',c.reference,'gt');renderTimeline(c);updateDescription(c)}}
function draw(){{if(!video.videoWidth)return;canvas.width=video.videoWidth;canvas.height=video.videoHeight;ctx.drawImage(video,0,0);let c=clips[select.selectedIndex],row=(tracks||[]).reduce((a,b)=>Math.abs(b.timestamp_s-video.currentTime)<Math.abs(a.timestamp_s-video.currentTime)?b:a,{{timestamp_s:Infinity,objects:[]}}),objs=row.objects||[];let ae=c.interactions.filter(e=>e.spans.some(s=>s.start_s<=video.currentTime&&video.currentTime<s.end_s));let ids=new Set(ae.flatMap(e=>[e.vehicle.vehicle_id,...e.persons.map(p=>p.person_id)]));for(const o of objs)if(ids.has(o.id)){{let [x,y,w,h]=o.bbox;ctx.strokeStyle='#e646e6';ctx.lineWidth=2;ctx.strokeRect(x,y,w-x,h-y);ctx.fillStyle='#e646e6';ctx.font='18px Arial';ctx.fillText(o.id,x,Math.max(20,y-4))}}let activeGT=c.reference.filter(e=>e.spans.some(s=>s.start_s<=video.currentTime&&video.currentTime<s.end_s));ctx.font='20px Arial';let lines=[...ae.map(e=>'ALG '+e.type),...activeGT.map(e=>'GT '+e.type)];ctx.fillStyle='rgba(0,0,0,.62)';ctx.fillRect(0,0,canvas.width,lines.length*28+8);lines.forEach((x,i)=>{{ctx.fillStyle=x.startsWith('ALG')?'#ff70ff':'#ffe05b';ctx.fillText(x,8,25+i*28)}});requestAnimationFrame(draw)}}
function step(n){{video.pause();let c=clips[select.selectedIndex],i=(tracks||[]).findIndex(r=>r.timestamp_s>=video.currentTime);if(i<0)i=tracks.length-1;video.currentTime=tracks[Math.max(0,Math.min(tracks.length-1,i+n))]?.timestamp_s??Math.max(0,video.currentTime+n/30)}}
video.addEventListener('timeupdate',()=>{{seek.value=video.currentTime;seek.max=video.duration||1;document.querySelector('#time').textContent=fmt(video.currentTime);updatePlayhead();updateDescription(clips[select.selectedIndex])}});video.addEventListener('play',()=>requestAnimationFrame(draw));video.addEventListener('seeked',()=>{{updatePlayhead();updateDescription(clips[select.selectedIndex]);requestAnimationFrame(draw)}});seek.oninput=()=>video.currentTime=Number(seek.value);select.onchange=load;load();</script>'''
    destination.write_text(page, encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
