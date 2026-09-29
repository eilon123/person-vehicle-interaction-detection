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
    parser.add_argument("--reference", help="Optional GT events.json")
    args = parser.parse_args()
    input_dir, event_dir, destination = Path(args.input), Path(args.events), Path(args.output)
    clips = []
    reference = read_json(args.reference) if args.reference else {}
    for event_path in sorted(event_dir.glob("*.json")):
        clip = read_json(event_path)
        source = next((p for p in input_dir.glob(f"{clip['clip_id']}.*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}), None)
        if not source or clip.get("status") != "ok":
            continue
        clips.append({"clip_id": clip["clip_id"], "source": source.resolve().as_uri(),
                      "duration_s": clip["duration_s"], "interactions": clip["interactions"],
                      "reference": reference.get(clip["clip_id"], {}).get("interactions", [])})
    payload = json.dumps(clips, ensure_ascii=False).replace("</", "<\\/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    title = destination.stem.replace("_", " ")
    page = f'''<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:15px Segoe UI,Arial;margin:24px;background:#f4f6f8;color:#18212b}}main{{max-width:1100px;margin:auto}}video{{width:100%;background:#111}}.bar{{display:flex;gap:8px;align-items:center;margin:10px 0;flex-wrap:wrap}}button,select{{padding:7px 10px}}input[type=range]{{flex:1;min-width:280px}}.event{{padding:7px 10px;border-left:5px solid #c020c0;background:#fff;margin:5px 0;cursor:pointer}}.gt{{border-color:#e0a400}}.hint{{color:#5b6670}}#time{{font-variant-numeric:tabular-nums}}</style>
<main><h1>Live person–vehicle review</h1><p class="hint">Original videos are used directly. Drag the bar, pause, or step one frame at a time.</p>
<div class="bar"><select id="clip"></select><button onclick="step(-1)">◀ frame</button><button onclick="step(1)">frame ▶</button><span id="time">0.00 s</span></div>
<video id="video" controls preload="metadata"></video><div class="bar"><input id="seek" type="range" min="0" max="1" step="0.001" value="0"><button onclick="video.currentTime=0">⏮ start</button><button onclick="video.play()">▶ play</button><button onclick="video.pause()">⏸ pause</button></div>
<h2>Algorithm events</h2><div id="events"></div><h2>Ground truth</h2><div id="gt"></div></main>
<script>const clips={payload};let video=document.querySelector('#video'),seek=document.querySelector('#seek'),select=document.querySelector('#clip');
for(const c of clips){{let o=document.createElement('option');o.textContent=c.clip_id;select.append(o)}}
function fmt(x){{return Number(x).toFixed(2)+' s'}} function spans(e){{return (e.spans||[]).map(s=>fmt(s.start_s)+'–'+fmt(s.end_s)).join(', ')}}
function renderList(id,list,cls){{let box=document.querySelector(id);box.innerHTML=''; if(!list.length){{box.innerHTML='<p class="hint">None</p>';return}} for(const e of list){{let d=document.createElement('div');d.className='event '+(cls||'');d.textContent=e.event_id+' · '+e.type+' · '+spans(e)+' · '+(e.persons?.[0]?.person_id||'')+' → '+(e.vehicle?.vehicle_id||'');d.onclick=()=>{{let s=e.spans[0];video.currentTime=s.start_s;video.pause()}};box.append(d)}}}}
function load(){{let c=clips[select.selectedIndex];video.src=c.source;video.load();renderList('#events',c.interactions,'');renderList('#gt',c.reference,'gt')}}
function step(n){{video.pause();video.currentTime=Math.max(0,video.currentTime+n/(video.webkitDecodedFrameCount?30:30))}}
video.addEventListener('timeupdate',()=>{{seek.value=video.currentTime;seek.max=video.duration||1;document.querySelector('#time').textContent=fmt(video.currentTime)}});seek.oninput=()=>video.currentTime=Number(seek.value);select.onchange=load;load();</script>'''
    destination.write_text(page, encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
