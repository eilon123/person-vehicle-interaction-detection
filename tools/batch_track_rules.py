"""Apply experimental appearance splitting and optional emergence recovery."""
import argparse, json, math
from pathlib import Path
import cv2
import numpy as np

def ctr(b): return ((b[0]+b[2])/2,(b[1]+b[3])/2)
def area(b): return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def inter(a,b): return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def iou(a,b):
 x=inter(a,b); return x/max(area(a)+area(b)-x,1)
def feat(im,b):
 x1,y1,x2,y2=map(int,b); c=im[max(0,y1):max(y1+1,y2),max(0,x1):max(x1+1,x2)]; c=c[:max(1,round(len(c)*.75))]
 h=cv2.calcHist([cv2.cvtColor(c,cv2.COLOR_BGR2HSV)],[0,1],None,[24,16],[0,180,0,256]); return cv2.normalize(h,h).flatten().astype(np.float32)
def near_vehicle(row,b):
 for v in row['objects']:
  if v['kind']=='person': continue
  q=v['bbox']; m=.15*max(q[2]-q[0],q[3]-q[1]);
  if b[0]<=q[2]+m and b[2]>=q[0]-m and b[1]<=q[3]+m and b[3]>=q[1]-m: return True
 return False
def split_appearance(rows,ims):
 ids={}
 for r in rows:
  for o in r['objects']:
   if o['kind']=='person' and o.get('observed',True): ids.setdefault(o['id'],[]).append((r['frame_index'],o))
 splits=[]
 for pid,obs in ids.items():
  if len(obs)<6: continue
  fs=[feat(ims[f],o['bbox']) for f,o in obs]
  for i in range(2,len(obs)-2):
   frame_gap=obs[i+1][0]-obs[i][0]
   area_ratio=area(obs[i+1][1]['bbox'])/max(area(obs[i][1]['bbox']),1)
   if frame_gap>2 or not .6<=area_ratio<=1.6: continue
   d=cv2.compareHist(fs[i],fs[i+1],cv2.HISTCMP_BHATTACHARYYA)
   old=np.mean(fs[i-2:i+1],0).astype(np.float32); new=np.mean(fs[i+1:i+3],0).astype(np.float32)
   if d>=.42 and cv2.compareHist(old,new,cv2.HISTCMP_BHATTACHARYYA)>=.38 and cv2.compareHist(fs[i+1],fs[i+2],cv2.HISTCMP_BHATTACHARYYA)<=.32 and (near_vehicle(rows[obs[i][0]],obs[i][1]['bbox']) or near_vehicle(rows[obs[i+1][0]],obs[i+1][1]['bbox'])):
    splits.append((pid,obs[i+1][0],pid+'b')); break
 for pid,frame,new in splits:
  for r in rows:
   if r['frame_index']>=frame:
    for o in r['objects']:
     if o['id']==pid: o['raw_id']=o.get('raw_id',pid); o['id']=new
 return splits
def emergence(rows,raw):
 tracks={}
 for r in rows:
  for o in r['objects']:
   if o['kind']=='person' and o.get('observed',True): tracks.setdefault(o['id'],[]).append((r['frame_index'],o))
 cues=[]
 for pid,obs in list(tracks.items()):
  if len(obs)>=2: continue
  sf,so=obs[0]; vehicles=[]
  for o in rows[sf]['objects']:
   if o['kind']=='person': continue
   q=o['bbox']; pw=max(1,so['bbox'][2]-so['bbox'][0]); vw=max(1,q[2]-q[0])
   xgap=max(0,q[0]-so['bbox'][2],so['bbox'][0]-q[2]); yover=max(0,min(so['bbox'][3],q[3])-max(so['bbox'][1],q[1]))
   # An emerging person is commonly first detected just outside the vehicle box.
   if inter(so['bbox'],q)>0 or (xgap<=max(24,.12*vw) and yover>=.20*min(so['bbox'][3]-so['bbox'][1],q[3]-q[1])): vehicles.append(o)
  if not vehicles: continue
  # Duplicate vehicle boxes are common; prefer the full, larger vehicle box.
  vehicle=max(vehicles,key=lambda v:(area(v['bbox']),inter(so['bbox'],v['bbox']))); vid=vehicle['id']; lastf=sf; last=so['bbox']; vel=(0,0); linked=[(sf,last,so['confidence'])]
  for f in range(sf+1,min(len(rows),sf+16)):
   gap=f-lastf
   if gap>5: break
   v=next((o for o in rows[f]['objects'] if o['id']==vid),None)
   if not v: continue
   pred=(ctr(last)[0]+vel[0]*gap,ctr(last)[1]+vel[1]*gap); vw=max(1,v['bbox'][2]-v['bbox'][0]); choices=[]
   for box in raw.get(f,[]):
    b=box[:4]; dist=math.dist(pred,ctr(b))
    if dist<=max(55,.18*vw*gap): choices.append((dist/vw-.03*min(area(b)/max(area(last),1),4),-area(b),b,box[4]))
   if not choices: continue
   _,_,b,conf=min(choices); a,c=ctr(last),ctr(b); vel=((c[0]-a[0])/gap,(c[1]-a[1])/gap); linked.append((f,b,conf)); lastf,last=f,b
  growth=area(linked[-1][1])/max(area(linked[0][1]),1)
  if len(linked)<3 or growth<3: continue
  # Reject a detector duplicate of an already tracked person.
  duplicate_frames=0
  for f,b,_ in linked:
   if any(o['kind']=='person' and o['id']!=pid and iou(b,o['bbox'])>.5 for o in rows[f]['objects']): duplicate_frames+=1
  if duplicate_frames>=2: continue
  # Require sustained movement away from the associated vehicle.
  distances=[]
  for f,b,_ in linked:
   v=next((o for o in rows[f]['objects'] if o['id']==vid),None)
   if v: distances.append(math.dist(ctr(b),ctr(v['bbox']))/max(v['bbox'][2]-v['bbox'][0],v['bbox'][3]-v['bbox'][1],1))
  if len(distances)<3 or distances[-1]-distances[0]<.5: continue
  outward=sum(b>=a-.03 for a,b in zip(distances,distances[1:]))
  if outward < math.ceil(.7*(len(distances)-1)): continue
  by={f:(b,c) for f,b,c in linked}
  for r in rows:
   r['objects']=[o for o in r['objects'] if o['id']!=pid]
   if r['frame_index'] in by:
    b,c=by[r['frame_index']]; r['objects'].append({'id':pid,'kind':'person','bbox':[round(x,2) for x in b],'confidence':round(float(c),4),'observed':True,'source':'emergence'})
  first_t=rows[sf]['timestamp_s']; start=max(0,first_t-2.0); end=min(rows[linked[-1][0]]['timestamp_s'],first_t+.5)
  cues.append({'person_id':pid,'vehicle_id':vid,'start_s':start,'end_s':end,'frames':[x[0] for x in linked],'area_growth':growth})
 return cues
def main():
 p=argparse.ArgumentParser(); p.add_argument('--videos',type=Path,required=True); p.add_argument('--source-tracks',type=Path,required=True); p.add_argument('--raw',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--emergence',action='store_true'); p.add_argument('--no-appearance-split',action='store_true'); a=p.parse_args()
 for video in sorted(a.videos.glob('*.mp4')):
  src=a.source_tracks/f'{video.stem}.jsonl'
  if not src.exists(): continue
  rows=[json.loads(x) for x in src.read_text().splitlines()]; cap=cv2.VideoCapture(str(video)); ims=[]
  while True:
   ok,im=cap.read()
   if not ok: break
   ims.append(im)
  splits=[] if a.no_appearance_split else split_appearance(rows,ims); cues=[]
  if a.emergence:
   raw={x['frame']:x['boxes'] for x in json.loads((a.raw/f'{video.stem}.json').read_text())}; cues=emergence(rows,raw)
  out=a.output/'tracks'/f'{video.stem}.jsonl'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
  (a.output/'rules').mkdir(parents=True,exist_ok=True); (a.output/'rules'/f'{video.stem}.json').write_text(json.dumps({'appearance_splits':splits,'emergence_cues':cues},indent=2))
  print(video.stem,'splits',splits,'emergence',len(cues),flush=True)
if __name__=='__main__': main()
