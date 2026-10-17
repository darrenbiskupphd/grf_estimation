#!/usr/bin/env python3
"""Generate a standalone, offline, reference-only HTML inspection (no GUI opens).

  python python/mocap_preview.py mocap/cmu/143/143_38.reference.json \
    --c3d mocap/cmu/143/143_38.c3d \
    --output mocap/cmu/143/143_38.html --force

Open the resulting HTML in any modern browser. Play at native timestamps, step
samples, scrub, rotate/zoom, inspect all raw markers and the 16 named tracking
landmarks, or focus on the feet and their heights above Z=0. Supply --c3d to
embed raw trajectories; invalid observations are hidden, never filled.
Drawing lines are visual aids, NOT bones. No simulation, interpolation, retiming,
network assets or ground fitting occurs. All frames and
provenance are embedded; no web server or plotting dependency is necessary.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import numpy as np

from mocap_import import (ImportValidationError, load_json, validate_reference,
                          convert_points, read_c3d, UNIT_TO_M)


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ — reference inspection</title>
<style>
:root{color-scheme:light}body{font:15px system-ui,sans-serif;margin:1rem;background:#f5f7fa;color:#172b42}h1{font-size:1.3rem;margin:.4rem 0}.warning{background:#fff1c4;border:1px solid #ba8100;padding:.7rem;border-radius:6px}.controls{display:flex;flex-wrap:wrap;gap:.65rem;align-items:center;margin:.65rem 0}button,select{padding:.4rem}input[type=range]{flex:1;min-width:180px}main{display:grid;grid-template-columns:minmax(450px,2fr) minmax(360px,1fr);gap:1rem}canvas{width:100%;background:white;border:1px solid #bbc8d6;border-radius:5px}#view{height:560px;touch-action:none;cursor:grab}#heights{height:190px}table{border-collapse:collapse;width:100%;font:13px ui-monospace,monospace;background:white}td,th{padding:.25rem;text-align:right;border-bottom:1px solid #dde3eb}td:first-child,th:first-child{text-align:left}tr.selected{background:#f9e3ff}tbody tr{cursor:pointer}.small{font-size:13px;color:#425267}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:450px;overflow:auto;background:#eef2f6;padding:.5rem}details{margin:.6rem 0}#definition{background:white;padding:.7rem;border-radius:5px;min-height:6rem}#mapping td{text-align:left;font:13px system-ui;vertical-align:top}@media(max-width:950px){main{grid-template-columns:1fr}#view{height:450px}}
</style></head><body>
<h1>__TITLE__ — reference-only preview</h1>
<div class="small">Right-handed XYZ · Z up · ground Z=0 · lines are visual guides, not bones.</div>
<div id="summary" class="small"></div>
<div class="controls">
<button id="play">Play</button><button id="previous" title="Previous frame" aria-label="Previous frame">◀</button><button id="next" title="Next frame" aria-label="Next frame">▶</button>
<select id="speed" aria-label="Playback speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option></select>
<input id="frame" aria-label="Source sample" type="range" min="0" step="1"><output id="clock"></output>
</div>
<main><section>
<div class="controls">
<button id="reset">Reset camera</button>
<label><input id="raw" type="checkbox">Raw markers</label><label><input id="tracking" type="checkbox" checked>Tracking landmarks</label>
<label><input id="labels" type="checkbox" checked>Names</label><label><input id="follow" type="checkbox" checked>Follow pelvis</label><label><input id="feet" type="checkbox">Feet close-up</label>
</div>
<canvas id="view" aria-label="Named targets and ground plane"></canvas>
<div class="small">Drag to rotate · scroll to zoom · click to select. Gray: raw markers · gold rings: selected source markers · blue/orange: left/right tracking landmarks · purple: midline · diamond: derived point.</div>
<h3>Foot height (m)</h3><canvas id="heights" aria-label="Full native toe and heel height traces"></canvas>
<div class="small">Dashed red: Z=0 · click to seek. Negative heights are unchanged.</div>
</section><aside>
<h3>Landmarks (m)</h3>
<table><thead><tr><th>Target</th><th>X</th><th>Y</th><th>Z</th><th>min Z</th></tr></thead><tbody id="targets"></tbody></table>
<h3>Selected landmark source</h3><div id="definition"></div>
</aside></main>
<details><summary>Marker mapping</summary><p class="small">How C3D markers produce the 16 landmarks. Includes anatomical caveats and source documentation.</p><table id="mapping"><thead><tr><th>Landmark</th><th>C3D source / expression</th><th>Meaning and caveats</th><th>Evidence</th></tr></thead><tbody></tbody></table></details>
<details><summary>Import metadata (JSON)</summary><p class="small">Debug record: source file/checksum, native timing, coordinate transform and mapping settings.</p><pre id="metadata"></pre></details>
<script id="reference-data" type="application/json">__DATA__</script>
<script>
'use strict';
const ref=JSON.parse(document.getElementById('reference-data').textContent);
const names=ref.target_names, times=ref.time_s, samples=ref.positions_m, meta=ref.metadata;
const raw=ref.preview_markers||null;
const cfg=(meta.mapping||{}).configuration||{}, specs=cfg.targets||{};
const $=id=>document.getElementById(id), canvas=$('view'), chart=$('heights');
if(!raw){$('raw').checked=false;$('raw').disabled=true;$('raw').parentElement.title='Regenerate with the source C3D to show raw markers';}
const sides=names.map(n=>n.startsWith('l')?'#116bb5':n.startsWith('r')?'#cb5500':'#8436ab');
const links=[['pelvis','head'],['lshoulder','rshoulder'],['lshoulder','lelbow'],['lelbow','lhand'],['rshoulder','relbow'],['relbow','rhand'],['lhip','rhip'],['lhip','lknee'],['rhip','rknee'],['lknee','lheel'],['rknee','rheel'],['lheel','ltoe'],['rheel','rtoe']].map(e=>e.map(n=>names.indexOf(n)));
const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity],minZ=names.map(()=>Infinity);
for(const points of samples)points.forEach((p,i)=>p.forEach((v,k)=>{lo[k]=Math.min(lo[k],v);hi[k]=Math.max(hi[k],v);if(k===2)minZ[i]=Math.min(minZ[i],v);}));
if(raw)for(const points of raw.positions_m)for(const p of points)if(p)p.forEach((v,k)=>{lo[k]=Math.min(lo[k],v);hi[k]=Math.max(hi[k],v);});
lo[2]=Math.min(0,lo[2]);hi[2]=Math.max(0,hi[2]);
let frame=0,selected=names.indexOf('pelvis'),yaw=-.6,elevation=.35,zoom=1,playing=false,playStart=0,timeStart=0,projected=[];
$('frame').max=samples.length-1;
const native=(meta.timing||{}).native_rate_hz;
$('summary').textContent=`${samples.length} frames · ${times.at(-1).toFixed(2)} s · ${native?native+' Hz':'native timestamps'}`;
$('metadata').textContent=JSON.stringify(meta,null,2);
const cells=[];
function sourceLabels(n){return ((meta.mapping||{}).resolved_labels||{})[n]||[];}
function sourceText(n){const s=specs[n];if(!s)return 'No mapping recorded';const labels=((meta.mapping||{}).resolved_labels||{})[n]||(s.labels||[s.label]);return s.kind==='marker'?labels[0]:'weighted_mean('+labels.map((l,i)=>`${s.weights[i]} × ${l}`).join(' + ')+')';}
names.forEach((name,i)=>{
 const row=document.createElement('tr'),values=[];for(let k=0;k<5;k++){const td=document.createElement('td');row.appendChild(td);values.push(td);}values[0].textContent=name;values[0].style.color=sides[i];values[4].textContent=minZ[i].toFixed(4);row.onclick=()=>{selected=i;draw();};$('targets').appendChild(row);cells.push({row,values});
 const mr=document.createElement('tr'),s=specs[name]||{};for(const text of [name,sourceText(name),(s.anatomical_meaning||'Unknown')+' '+(s.limitations||[]).join(' '),(s.evidence||[]).join('; ')]){const td=document.createElement('td');td.textContent=text;mr.appendChild(td);}$('mapping').querySelector('tbody').appendChild(mr);
});
function context(c){const box=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=box.width,h=box.height;if(c.width!==Math.round(w*dpr)||c.height!==Math.round(h*dpr)){c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);}const ctx=c.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);return [ctx,w,h];}
function projection(w,h){
 const points=samples[frame],foot=$('feet').checked,follow=$('follow').checked;
 let center=lo.map((v,k)=>(v+hi[k])/2),span=lo.map((v,k)=>hi[k]-v+.3);
 if(follow){center[0]=points[0][0];center[1]=points[0][1];span=[2.2,2.2,2.1];center[2]=.85;}
 if(foot){const f=[2,3,4,5].map(i=>points[i]);center=[0,1].map(k=>f.reduce((a,p)=>a+p[k],0)/4);center.push(.15);span=[1.6,1.6,.7];}
 const cy=Math.cos(yaw),sy=Math.sin(yaw),ce=Math.cos(elevation),se=Math.sin(elevation);
 const rw=Math.abs(cy)*span[0]+Math.abs(sy)*span[1],rh=Math.abs(se*sy)*span[0]+Math.abs(se*cy)*span[1]+Math.abs(ce)*span[2];
 const scale=.85*Math.min(w/Math.max(rw,.1),h/Math.max(rh,.1))*zoom;
 return p=>{const x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];return [w/2+scale*(cy*x-sy*y),h/2-scale*(se*(sy*x+cy*y)+ce*z)];};
}
function line(ctx,project,a,b,color,width=1,dash=[]){const p=project(a),q=project(b);ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();ctx.setLineDash([]);}
function drawView(){
 const [ctx,w,h]=context(canvas),project=projection(w,h),pts=samples[frame];ctx.font='12px system-ui';
 const xmin=Math.floor(lo[0]-.5),xmax=Math.ceil(hi[0]+.5),ymin=Math.floor(lo[1]-.5),ymax=Math.ceil(hi[1]+.5);
 const corners=[[xmin,ymin,0],[xmax,ymin,0],[xmax,ymax,0],[xmin,ymax,0]].map(project);ctx.fillStyle='#e9f0e9';ctx.beginPath();corners.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fill();
 const step=Math.max(.5,Math.ceil(Math.max(xmax-xmin,ymax-ymin)/40));
 for(let x=xmin;x<=xmax;x+=step)line(ctx,project,[x,ymin,0],[x,ymax,0],'#b9cabc');
 for(let y=ymin;y<=ymax;y+=step)line(ctx,project,[xmin,y,0],[xmax,y,0],'#b9cabc');
 const colors=['#b22c35','#258247','#2867ad'];for(let k=0;k<3;k++){const p=[0,0,0];p[k]=.5;line(ctx,project,[0,0,0],p,colors[k],2);ctx.fillStyle=colors[k];const q=project(p);ctx.fillText('+'+'XYZ'[k]+' (0.5 m)',q[0]+4,q[1]-4);}
 ctx.fillStyle='#425267';ctx.fillText(`Ground Z=0 plane · grid ${step} m · ${$('feet').checked?'feet camera':'equal metric scale'}`,10,h-12);
 // Pelvis path uses all native samples, for spatial context only.
 ctx.beginPath();ctx.strokeStyle='#c9b2d3';ctx.lineWidth=1;samples.forEach((p,i)=>{const q=project(p[0]);i?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.stroke();
 if(raw&&$('raw').checked){const contributors=new Set(sourceLabels(names[selected]));raw.positions_m[frame].forEach((p,i)=>{if(!p)return;const q=project(p),active=contributors.has(raw.labels[i]);ctx.beginPath();ctx.arc(...q,active?9:3,0,2*Math.PI);ctx.fillStyle=active?'#f5cf74':'#929da8';ctx.fill();if(active){ctx.strokeStyle='#886000';ctx.lineWidth=2;ctx.stroke();}if($('labels').checked||active){ctx.fillStyle=active?'#745000':'#606c78';ctx.font='11px system-ui';ctx.fillText(raw.labels[i],q[0]+10,q[1]+13);}if(active&&$('tracking').checked)line(ctx,project,p,pts[selected],'#b38b32',1,[2,3]);});}
 projected=[];
 if(!$('tracking').checked)return;
 for(const [a,b] of links)line(ctx,project,pts[a],pts[b],'#8094a5',1.5,[4,3]);
 projected=pts.map(project);projected.forEach((p,i)=>{const derived=(specs[names[i]]||{}).kind==='weighted_mean';ctx.beginPath();if(derived){ctx.moveTo(p[0],p[1]-6);ctx.lineTo(p[0]+6,p[1]);ctx.lineTo(p[0],p[1]+6);ctx.lineTo(p[0]-6,p[1]);ctx.closePath();}else ctx.arc(...p,5,0,2*Math.PI);ctx.fillStyle=sides[i];ctx.fill();ctx.strokeStyle=i===selected?'#171b22':'white';ctx.lineWidth=i===selected?3:1;ctx.stroke();if($('labels').checked||i===selected){ctx.font=(i===selected?'bold ':'')+'12px system-ui';ctx.strokeStyle='white';ctx.lineWidth=3;ctx.strokeText(names[i],p[0]+8,p[1]-6);ctx.fillStyle=sides[i];ctx.fillText(names[i],p[0]+8,p[1]-6);}});
}
const footColors=['#116bb5','#cb5500','#179dc2','#9b6815'];
function drawChart(){
 const [ctx,w,h]=context(chart),left=48,right=w-12,top=28,bottom=h-28,duration=times.at(-1);
 let zlo=0,zhi=.05;for(const pts of samples)for(const i of [2,3,4,5]){zlo=Math.min(zlo,pts[i][2]);zhi=Math.max(zhi,pts[i][2]);}zlo-=.015;zhi+=.015;
 const x=t=>left+(right-left)*t/duration,y=z=>bottom-(bottom-top)*(z-zlo)/(zhi-zlo);ctx.font='12px system-ui';
 for(let j=0;j<=4;j++){const z=zlo+j*(zhi-zlo)/4;ctx.fillStyle='#425267';ctx.fillText(z.toFixed(3),2,y(z)+4);ctx.strokeStyle='#e4e8ee';ctx.beginPath();ctx.moveTo(left,y(z));ctx.lineTo(right,y(z));ctx.stroke();}
 ctx.strokeStyle='#c33838';ctx.setLineDash([5,3]);ctx.beginPath();ctx.moveTo(left,y(0));ctx.lineTo(right,y(0));ctx.stroke();ctx.setLineDash([]);
 [2,3,4,5].forEach((i,k)=>{ctx.beginPath();ctx.strokeStyle=footColors[k];ctx.lineWidth=1.7;samples.forEach((p,j)=>{const q=[x(times[j]),y(p[i][2])];j?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.stroke();ctx.fillStyle=footColors[k];ctx.fillText(names[i],left+k*70,16);});
 ctx.strokeStyle='#333';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(times[frame]),top);ctx.lineTo(x(times[frame]),bottom);ctx.stroke();ctx.fillStyle='#425267';ctx.fillText('0 s',left,h-7);ctx.fillText(duration.toFixed(3)+' s',right-65,h-7);
}
function draw(){
 $('frame').value=frame;const first=(meta.timing||{}).first_frame;const raw=first===undefined?'':` · C3D frame ${first+frame}`;
 $('clock').textContent=`${times[frame].toFixed(4)} / ${times.at(-1).toFixed(4)} s · sample ${frame}/${samples.length-1}${raw}`;
 cells.forEach(({row,values},i)=>{row.classList.toggle('selected',i===selected);samples[frame][i].forEach((v,k)=>values[k+1].textContent=v.toFixed(4));values[3].style.color=samples[frame][i][2]<0?'#b42121':'';});
 const name=names[selected],s=specs[name]||{};$('definition').textContent=`${name} ← ${sourceText(name)}. ${s.anatomical_meaning||'No definition recorded'}`;
 drawView();drawChart();
}
function stop(){playing=false;$('play').textContent='Play';}
function seek(i){stop();frame=Math.max(0,Math.min(samples.length-1,i));draw();}
function sampleAt(t){let a=0,b=times.length;while(a<b){const m=(a+b)>>1;if(times[m]<=t)a=m+1;else b=m;}return Math.max(0,a-1);}
$('frame').oninput=e=>seek(Number(e.target.value));$('previous').onclick=()=>seek(frame-1);$('next').onclick=()=>seek(frame+1);
$('play').onclick=()=>{if(playing){stop();return;}if(frame===samples.length-1)frame=0;playing=true;timeStart=times[frame];playStart=performance.now();$('play').textContent='Pause';requestAnimationFrame(tick);};
function tick(now){if(!playing)return;const t=timeStart+(now-playStart)/1000*Number($('speed').value);const next=sampleAt(t);if(next!==frame){frame=next;draw();}if(t>=times.at(-1)){stop();return;}requestAnimationFrame(tick);}
$('speed').onchange=()=>{timeStart=times[frame];playStart=performance.now();};
for(const id of ['labels','follow','feet','raw','tracking'])$(id).onchange=draw;
$('reset').onclick=()=>{yaw=-.6;elevation=.35;zoom=1;$('follow').checked=false;$('feet').checked=false;draw();};
let drag=null;canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,moved:false};canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>2)drag.moved=true;yaw+=dx*.008;elevation=Math.max(-1.55,Math.min(1.55,elevation+dy*.008));drag.x=e.clientX;drag.y=e.clientY;draw();};
canvas.onpointerup=e=>{if(drag&&!drag.moved){const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let distance=22;projected.forEach((p,i)=>{const d=Math.hypot(p[0]-x,p[1]-y);if(d<distance){distance=d;selected=i;}});draw();}drag=null;};
canvas.onpointercancel=()=>drag=null;canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.25,Math.min(12,zoom*Math.exp(-e.deltaY*.001)));draw();};
chart.onclick=e=>{const r=chart.getBoundingClientRect(),t=Math.max(0,Math.min(1,(e.clientX-r.left-48)/(r.width-60)))*times.at(-1);seek(sampleAt(t));};
window.addEventListener('resize',draw);draw();
</script></body></html>
'''


def render_preview(reference, *, points=None):
    validate_reference(reference)
    if points is not None:
        mapping = reference['metadata']['mapping']['configuration']
        rebuilt = convert_points(points, mapping)
        if (rebuilt['time_s'] != reference['time_s'] or
                rebuilt['positions_m'] != reference['positions_m']):
            raise ImportValidationError('Preview C3D does not match the reference')
        transform = mapping['world_from_source_m']
        positions = ((points.positions * UNIT_TO_M[points.units]) @
                     np.asarray(transform['rotation']).T + transform['translation_m'])
        raw_positions = [[p.tolist() if valid else None for p, valid in zip(frame, mask)]
                         for frame, mask in zip(positions, points.valid)]
        reference = {**reference, 'preview_markers': {
            'labels': list(points.labels), 'positions_m': raw_positions}}
    title = str(reference["metadata"].get("source", {}).get("source_id", "Mocap reference"))
    # Data must not be able to close the script element, including through a label
    # or provenance string from an untrusted input file.
    data = json.dumps(reference, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return HTML.replace("__TITLE__", html.escape(title)).replace("__DATA__", data)


def write_preview(path, reference, *, force=False, points=None):
    text = render_preview(reference, points=points)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if force else "x", encoding="utf-8") as stream:
        stream.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--c3d", type=Path, help="Matching source C3D for raw-marker audit")
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() in {args.reference.resolve(),
                                     args.c3d.resolve() if args.c3d else None}:
            raise ImportValidationError("Preview output cannot overwrite the reference")
        write_preview(args.output, load_json(args.reference), force=args.force,
                      points=read_c3d(args.c3d) if args.c3d else None)
    except (ValueError, OSError) as error:
        parser.exit(2, f"mocap preview: {error}\n")
    print(f"Open {args.output} in a browser (offline). Mocap reference preview.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
