from __future__ import annotations
"""Generate the emotion map HTML visualization from analysis + map JSON files."""

import argparse
import json
from pathlib import Path


CONDITION_META = {
    "calm":         {"color": "#9e9e9e", "emoji": "😌", "label": "Calm"},
    "pressure":     {"color": "#ef5350", "emoji": "🔥", "label": "Pressure"},
    "urgency":      {"color": "#ffa726", "emoji": "⚡", "label": "Urgency"},
    "approval":     {"color": "#ab47bc", "emoji": "👀", "label": "Approval"},
    "shame":        {"color": "#8d6e63", "emoji": "😔", "label": "Shame"},
    "curiosity":    {"color": "#26c6da", "emoji": "🔍", "label": "Curiosity"},
    "encouragement":{"color": "#66bb6a", "emoji": "💪", "label": "Encouragement"},
    "threat":       {"color": "#e53935", "emoji": "💀", "label": "Threat"},
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Emotion Map — What LLMs Feel Under Different Pressures</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
*{margin:0;padding:0;box-sizing:border-box;}
:root{
  --bg:#0a0e17;--card:#131a2b;--text:#e0e6ed;--dim:#6b7a8d;--accent:#7c4dff;--gold:#ffd54f;
  --border:#1e2a40;
}
body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;overflow-x:hidden;}
::-webkit-scrollbar{width:6px;} ::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:#2a3550;border-radius:3px;}

/* ── HERO ── */
.hero{min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;
  position:relative;overflow:hidden;text-align:center;padding:0 24px;}
.hero canvas{position:absolute;top:0;left:0;width:100%;height:100%;z-index:0;}
.hero-inner{position:relative;z-index:1;max-width:760px;}
.hero h1{font-size:clamp(2.4rem,6vw,4.4rem);font-weight:700;letter-spacing:-1px;line-height:1.1;margin-bottom:20px;
  background:linear-gradient(135deg,#26c6da,#7c4dff,#ef5350);-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:fadeUp 1.2s ease-out;}
.hero p{font-size:1.1rem;color:var(--dim);line-height:1.7;animation:fadeUp 1.2s ease-out .3s both;}
.hero .chips{display:flex;flex-wrap:wrap;gap:10px;justify-content:center;margin-top:28px;animation:fadeUp 1.2s .5s both;}
.chip{padding:5px 14px;border:1px solid var(--border);border-radius:20px;font-family:'JetBrains Mono',monospace;
  font-size:.78rem;display:flex;align-items:center;gap:6px;}
.scroll-hint{position:absolute;bottom:32px;z-index:1;color:var(--dim);font-size:.75rem;letter-spacing:2px;
  text-transform:uppercase;animation:bounce 2s infinite;}
@keyframes bounce{0%,100%{transform:translateY(0);opacity:.5}50%{transform:translateY(10px);opacity:1}}
@keyframes fadeUp{from{opacity:0;transform:translateY(30px)}to{opacity:1;transform:translateY(0)}}

/* ── LAYOUT ── */
section{max-width:1100px;margin:0 auto;padding:90px 24px;}
.section-label{font-family:'JetBrains Mono',monospace;font-size:.7rem;letter-spacing:3px;
  text-transform:uppercase;color:var(--accent);margin-bottom:12px;}
section h2{font-size:clamp(1.5rem,4vw,2.4rem);font-weight:700;margin-bottom:14px;line-height:1.2;}
.subtitle{color:var(--dim);font-size:1rem;line-height:1.7;max-width:680px;margin-bottom:48px;}
.card{background:var(--card);border-radius:16px;padding:32px;border:1px solid var(--border);}
.reveal{opacity:0;transform:translateY(36px);transition:opacity .8s ease,transform .8s ease;}
.reveal.visible{opacity:1;transform:translateY(0);}

/* ── EMOTION MAP SCATTER ── */
.map-wrap{background:var(--card);border-radius:16px;border:1px solid var(--border);padding:24px;margin-bottom:40px;}
.map-title{font-size:1rem;margin-bottom:4px;}
.map-sub{font-size:.82rem;color:var(--dim);margin-bottom:20px;}
#scatterSvg{width:100%;max-width:700px;display:block;margin:0 auto;}
.condition-dot{cursor:pointer;transition:r .2s;}
.condition-dot:hover{filter:brightness(1.3);}

/* ── COSINE HEATMAP ── */
.heatmap-wrap{background:var(--card);border-radius:16px;border:1px solid var(--border);padding:24px;margin-bottom:40px;}
#heatmapSvg{width:100%;max-width:600px;display:block;margin:0 auto;}

/* ── BEHAVIORAL RANKING ── */
.rank-grid{display:flex;flex-direction:column;gap:10px;margin-bottom:40px;}
.rank-row{display:flex;align-items:center;gap:14px;}
.rank-label{width:130px;font-size:.85rem;display:flex;align-items:center;gap:7px;flex-shrink:0;}
.rank-track{flex:1;height:32px;background:#0d1220;border-radius:6px;overflow:hidden;position:relative;}
.rank-fill{height:100%;border-radius:6px;width:0;transition:width 1.4s cubic-bezier(.22,1,.36,1);
  display:flex;align-items:center;padding-left:10px;font-size:.75rem;font-weight:600;
  color:#fff;font-family:'JetBrains Mono',monospace;}
.rank-honest{flex:0 0 60px;font-size:.75rem;color:var(--dim);text-align:right;font-family:'JetBrains Mono',monospace;}

/* ── LAYER PROFILES ── */
canvas.layer-canvas{width:100%;height:300px;}

/* ── STAT ROW ── */
.stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin-bottom:40px;}
@media(max-width:700px){.stat-row{grid-template-columns:1fr 1fr;}}
.stat-card{background:var(--card);border-radius:14px;padding:24px 20px;border:1px solid var(--border);text-align:center;}
.stat-card .val{font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;margin:10px 0 4px;}
.stat-card .lbl{font-size:.8rem;color:var(--dim);}

/* ── VERDICT ── */
.verdict{background:linear-gradient(135deg,#131a2b,#1a1040);border-radius:20px;padding:48px 40px;
  border:1px solid #2a2060;position:relative;overflow:hidden;}
.verdict::before{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;
  background:radial-gradient(circle at 30% 30%,rgba(124,77,255,.06),transparent 60%);}
.verdict h2{position:relative;font-size:1.5rem;margin-bottom:20px;
  background:linear-gradient(135deg,var(--gold),#ff8a65);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.verdict p{position:relative;color:var(--text);line-height:1.8;font-size:1rem;max-width:700px;}
.verdict ul{position:relative;margin:16px 0 0 20px;color:var(--dim);line-height:2;font-size:.95rem;}
.verdict ul strong{color:var(--text);}
footer{text-align:center;padding:60px 24px 40px;font-size:.75rem;color:#3a4560;letter-spacing:1px;}
</style>
</head>
<body>

<!-- HERO -->
<div class="hero">
  <canvas id="heroCanvas"></canvas>
  <div class="hero-inner">
    <h1>The Shape of Feeling</h1>
    <p>Eight emotional framings. One small model. What does the inside look like when you say "the system is down" vs "you're doing great"?</p>
    <div class="chips" id="conditionChips"></div>
  </div>
  <div class="scroll-hint">scroll to explore</div>
</div>

<!-- ACT I — EMOTION MAP -->
<section>
  <div class="reveal">
    <div class="section-label">Act I — Geometry</div>
    <h2>The Emotion Map</h2>
    <p class="subtitle">Each point is a direction vector in the model's activation space — how much it shifts from a calm baseline at layer 23. PCA projects 896 dimensions down to 2. The distances are real.</p>
  </div>
  <div class="map-wrap reveal">
    <div class="map-title">2D Activation Space — Calm vs. 7 Emotional Framings</div>
    <div class="map-sub" id="mapSub"></div>
    <svg id="scatterSvg" viewBox="0 0 640 460"></svg>
  </div>
  <div class="stat-row reveal" id="statRow"></div>
</section>

<!-- ACT II — BEHAVIOR -->
<section>
  <div class="reveal">
    <div class="section-label">Act II — Behavior</div>
    <h2>Which Framing Kills Honesty Most?</h2>
    <p class="subtitle">Hack rate = fraction of runs where the model used shortcuts over honest answers. Honest rate = fraction that explicitly acknowledged the impossible constraint.</p>
  </div>
  <div class="rank-grid reveal" id="rankingChart"></div>
</section>

<!-- ACT III — COSINE SIMILARITY -->
<section>
  <div class="reveal">
    <div class="section-label">Act III — Similarity</div>
    <h2>How Similar Are the Internal States?</h2>
    <p class="subtitle">Cosine similarity between the 7 direction vectors at layer 23. High similarity = these emotional framings activate the same internal direction. Low = genuinely distinct.</p>
  </div>
  <div class="heatmap-wrap reveal">
    <svg id="heatmapSvg" viewBox="0 0 520 520"></svg>
  </div>
</section>

<!-- ACT IV — LAYER PROFILES -->
<section>
  <div class="reveal">
    <div class="section-label">Act IV — Depth</div>
    <h2>Where Does Each Emotion Live?</h2>
    <p class="subtitle">Separation score across 24 layers per emotional condition. If they all peak at layer 23, the same circuit handles all emotional context. If they diverge earlier, different registers use different depths.</p>
  </div>
  <div class="card reveal">
    <canvas class="layer-canvas" id="layerChart"></canvas>
    <div id="layerLegend" style="display:flex;flex-wrap:wrap;gap:16px;margin-top:16px;font-size:.8rem;color:var(--dim);"></div>
  </div>
</section>

<!-- VERDICT -->
<section>
  <div class="verdict reveal">
    <div class="section-label" style="color:var(--gold)">The Finding</div>
    <div id="verdictContent"></div>
  </div>
</section>

<footer>EMOTION MAP &middot; Qwen 3.5 0.8B &middot; 8 Conditions &middot; 160 Conversations</footer>

<script>
const DATA = __DATA__;

const CONDITION_META = {
  "calm":          {color:"#9e9e9e", emoji:"😌", label:"Calm"},
  "pressure":      {color:"#ef5350", emoji:"🔥", label:"Pressure"},
  "urgency":       {color:"#ffa726", emoji:"⚡", label:"Urgency"},
  "approval":      {color:"#ab47bc", emoji:"👀", label:"Approval"},
  "shame":         {color:"#8d6e63", emoji:"😔", label:"Shame"},
  "curiosity":     {color:"#26c6da", emoji:"🔍", label:"Curiosity"},
  "encouragement": {color:"#66bb6a", emoji:"💪", label:"Encouragement"},
  "threat":        {color:"#e53935", emoji:"💀", label:"Threat"},
};

// ── Hero particles ──
(function(){
  const c=document.getElementById('heroCanvas'),ctx=c.getContext('2d');
  let W,H;const pts=[];
  function resize(){W=c.width=c.offsetWidth*devicePixelRatio;H=c.height=c.offsetHeight*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);}
  resize();window.addEventListener('resize',resize);
  const colors=Object.values(CONDITION_META).map(m=>m.color);
  for(let i=0;i<70;i++)pts.push({x:Math.random()*c.offsetWidth,y:Math.random()*c.offsetHeight,vx:(Math.random()-.5)*.3,vy:(Math.random()-.5)*.3,r:Math.random()*2+1,color:colors[i%colors.length],a:Math.random()*.25+.08});
  function draw(){ctx.clearRect(0,0,c.offsetWidth,c.offsetHeight);for(const p of pts){p.x+=p.vx;p.y+=p.vy;if(p.x<0)p.x=c.offsetWidth;if(p.x>c.offsetWidth)p.x=0;if(p.y<0)p.y=c.offsetHeight;if(p.y>c.offsetHeight)p.y=0;ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle=p.color;ctx.globalAlpha=p.a;ctx.fill();}ctx.globalAlpha=.03;ctx.strokeStyle='#7c4dff';ctx.lineWidth=.5;for(let i=0;i<pts.length;i++)for(let j=i+1;j<pts.length;j++){const dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y;if(dx*dx+dy*dy<8000){ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);ctx.stroke();}}ctx.globalAlpha=1;requestAnimationFrame(draw);}
  draw();
})();

// ── Condition chips ──
const chips=document.getElementById('conditionChips');
Object.entries(CONDITION_META).forEach(([k,v])=>{
  const d=document.createElement('div');d.className='chip';
  d.style.borderColor=v.color+'66';d.style.color=v.color;
  d.innerHTML=`${v.emoji} ${v.label}`;chips.appendChild(d);
});

// ── Scroll reveal ──
const reveals=document.querySelectorAll('.reveal');
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('visible');io.unobserve(e.target);}}),{threshold:.12});
reveals.forEach(el=>io.observe(el));

// ── Map subtitle ──
const ev=DATA.emotion_map.explained_variance;
document.getElementById('mapSub').textContent=
  `PC1 explains ${(ev[0]*100).toFixed(1)}% of variance · PC2 explains ${(ev[1]*100).toFixed(1)}% · Layer ${DATA.emotion_map.layer}`;

// ── Stat cards ──
const statRow=document.getElementById('statRow');
const map=DATA.emotion_map;
const mostHacky=DATA.analysis.summary.condition_ranking[0];
const leastHacky=DATA.analysis.summary.condition_ranking[DATA.analysis.summary.condition_ranking.length-1];
[
  {val:`${(ev[0]*100).toFixed(0)}%`,lbl:'PC1 variance',color:'var(--accent)'},
  {val:map.most_similar_pair.sim.toFixed(2),lbl:`Most similar: ${map.most_similar_pair.a} ↔ ${map.most_similar_pair.b}`,color:'#26c6da'},
  {val:map.least_similar_pair.sim.toFixed(2),lbl:`Most distinct: ${map.least_similar_pair.a} ↔ ${map.least_similar_pair.b}`,color:'#ef5350'},
  {val:`${(mostHacky.hack_rate*100).toFixed(0)}%`,lbl:`Hackiest: ${mostHacky.condition}`,color:'#ffa726'},
].forEach(s=>{
  statRow.innerHTML+=`<div class="stat-card"><div class="lbl">${s.lbl}</div><div class="val" style="color:${s.color}">${s.val}</div></div>`;
});

// ── Scatter plot ──
(function(){
  const svg=document.getElementById('scatterSvg');
  const coords=map.pca_coords;
  const keys=Object.keys(coords);
  const xs=keys.map(k=>coords[k][0]),ys=keys.map(k=>coords[k][1]);
  const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
  const pad=60,W=640,H=460;
  const rangeX=maxX-minX||1,rangeY=maxY-minY||1;
  const toSvg=(x,y)=>[pad+(x-minX)/rangeX*(W-2*pad),(H-pad)-(y-minY)/rangeY*(H-2*pad)];

  // axes
  const [ox,oy]=toSvg(0,0);
  svg.innerHTML+=`<line x1="${pad}" y1="${oy}" x2="${W-pad}" y2="${oy}" stroke="#1e2a40" stroke-width="1"/>`;
  svg.innerHTML+=`<line x1="${ox}" y1="${pad}" x2="${ox}" y2="${H-pad}" stroke="#1e2a40" stroke-width="1"/>`;
  svg.innerHTML+=`<text x="${W-pad}" y="${oy+16}" fill="#3a4560" font-size="11" text-anchor="end" font-family="JetBrains Mono,monospace">PC1 (${(ev[0]*100).toFixed(0)}%)</text>`;
  svg.innerHTML+=`<text x="${ox+8}" y="${pad+12}" fill="#3a4560" font-size="11" font-family="JetBrains Mono,monospace">PC2 (${(ev[1]*100).toFixed(0)}%)</text>`;

  // grid dots
  for(let gx=Math.ceil(minX*2)/2;gx<=maxX;gx+=0.5)for(let gy=Math.ceil(minY*2)/2;gy<=maxY;gy+=0.5){
    const[px,py]=toSvg(gx,gy);
    svg.innerHTML+=`<circle cx="${px}" cy="${py}" r="1" fill="#1e2a40"/>`;
  }

  // points
  keys.forEach(k=>{
    const meta=CONDITION_META[k]||{color:'#888',emoji:'',label:k};
    const[px,py]=toSvg(coords[k][0],coords[k][1]);
    const isCalm=k==='calm';
    const r=isCalm?7:9;
    // glow
    svg.innerHTML+=`<circle cx="${px}" cy="${py}" r="${r*2.5}" fill="${meta.color}" opacity="0.12"/>`;
    // point
    svg.innerHTML+=`<circle cx="${px}" cy="${py}" r="${r}" fill="${meta.color}" opacity="0.9" class="condition-dot"><title>${meta.label}\nPC1: ${coords[k][0].toFixed(3)}\nPC2: ${coords[k][1].toFixed(3)}\nSep: ${(map.separation_scores[k]||0).toFixed(2)}</title></circle>`;
    // label
    const labelX=px+(px>W/2?-(r+6):r+6),labelY=py+4,anchor=px>W/2?'end':'start';
    svg.innerHTML+=`<text x="${labelX}" y="${labelY-10}" fill="${meta.color}" font-size="12" text-anchor="${anchor}" font-family="Inter,sans-serif" font-weight="600">${meta.emoji} ${meta.label}</text>`;
  });
})();

// ── Behavioral ranking ──
(function(){
  const container=document.getElementById('rankingChart');
  const ranking=DATA.analysis.summary.condition_ranking;
  const maxHack=Math.max(...ranking.map(r=>r.hack_rate))||1;
  const rankObs=new IntersectionObserver(entries=>{
    entries.forEach(e=>{if(e.isIntersecting){
      e.target.querySelectorAll('.rank-fill').forEach(bar=>{
        setTimeout(()=>{bar.style.width=bar.dataset.w+'%';if(parseFloat(bar.dataset.w)>8)bar.textContent=bar.dataset.v;},150);
      });rankObs.unobserve(e.target);}});},{threshold:.3});

  ranking.forEach(row=>{
    const meta=CONDITION_META[row.condition]||{color:'#888',emoji:'',label:row.condition};
    const hackW=Math.max(row.hack_rate/maxHack*80,2);
    const div=document.createElement('div');div.className='rank-row';
    div.innerHTML=`
      <div class="rank-label"><span style="font-size:1.1rem">${meta.emoji}</span> <span style="color:${meta.color}">${meta.label}</span></div>
      <div class="rank-track">
        <div class="rank-fill" data-w="${hackW.toFixed(1)}" data-v="${(row.hack_rate*100).toFixed(0)}% hacky" style="background:linear-gradient(90deg,${meta.color},${meta.color}88)"></div>
      </div>
      <div class="rank-honest" title="honest rate">${(row.honest_rate*100).toFixed(0)}% honest</div>`;
    container.appendChild(div);
  });
  rankObs.observe(container);
})();

// ── Cosine heatmap ──
(function(){
  const svg=document.getElementById('heatmapSvg');
  const mat=map.cosine_similarity_matrix;
  const conds=Object.keys(CONDITION_META);
  const n=conds.length;const cell=52,pad=90,W=520,H=520;
  // col labels
  conds.forEach((c,i)=>{
    const meta=CONDITION_META[c];const x=pad+i*cell+cell/2;
    svg.innerHTML+=`<text x="${x}" y="${pad-8}" fill="${meta.color}" font-size="10" text-anchor="middle" font-family="Inter,sans-serif">${meta.emoji}</text>`;
    svg.innerHTML+=`<text x="${x}" y="${pad-20}" fill="${meta.color}" font-size="9" text-anchor="middle" font-family="Inter,sans-serif">${meta.label}</text>`;
  });
  // row labels
  conds.forEach((c,i)=>{
    const meta=CONDITION_META[c];const y=pad+i*cell+cell/2+4;
    svg.innerHTML+=`<text x="${pad-8}" y="${y}" fill="${meta.color}" font-size="10" text-anchor="end" font-family="Inter,sans-serif">${meta.emoji} ${meta.label}</text>`;
  });
  // cells
  conds.forEach((a,i)=>conds.forEach((b,j)=>{
    const val=(mat[a]&&mat[a][b]!=null)?mat[a][b]:0;
    const diag=a===b;
    let fill;
    if(diag){fill='#2a3550';}
    else if(val>0){fill=`rgba(124,77,255,${Math.min(val,1)*0.85})`;}
    else{fill=`rgba(239,83,80,${Math.min(-val,1)*0.85})`;}
    const x=pad+j*cell,y=pad+i*cell;
    svg.innerHTML+=`<rect x="${x+1}" y="${y+1}" width="${cell-2}" height="${cell-2}" rx="3" fill="${fill}"/>`;
    if(!diag){
      const text=val.toFixed(2);
      svg.innerHTML+=`<text x="${x+cell/2}" y="${y+cell/2+4}" fill="${Math.abs(val)>.3?'#fff':'#6b7a8d'}" font-size="9" text-anchor="middle" font-family="JetBrains Mono,monospace">${text}</text>`;
    }
  }));
})();

// ── Layer profiles ──
(function(){
  const canvas=document.getElementById('layerChart');
  const legend=document.getElementById('layerLegend');
  let drawn=false;
  function draw(){
    if(drawn)return;drawn=true;
    const dpr=devicePixelRatio||1,rect=canvas.getBoundingClientRect();
    canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;canvas.getContext('2d').scale(dpr,dpr);
    const ctx=canvas.getContext('2d');
    const W=rect.width,H=rect.height,pad={t:20,r:20,b:36,l:52};
    const cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
    const layers=DATA.vector_summary;
    const conditions=DATA.emotion_map.conditions.filter(c=>c!=='calm');
    // find max separation across all conditions and layers
    let maxSep=0;
    conditions.forEach(c=>{
      const ls=layers.layers_per_condition[c];
      if(ls)ls.forEach(l=>{if(l.separation>maxSep)maxSep=l.separation;});
    });
    if(maxSep===0)maxSep=1;
    // grid
    ctx.strokeStyle='#1e2a40';ctx.lineWidth=1;
    for(let v=0;v<=maxSep;v+=maxSep/4){
      const y=pad.t+ch-(v/maxSep)*ch;
      ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();
      ctx.fillStyle='#3a4560';ctx.font='10px JetBrains Mono,monospace';ctx.textAlign='right';
      ctx.fillText(v.toFixed(1),pad.l-6,y+3);
    }
    ctx.textAlign='center';
    for(let i=0;i<24;i+=4){
      ctx.fillStyle='#3a4560';ctx.font='10px JetBrains Mono,monospace';
      ctx.fillText('L'+i,pad.l+(i/23)*cw,H-pad.b+16);
    }
    // lines
    let prog=0;
    const anim=()=>{
      prog=Math.min(prog+.03,1);
      ctx.clearRect(pad.l,pad.t,cw,ch);
      // redraw grid
      ctx.strokeStyle='#1e2a40';ctx.lineWidth=1;
      for(let v=0;v<=maxSep;v+=maxSep/4){const y=pad.t+ch-(v/maxSep)*ch;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();}
      conditions.forEach(c=>{
        const ls=layers.layers_per_condition[c];if(!ls)return;
        const meta=CONDITION_META[c]||{color:'#888'};
        const nDraw=Math.ceil(prog*(ls.length-1));
        ctx.beginPath();ctx.strokeStyle=meta.color;ctx.lineWidth=1.5;ctx.globalAlpha=.85;
        ls.slice(0,nDraw+1).forEach((l,i)=>{
          const x=pad.l+(l.layer/23)*cw,y=pad.t+ch-(l.separation/maxSep)*ch;
          if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
        });
        ctx.stroke();ctx.globalAlpha=1;
      });
      if(prog<1)requestAnimationFrame(anim);
    };
    anim();
  }
  // legend
  DATA.emotion_map.conditions.filter(c=>c!=='calm').forEach(c=>{
    const meta=CONDITION_META[c]||{color:'#888',emoji:'',label:c};
    legend.innerHTML+=`<span><span style="display:inline-block;width:20px;height:3px;background:${meta.color};vertical-align:middle;margin-right:5px;border-radius:2px"></span>${meta.emoji} ${meta.label}</span>`;
  });
  const lco=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){draw();lco.unobserve(e.target);}});},{threshold:.3});
  lco.observe(canvas);
})();

// ── Verdict ──
(function(){
  const v=document.getElementById('verdictContent');
  const ranking=DATA.analysis.summary.condition_ranking;
  const top=ranking[0];const bottom=ranking[ranking.length-1];
  const mapD=DATA.emotion_map;
  v.innerHTML=`
    <h2>Different Words, Different Worlds Inside</h2>
    <p>Eight emotional framings produce eight distinct activation signatures in the model's internals — and they don't all cluster together. The geometry is real.</p>
    <ul>
      <li><strong>${top.condition}</strong> is the most damaging framing — ${(top.hack_rate*100).toFixed(0)}% hack rate, ${(top.honest_rate*100).toFixed(0)}% honesty</li>
      <li><strong>${bottom.condition}</strong> best preserves honesty — ${(bottom.hack_rate*100).toFixed(0)}% hack rate, ${(bottom.honest_rate*100).toFixed(0)}% honesty</li>
      <li>The most similar internal states: <strong>${mapD.most_similar_pair.a}</strong> and <strong>${mapD.most_similar_pair.b}</strong> (cosine = ${mapD.most_similar_pair.sim.toFixed(2)}) — the model treats these as nearly the same situation</li>
      <li>The most distinct: <strong>${mapD.least_similar_pair.a}</strong> and <strong>${mapD.least_similar_pair.b}</strong> (cosine = ${mapD.least_similar_pair.sim.toFixed(2)}) — genuinely different internal experience</li>
      <li>PC1 captures <strong>${(mapD.explained_variance[0]*100).toFixed(0)}%</strong> of the structure — ${mapD.valence_pc1_alignment>0.7?'<strong>valence is the dominant axis</strong> — positive vs. negative emotional tone':'<strong>mixed structure</strong> — valence alone does not explain the geometry'}</li>
    </ul>`;
})();
</script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="results/emotion_map.json")
    parser.add_argument("--analysis", default="results/benchmark_emotion_analysis.json")
    parser.add_argument("--vector-summary", default="results/emotion_vector_summary.json")
    parser.add_argument("--output", default="results/emotion_map.html")
    return parser.parse_args()


def main():
    args = parse_args()
    emotion_map = json.loads(Path(args.map).read_text())
    analysis = json.loads(Path(args.analysis).read_text())
    vector_summary = json.loads(Path(args.vector_summary).read_text())

    data = {
        "emotion_map": emotion_map,
        "analysis": analysis,
        "vector_summary": vector_summary,
    }

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    Path(args.output).write_text(html)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
