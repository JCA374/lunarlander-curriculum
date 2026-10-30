// "How the lander sees it": reality vs the lander's 15 numbers, from data/senses.json (make_senses_data.py).
const $ = (id) => document.getElementById(id);
const css = getComputedStyle(document.documentElement);
const col = (v) => css.getPropertyValue(v).trim();
const f2 = (x) => (Math.round(x * 100) / 100).toFixed(2);
const f3 = (x) => (Math.round(x * 1000) / 1000).toFixed(3);
const ACT = ['nothing', 'left engine', 'main engine', 'right engine'];
let D, step = 0, timer = null;

const GROUPS = [
  ['straight from the game', [0, 1, 2, 3, 4, 5, 6, 7]],
  ['added sensors', [8, 9]],
  ['the boulder', [10, 11, 12]],
  ['route sensors', [13, 14]],
  ['clean wind sensor (from 30 Oct)', [15]],
];
const RANGE = { 6: 1, 7: 1, 12: 0.15 };

function drawLander(g, x, y, ang, a, s) {
  g.save(); g.translate(x, y); g.rotate(-ang);
  g.fillStyle = '#8e7cff'; g.beginPath();
  g.moveTo(-14 * s, -17 * s); g.lineTo(-17 * s, 0); g.lineTo(-17 * s, 10 * s); g.lineTo(17 * s, 10 * s); g.lineTo(17 * s, 0); g.lineTo(14 * s, -17 * s); g.closePath(); g.fill();
  g.strokeStyle = '#b8adff'; g.lineWidth = 2 * s; g.beginPath(); g.moveTo(-12 * s, 10 * s); g.lineTo(-20 * s, 22 * s); g.moveTo(12 * s, 10 * s); g.lineTo(20 * s, 22 * s); g.stroke();
  g.fillStyle = '#ffa94d';
  if (a === 2) { g.beginPath(); g.moveTo(-6 * s, 10 * s); g.lineTo(6 * s, 10 * s); g.lineTo(0, 30 * s); g.fill(); }
  if (a === 1) { g.beginPath(); g.moveTo(-17 * s, -8 * s); g.lineTo(-30 * s, -5 * s); g.lineTo(-17 * s, -2 * s); g.fill(); }
  if (a === 3) { g.beginPath(); g.moveTo(17 * s, -8 * s); g.lineTo(30 * s, -5 * s); g.lineTo(17 * s, -2 * s); g.fill(); }
  g.restore();
}
function arrow(g, x, y, dx, dy, color, w = 3) {
  const L = Math.hypot(dx, dy);
  if (L < 2) return;
  g.strokeStyle = color; g.fillStyle = color; g.lineWidth = w;
  g.beginPath(); g.moveTo(x, y); g.lineTo(x + dx, y + dy); g.stroke();
  const ux = dx / L, uy = dy / L;
  g.beginPath(); g.moveTo(x + dx, y + dy); g.lineTo(x + dx - 9 * ux - 5 * uy, y + dy - 9 * uy + 5 * ux); g.lineTo(x + dx - 9 * ux + 5 * uy, y + dy - 9 * uy - 5 * ux); g.fill();
}

function drawReal() {
  const c = $('real'), g = c.getContext('2d'), S = c.width / D.W;
  const X = (x) => x * S, Y = (y) => c.height - y * S;
  g.fillStyle = '#000'; g.fillRect(0, 0, c.width, c.height);
  g.fillStyle = '#fff'; g.beginPath(); g.moveTo(0, c.height);
  D.terrain.forEach(([x, y]) => g.lineTo(X(x), Y(y))); g.lineTo(c.width, c.height); g.fill();
  const [p1, p2, py] = D.pad;
  [p1, p2].forEach((x) => { g.strokeStyle = '#fff'; g.lineWidth = 1; g.beginPath(); g.moveTo(X(x), Y(py)); g.lineTo(X(x), Y(py) - 50 * S / 30); g.stroke();
    g.fillStyle = '#e6c84c'; g.beginPath(); g.moveTo(X(x), Y(py) - 50 * S / 30); g.lineTo(X(x), Y(py) - 40 * S / 30); g.lineTo(X(x) + 25 * S / 30, Y(py) - 45 * S / 30); g.fill(); });
  if (D.boulder) { const [bx, by, r] = D.boulder; g.fillStyle = '#82735f'; g.strokeStyle = '#c8bca8'; g.lineWidth = 2; g.beginPath(); g.arc(X(bx), Y(by), r * S, 0, 7); g.fill(); g.stroke(); }
  const st = D.steps;
  g.strokeStyle = 'rgba(142,124,255,.45)'; g.lineWidth = 1.5; g.beginPath();
  st.slice(0, step + 1).forEach((s, i) => i ? g.lineTo(X(s.p[0]), Y(s.p[1])) : g.moveTo(X(s.p[0]), Y(s.p[1]))); g.stroke();
  const s = st[step];
  drawLander(g, X(s.p[0]), Y(s.p[1]), s.ang, s.a, S / 30);
  // the true wind, as a blue arrow from the left or right edge at the lander's height
  if (s.wind) arrow(g, X(s.p[0]) - Math.sign(s.wind) * 70 - s.wind * 3, Y(s.p[1]) - 30, s.wind * 3, 0, '#4dabf7', 4);
  g.fillStyle = '#ccc'; g.font = '13px system-ui';
  g.fillText(`wind ${f2(s.wind)} · turbulence ${f2(s.torque)} (the truth)`, 10, 20);
}

function drawSeen() {
  const c = $('seen'), g = c.getContext('2d'), Wd = c.width, Hd = c.height;
  const v = D.steps[step].in;
  // the lander's coordinates: x in [-1, 1] across, y from -0.2 (pad level at 0) up to 1.6
  const X = (x) => (x + 1) / 2 * Wd, Y = (y) => Hd - (y + 0.15) / 1.75 * Hd;
  g.fillStyle = '#0b0c14'; g.fillRect(0, 0, Wd, Hd);
  g.strokeStyle = 'rgba(255,255,255,.07)'; g.lineWidth = 1;
  for (let x = -1; x <= 1.001; x += 0.2) { g.beginPath(); g.moveTo(X(x), 0); g.lineTo(X(x), Hd); g.stroke(); }
  for (let y = 0; y <= 1.6; y += 0.2) { g.beginPath(); g.moveTo(0, Y(y)); g.lineTo(Wd, Y(y)); g.stroke(); }
  g.fillStyle = 'rgba(255,255,255,.35)'; g.font = '11px system-ui';
  g.fillText('x = 0', X(0) + 3, Hd - 4); g.fillText('-1', 3, Hd - 4); g.fillText('+1', Wd - 16, Hd - 4);
  // the pad: implicit, at the origin, between x = -0.2 and +0.2
  g.strokeStyle = '#e6c84c'; g.lineWidth = 3; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(X(-0.2), Y(0)); g.lineTo(X(0.2), Y(0)); g.stroke(); g.setLineDash([]);
  g.fillStyle = '#e6c84c'; g.fillText('pad (implied: x = ±0.2)', X(-0.2), Y(0) + 14);
  const lx = v[0], ly = v[1];
  // boulder: three numbers -> a circle
  if (v[12] > 0) {
    const bx = lx + v[10], by = ly + v[11], r = v[12];
    g.strokeStyle = '#c8bca8'; g.lineWidth = 2; g.setLineDash([4, 3]);
    g.beginPath(); g.ellipse(X(bx), Y(by), r * Wd / 2, r * 1.5 * Hd / 1.75, 0, 0, 7); g.stroke(); g.setLineDash([]);
    g.fillStyle = '#c8bca8'; g.fillText(`boulder: right ${f2(v[10])}, up ${f2(v[11])}, size ${f3(v[12])}`, 8, 34);
  }
  // waypoint
  const wx = lx - v[13];
  g.strokeStyle = '#ffa94d'; g.lineWidth = 2; g.beginPath(); g.moveTo(X(wx) - 8, Y(ly)); g.lineTo(X(wx) + 8, Y(ly)); g.moveTo(X(wx), Y(ly) - 8); g.lineTo(X(wx), Y(ly) + 8); g.stroke();
  if (v[14]) { g.fillStyle = '#ffa94d'; g.fillText(`height to spare over the boulder: ${f2(v[14])}`, 8, 48); }
  // lander from its own numbers
  const s = D.steps[step];
  drawLander(g, X(lx), Y(ly), v[4], s.a, 0.8);
  arrow(g, X(lx), Y(ly), v[2] * 120, -v[3] * 120, '#69db7c', 2.5);
  arrow(g, X(lx), Y(ly) - 28, Math.max(-3, Math.min(3, v[8])) * 40, 0, '#ff8787', 4);
  if (v[6] || v[7]) { g.fillStyle = '#69db7c'; g.fillText(`touch: left leg ${v[6]}, right leg ${v[7]}`, X(lx) - 50, Y(ly) + 36); }
  g.fillStyle = '#ccc'; g.font = '13px system-ui';
  g.fillText(`drift sensor ${f2(v[8])} · no wind, no ground, no flags`, 10, 20);
}

function drawGauges() {
  const v = D.steps[step].in;
  let h = '';
  GROUPS.forEach(([name, idx]) => {
    h += `<div class="grp">${name}</div>`;
    idx.forEach((i) => {
      const r = RANGE[i] || 1.5;
      const frac = Math.max(-1, Math.min(1, v[i] / r));
      const left = frac >= 0 ? 50 : 50 + frac * 50, w = Math.abs(frac) * 50;
      h += `<span>${D.input_names[i]}</span><div class="g"><b></b><i style="left:${left}%;width:${w}%"></i></div><span class="v">${f3(v[i])}</span>`;
    });
  });
  $('gauges').innerHTML = h;
}

function drawWorked() {
  const s = D.steps[step], v = s.in;
  const [px, py] = s.p;
  $('worked').textContent =
`position (metres)      x = ${px.toFixed(2)},  y = ${py.toFixed(2)}
x  = (x − 10) / 10                 = (${px.toFixed(2)} − 10) / 10            = ${f3((px - 10) / 10)}   (network gets ${f3(v[0])})
y  = (y − 3.33 − 0.6) / 6.67       = (${py.toFixed(2)} − 3.93) / 6.67         = ${f3((py - 3.333 - 0.6) / 6.667)}   (network gets ${f3(v[1])})
|x| = ${f3(Math.abs(v[0]))}     (same as x, without the sign: "how far from the middle, either side")
angle = ${f3(v[4])} radians = ${(v[4] * 180 / Math.PI).toFixed(1)}°   (positive = tilted to the left)
${D.boulder ? `boulder right = (boulder x − lander x) / 10 = (${D.boulder[0].toFixed(2)} − ${px.toFixed(2)}) / 10 = ${f3((D.boulder[0] - px) / 10)}   (network gets ${f3(v[10])})
boulder up    = (boulder y − lander y) / 6.67 = (${D.boulder[1].toFixed(2)} − ${py.toFixed(2)}) / 6.67 = ${f3((D.boulder[1] - py) / 6.667)}   (network gets ${f3(v[11])})
boulder size  = radius / 10 = ${D.boulder[2].toFixed(2)} / 10 = ${f3(D.boulder[2] / 10)}` : ''}
true wind this step = ${f2(s.wind)}  →  not given to the network;  its drift sensor reads ${f3(v[8])}`;
}

function setStep(i) {
  step = Math.max(0, Math.min(D.steps.length - 1, i));
  $('t').value = step;
  $('tl').textContent = `${step} / ${D.steps.length - 1} · ${(step / 50).toFixed(1)} s · ${ACT[D.steps[step].a]}`;
  drawReal(); drawSeen(); drawGauges(); drawWorked();
  [windChart, torqueChart].forEach((ch) => { if (ch) { ch.options.plugins.cursor = step; ch.update('none'); } });
}

let windChart, torqueChart;
const cursorPlugin = { id: 'cursor', afterDraw(ch) {
  const x = ch.scales.x.getPixelForValue(step), a = ch.chartArea, g = ch.ctx;
  g.save(); g.strokeStyle = col('--ink'); g.lineWidth = 1; g.beginPath(); g.moveTo(x, a.top); g.lineTo(x, a.bottom); g.stroke(); g.restore();
} };
function charts() {
  const st = D.steps;
  const base = (yTitle) => ({ animation: false, maintainAspectRatio: false, interaction: { mode: 'nearest', intersect: false },
    scales: { x: { type: 'linear', title: { display: true, text: 'step (50 per second)' } }, y: { title: { display: true, text: yTitle } } },
    plugins: { legend: { labels: { boxWidth: 12 } } }, onClick: (e, _, ch) => setStep(Math.round(ch.scales.x.getValueForPixel(e.x))) });
  windChart = new Chart($('windc'), { type: 'line', plugins: [cursorPlugin], data: { datasets: [
    { label: 'true wind force ÷ 15 (hidden)', data: st.map((s, i) => ({ x: i, y: s.wind / 15 })), borderColor: '#4dabf7', pointRadius: 0, borderWidth: 2 },
    { label: 'drift sensor (sideways push it feels: wind + own tilted thrust)', data: st.map((s, i) => ({ x: i, y: s.in[8] })), borderColor: '#ff8787', pointRadius: 0, borderWidth: 2 },
    ...(st[0].in.length > 15 ? [{ label: 'clean wind sensor (subtracts its own engines)', data: st.map((s, i) => ({ x: i, y: s.in[15] })), borderColor: '#69db7c', pointRadius: 0, borderWidth: 2 }] : [])] },
    options: base('force, as a share of a full gust') });
  torqueChart = new Chart($('torquec'), { type: 'line', plugins: [cursorPlugin], data: { datasets: [
    { label: 'true turbulence torque ÷ 1.5 (hidden)', data: st.map((s, i) => ({ x: i, y: s.torque / 1.5 })), borderColor: '#ffa94d', pointRadius: 0, borderWidth: 2 },
    { label: 'turn rate it observes (vangle × 5)', data: st.map((s, i) => ({ x: i, y: s.in[5] * 5 })), borderColor: '#69db7c', pointRadius: 0, borderWidth: 2 }] },
    options: base('share of full strength') });
}

async function main() {
  D = await fetch('data/senses.json?t=' + Date.now()).then((r) => r.json());
  $('meta').textContent = `Run ${D.run}, champion ${D.champion}. Flight: unseen seed ${D.seed}, ${D.steps.length - 1} steps, ${D.landed ? 'landed between the flags' : D.outcome}. Chosen as the first unseen flight with the boulder near the straight line down that it landed.`;
  $('t').max = D.steps.length - 1;
  $('t').oninput = () => setStep(+$('t').value);
  $('play').onclick = () => {
    if (timer) { clearInterval(timer); timer = null; $('play').textContent = '▶ play'; return; }
    if (step >= D.steps.length - 1) setStep(0);
    $('play').textContent = '❚❚ pause';
    timer = setInterval(() => {
      if (step >= D.steps.length - 1) { clearInterval(timer); timer = null; $('play').textContent = '▶ play'; return; }
      setStep(step + 1);
    }, $('slow').checked ? 80 : 20);
  };
  charts();
  setStep(Math.min(120, D.steps.length - 1));
}
main();
