// The Network tab: everything computed in the browser from data/network.json (make_network_data.py).
const $ = (id) => document.getElementById(id);
const css = getComputedStyle(document.documentElement);
const col = (v) => css.getPropertyValue(v).trim();
const f3 = (x) => (Math.round(x * 1000) / 1000).toFixed(3);
const f2 = (x) => (Math.round(x * 100) / 100).toFixed(2);
const pct = (x) => x == null ? '–' : (100 * x).toFixed(0) + '%';
const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

const INPUT_INFO = {
  x: ['horizontal position', '0 = pad centre, ±0.2 = the flags, ±1 = screen edge'],
  y: ['height above the pad', '0 = pad level, ~1.4 at the start'],
  vx: ['sideways speed', 'positive = moving right'],
  vy: ['vertical speed', 'negative = falling'],
  angle: ['tilt', 'radians; positive = tilted left (main engine then pushes left)'],
  vangle: ['turn rate', 'how fast the tilt is changing'],
  leg_l: ['left leg touching', '1 or 0'],
  leg_r: ['right leg touching', '1 or 0'],
  drift: ['drift sensor', 'smoothed sideways acceleration, side-engine steps left out: wind + its own tilted main engine; +1 ≈ a full-strength gust to the right'],
  '|x|': ['distance from the middle', 'the same as x but without the sign: "how far off", either side'],
  boulder_dx: ['boulder, how far right', 'boulder centre minus lander, in x units; 0 if no boulder'],
  boulder_dy: ['boulder, how far up', 'negative = the boulder is below the lander'],
  boulder_r: ['boulder size', 'radius ÷ 10 m: 0.08-0.13 at full size; 0 = no boulder'],
  waypoint_dx: ['route: off the waypoint by', 'x minus where it should head (beside the boulder, else the pad)'],
  above_boulder: ['route: height to spare', 'height above the boulder top while over it; 0 elsewhere'],
  wind_est: ['clean wind sensor (new 30 Oct)', 'sideways speed change minus each action’s known push; +1 ≈ a full gust to the right (r = 0.97 with the true wind)'],
};

let D, NET, mode = 'newest', step = 0, thr = 0.6, hoverNode = null, cur = null;

// ---------- the forward pass, exactly as neat-python does it ----------
function prepare(net, nIn) {
  const into = {};
  net.conns.forEach((c) => (into[c.o] = into[c.o] || []).push(c));
  const order = [...net.nodes].sort((a, b) => a.d - b.d);
  return { net, into, order, nIn };
}
function act(a, z) { return a === 'identity' ? z : Math.tanh(Math.max(-60, Math.min(60, 2.5 * z))); }
function forward(P, inputs) {
  const v = {};
  for (let i = 0; i < P.nIn; i++) v[-(i + 1)] = inputs[i];
  for (const n of P.order) {
    let s = 0;
    for (const c of P.into[n.k] || []) s += c.w * (v[c.i] ?? 0);
    v[n.k] = act(n.a, n.b + n.r * s);
  }
  return v;
}

function nodeName(k) {
  if (k < 0) return D.input_names[-k - 1];
  if (k < 4) return D.output_names[k];
  return 'H' + k;
}

// ---------- 1. the diagram ----------
function layout(net) {
  const nIn = D.input_names.length;
  const hidden = net.nodes.filter((n) => n.k >= 4);
  const maxD = Math.max(2, ...net.nodes.map((n) => n.d));
  const pos = {};
  const W = 900, H = 620, x0 = 110, x1 = 790;
  for (let i = 0; i < nIn; i++) pos[-(i + 1)] = [x0, 30 + i * (H - 60) / (nIn - 1)];
  for (let o = 0; o < 4; o++) pos[o] = [x1, 150 + o * 105];
  const byDepth = {};
  hidden.forEach((n) => (byDepth[Math.min(n.d, maxD - 1)] = byDepth[Math.min(n.d, maxD - 1)] || []).push(n));
  const cols = Object.keys(byDepth).map(Number).sort((a, b) => a - b);
  cols.forEach((d, ci) => {
    const list = byDepth[d].sort((a, b) => a.k - b.k);
    const x = x0 + (x1 - x0) * (ci + 1) / (cols.length + 1);
    list.forEach((n, i) => (pos[n.k] = [x, list.length === 1 ? H / 2 : 18 + i * (H - 36) / (list.length - 1)]));
  });
  return pos;
}

function drawNet() {
  const svg = $('netsvg');
  const newest = D.champions.length ? D.champions[D.champions.length - 1] : null;
  const net = mode === 'seed' || !newest ? D.seed : newest.net;
  const seedW = {};
  D.seed.conns.forEach((c) => (seedW[c.i + ',' + c.o] = c.w));
  const pos = layout(net);
  const vals = cur ? cur.vals : {};
  const parts = [];
  const sel = hoverNode;
  const conns = net.conns.slice().sort((a, b) => Math.abs(a.w) - Math.abs(b.w));
  let shown = 0;
  for (const c of conns) {
    const p = pos[c.i], q = pos[c.o];
    if (!p || !q) continue;
    let w = c.w, color, dash = '';
    if (mode === 'diff') {
      const s = seedW[c.i + ',' + c.o];
      if (s === undefined) { color = col('--good'); dash = 'stroke-dasharray="5 3"'; w = Math.max(Math.abs(c.w), 0.8); }
      else { const dw = Math.abs(c.w) - Math.abs(s); w = dw * 8; color = dw >= 0 ? col('--good') : col('--warn'); }
    } else color = c.w >= 0 ? col('--accent') : col('--bad');
    const strong = Math.abs(w) >= thr;
    const touches = sel != null && (c.i === sel || c.o === sel);
    if (!strong && !touches) continue;
    if (sel != null && !touches) continue;
    shown++;
    const op = Math.min(0.85, 0.12 + Math.abs(w) / 6);
    parts.push(`<line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="${color}" stroke-width="${Math.min(5, 0.4 + Math.abs(w) * 0.6)}" stroke-opacity="${touches ? 0.95 : op}" ${dash}/>`);
  }
  const newNodes = new Set(newest && mode !== 'seed' ? newest.diff.new_node_keys : []);
  for (const [k, [x, y]] of Object.entries(pos)) {
    const key = +k;
    const v = vals[key];
    let fill = col('--code');
    if (v !== undefined && key >= 4) fill = v >= 0 ? `rgba(232,89,12,${Math.min(1, Math.abs(v))})` : `rgba(59,91,219,${Math.min(1, Math.abs(v))})`;
    if (key >= 0 && key < 4 && cur && key === cur.action) fill = col('--good');
    const r = key < 0 ? 7 : key < 4 ? 13 : 8;
    const ring = newNodes.has(key) ? `stroke="${col('--good')}" stroke-width="3"` : `stroke="${col('--muted')}" stroke-width="1"`;
    parts.push(`<circle data-k="${key}" cx="${x}" cy="${y}" r="${r}" fill="${fill}" ${ring} style="cursor:pointer"/>`);
    if (key < 0) parts.push(`<text x="${x - 12}" y="${y + 4}" text-anchor="end">${esc(nodeName(key))}${v !== undefined ? ' <tspan class="lbl-muted">' + f2(v) + '</tspan>' : ''}</text>`);
    else if (key < 4) parts.push(`<text x="${x + 18}" y="${y + 4}">${esc(nodeName(key))}${v !== undefined ? ' <tspan class="lbl-muted">' + f2(v) + '</tspan>' : ''}</text>`);
  }
  if (sel != null) {
    const n = net.nodes.find((n) => n.k === sel);
    const [x, y] = pos[sel] || [450, 20];
    if (n) parts.push(`<text x="${Math.min(x + 12, 700)}" y="${Math.max(14, y - 12)}" font-weight="600">${esc(nodeName(sel))}: bias ${f2(n.b)}, ${n.a}${vals[sel] !== undefined ? ', value ' + f3(vals[sel]) : ''}</text>`);
  }
  svg.innerHTML = parts.join('');
  svg.querySelectorAll('circle').forEach((c) => {
    c.addEventListener('mouseenter', () => { hoverNode = +c.dataset.k; drawNet(); });
    c.addEventListener('click', () => { hoverNode = hoverNode === +c.dataset.k ? null : +c.dataset.k; drawNet(); });
  });
  svg.onmouseleave = () => { if (hoverNode != null) { hoverNode = null; drawNet(); } };
  const hidden = net.nodes.filter((n) => n.k >= 4).length;
  const what = mode === 'seed' ? 'Start network (the exact copy of the pilot)' : mode === 'diff' ? 'Change since the start' : `Newest champion (${newest ? newest.file : ''})`;
  $('netcap').textContent = `${what}: ${D.input_names.length} inputs, ${hidden} hidden neurons, 4 outputs, ${net.conns.length} connections (${shown} drawn).` +
    (cur ? ` Neuron colours: values at step ${step} of the flight in section 6 (orange positive, blue negative).` : '');
}

// ---------- 2. inputs table ----------
function drawInputs() {
  const v = cur ? cur.inputs : [];
  $('intable').innerHTML = '<tr><th>#</th><th>name</th><th>what it means</th><th>units / range</th><th>at this step</th></tr>' +
    D.input_names.map((n, i) => `<tr><td>${i + 1}</td><td><b>${esc(n)}</b></td><td style="text-align:left">${esc((INPUT_INFO[n] || [''])[0])}</td><td style="text-align:left" class="muted">${esc((INPUT_INFO[n] || ['', ''])[1])}</td><td>${v.length ? f3(v[i]) : ''}</td></tr>`).join('');
}

// ---------- 3. one neuron by hand ----------
function busiestNeuron(net) {
  const score = {};
  net.conns.forEach((c) => { if (c.o < 4 && c.i >= 4) score[c.i] = (score[c.i] || 0) + Math.abs(c.w); });
  return +Object.entries(score).sort((a, b) => b[1] - a[1])[0][0];
}
function drawByHand() {
  if (!cur) return;
  const k = busiestNeuron(NET.net);
  const n = NET.net.nodes.find((x) => x.k === k);
  const ins = (NET.into[k] || []).slice().sort((a, b) => Math.abs(b.w * (cur.vals[b.i] ?? 0)) - Math.abs(a.w * (cur.vals[a.i] ?? 0)));
  let sum = 0;
  const rows = ins.map((c) => { const x = cur.vals[c.i] ?? 0; sum += c.w * x; return `<tr><td>${esc(nodeName(c.i))}</td><td>${f3(x)}</td><td>× ${f3(c.w)}</td><td>= ${f3(c.w * x)}</td></tr>`; });
  const z = n.b + sum;
  $('byhand').innerHTML = `<b>Neuron ${nodeName(k)}</b> at step ${step}: its ${ins.length} incoming connections, biggest contributions first.
    <table class="tbl" style="margin-top:6px"><tr><th>from</th><th>value</th><th>weight</th><th>contribution</th></tr>${rows.join('')}
    <tr><td><b>sum</b></td><td></td><td></td><td><b>${f3(sum)}</b></td></tr>
    <tr><td>+ bias</td><td></td><td></td><td>${f3(n.b)}</td></tr>
    <tr><td><b>z</b></td><td></td><td></td><td><b>${f3(z)}</b></td></tr>
    <tr><td><b>output = tanh(2.5 × z)</b></td><td></td><td></td><td><b>${f3(act(n.a, z))}</b></td></tr></table>`;
  const outs = [0, 1, 2, 3].map((o) => {
    const on = NET.net.nodes.find((x) => x.k === o);
    const s = (NET.into[o] || []).reduce((a, c) => a + c.w * (cur.vals[c.i] ?? 0), 0);
    const fromK = (NET.into[o] || []).find((c) => c.i === k);
    return { o, s, b: on.b, z: on.b + s, fromK };
  });
  const best = outs.reduce((a, b) => (b.z > a.z ? b : a));
  $('byhand2').innerHTML = `<table class="tbl"><tr><th>output</th><th>sum of (hidden × weight)</th><th>+ bias</th><th>= score</th><th>of which from ${nodeName(k)}</th></tr>` +
    outs.map((r) => `<tr${r === best ? ' style="font-weight:700"' : ''}><td>${esc(D.output_names[r.o])}${r === best ? ' ← wins' : ''}</td><td>${f3(r.s)}</td><td>${f3(r.b)}</td><td>${f3(r.z)}</td><td>${r.fromK ? f3(r.fromK.w * cur.vals[k]) : '–'}</td></tr>`).join('') + '</table>';
}

// ---------- 4. tanh plot ----------
function drawTanh() {
  const c = $('tanhc'), g = c.getContext('2d'), W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  const X = (z) => 30 + (z + 2) / 4 * (W - 40), Y = (v) => H / 2 - v * (H / 2 - 20);
  g.strokeStyle = col('--line'); g.lineWidth = 1;
  g.beginPath(); g.moveTo(30, Y(0)); g.lineTo(W - 10, Y(0)); g.moveTo(X(0), 10); g.lineTo(X(0), H - 10); g.stroke();
  g.fillStyle = col('--muted'); g.font = '11px system-ui';
  [-1, 1].forEach((v) => g.fillText((v > 0 ? '+' : '') + v, X(0) + 4, Y(v) + 4));
  [-2, -1, 1, 2].forEach((z) => g.fillText(z, X(z) - 4, Y(0) + 14));
  const line = (f, color, w) => { g.strokeStyle = color; g.lineWidth = w; g.beginPath(); for (let i = 0; i <= 200; i++) { const z = -2 + i / 50; i ? g.lineTo(X(z), Y(f(z))) : g.moveTo(X(z), Y(f(z))); } g.stroke(); };
  line((z) => Math.tanh(z), col('--muted'), 1.5);
  line((z) => Math.tanh(2.5 * z), col('--accent'), 2.5);
  line((z) => Math.max(-1, Math.min(1, z * 0.5)), col('--good'), 1.5);
  g.fillStyle = col('--accent'); g.fillText('NEAT: tanh(2.5 z)', X(0.3), Y(0.75));
  g.fillStyle = col('--muted'); g.fillText('textbook tanh(z)', X(0.9), Y(0.45));
  g.fillStyle = col('--good'); g.fillText('identity (outputs now), clipped here', X(-1.95), Y(-0.85));
}

// ---------- 5. try it ----------
let tryInputs = [];
function buildTry() {
  $('presetbtns').innerHTML = D.presets.map((p, i) => `<button class="pill${i ? '' : ' on'}" data-p="${i}">${esc(p.label)}</button>`).join('');
  $('presetbtns').querySelectorAll('button').forEach((b) => b.onclick = () => {
    $('presetbtns').querySelectorAll('button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); setPreset(+b.dataset.p);
  });
  setPreset(0);
}
function setPreset(i) {
  const p = D.presets[i];
  tryInputs = p.in.slice();
  $('presetwhy').textContent = `${p.why} (step ${p.step} of the recorded flight)`;
  const ranges = { leg_l: [0, 1, 1], leg_r: [0, 1, 1] };
  $('sliders').innerHTML = D.input_names.map((n, j) => {
    const [lo, hi, st] = ranges[n] || [-1.5, 1.5, 0.01];
    return `<span>${esc(n)}</span><input type="range" min="${lo}" max="${hi}" step="${st}" value="${tryInputs[j]}" data-j="${j}"><span class="val" id="sv${j}">${f2(tryInputs[j])}</span>`;
  }).join('');
  $('sliders').querySelectorAll('input').forEach((s) => s.oninput = () => { tryInputs[+s.dataset.j] = +s.value; $('sv' + s.dataset.j).textContent = f2(+s.value); runTry(); });
  runTry();
}
function runTry() {
  const v = forward(NET, tryInputs);
  const outs = [0, 1, 2, 3].map((o) => v[o]);
  const best = outs.indexOf(Math.max(...outs));
  $('action').textContent = ['💤 nothing', '◀ left engine', '▲ main engine', '▶ right engine'][best];
  const lo = Math.min(...outs, 0), hi = Math.max(...outs, 0.001);
  $('outbars').innerHTML = outs.map((z, o) => {
    const l = (Math.min(0, z) - lo) / (hi - lo) * 100, w = Math.abs(z) / (hi - lo) * 100;
    return `<span>${esc(D.output_names[o])}</span><div class="bar${o === best ? ' win' : ''}"><i style="left:${l}%;width:${w}%"></i></div><span class="val">${f2(z)}</span>`;
  }).join('');
  const hid = NET.net.nodes.filter((n) => n.k >= 4).map((n) => v[n.k]);
  const c = $('hidbars'), g = c.getContext('2d');
  g.clearRect(0, 0, c.width, c.height);
  const bw = c.width / hid.length;
  hid.forEach((h, i) => { g.fillStyle = h >= 0 ? '#e8590c' : '#3b5bdb'; const y = c.height / 2; g.fillRect(i * bw + 1, h >= 0 ? y - h * y : y, bw - 2, Math.abs(h) * y); });
  g.strokeStyle = col('--line'); g.beginPath(); g.moveTo(0, c.height / 2); g.lineTo(c.width, c.height / 2); g.stroke();
}

// ---------- 6. the flight ----------
function setStep(i) {
  const s = D.trace.steps[i];
  step = i;
  const vals = forward(NET, s.in);
  cur = { inputs: s.in, vals, action: s.a };
  $('tslider').value = i;
  const lab = ['nothing', 'left engine', 'main engine', 'right engine'][s.a];
  $('tlabel').textContent = `${i} / ${D.trace.steps.length - 1} · ${(i / 50).toFixed(1)} s · action: ${lab}`;
  drawPath(); drawHeat(); drawNet(); drawInputs(); drawByHand();
}
function drawPath() {
  const c = $('pathc'), g = c.getContext('2d'), W = c.width, H = c.height;
  const sx = W / 20, sy = H / 13.33;
  const X = (x) => x * sx, Y = (y) => H - y * sy;
  g.fillStyle = '#0b0c10'; g.fillRect(0, 0, W, H);
  g.fillStyle = '#d9d9d9'; g.fillRect(0, Y(3.33), W, H - Y(3.33));
  g.fillStyle = '#e6c84c'; g.fillRect(X(10 - 2), Y(3.33) - 18, 2, 18); g.fillRect(X(10 + 2) - 2, Y(3.33) - 18, 2, 18);
  const b = D.trace.boulder;
  if (b) { g.fillStyle = '#82735f'; g.beginPath(); g.arc(X(b[0]), Y(b[1]), b[2] * sx, 0, 7); g.fill(); }
  const st = D.trace.steps;
  g.strokeStyle = 'rgba(142,166,255,.6)'; g.lineWidth = 1.5; g.beginPath();
  st.forEach((s, i) => i ? g.lineTo(X(s.p[0]), Y(s.p[1])) : g.moveTo(X(s.p[0]), Y(s.p[1]))); g.stroke();
  const s = st[step];
  g.save(); g.translate(X(s.p[0]), Y(s.p[1])); g.rotate(-s.ang);
  g.fillStyle = '#8e7cff'; g.fillRect(-10, -9, 20, 14);
  if (s.a === 2) { g.fillStyle = '#ffa94d'; g.beginPath(); g.moveTo(-5, 5); g.lineTo(5, 5); g.lineTo(0, 18); g.fill(); }
  if (s.a === 1) { g.fillStyle = '#ffa94d'; g.fillRect(-16, -6, 6, 4); }
  if (s.a === 3) { g.fillStyle = '#ffa94d'; g.fillRect(10, -6, 6, 4); }
  g.restore();
  g.fillStyle = '#ccc'; g.font = '11px system-ui';
  g.fillText(`flight ${D.trace.outcome}${D.trace.landed ? ' between the flags' : ''} · drift ${f2(s.in[8])}`, 8, 14);
}
function drawHeat() {
  const c = $('heat'), g = c.getContext('2d'), W = c.width, H = c.height;
  const st = D.trace.steps;
  const keys = NET.net.nodes.filter((n) => n.k >= 4).map((n) => String(n.k));
  const rows = keys.length, rh = (H - 30) / rows, cw = W / st.length;
  g.clearRect(0, 0, W, H);
  st.forEach((s, i) => keys.forEach((k, r) => {
    const v = s.h[k] ?? 0;
    g.fillStyle = v >= 0 ? `rgba(232,89,12,${Math.abs(v)})` : `rgba(59,91,219,${Math.abs(v)})`;
    g.fillRect(i * cw, r * rh, Math.ceil(cw), Math.ceil(rh));
  }));
  const ac = ['#555', '#e8590c', '#2f9e44', '#3b5bdb'];
  st.forEach((s, i) => { g.fillStyle = ac[s.a]; g.fillRect(i * cw, H - 24, Math.ceil(cw), 12); });
  g.fillStyle = col('--muted'); g.font = '10px system-ui';
  g.fillText('actions: grey nothing · orange left · green main · blue right', 2, H - 2);
  g.strokeStyle = col('--ink'); g.lineWidth = 1.5; g.beginPath(); g.moveTo(step * cw, 0); g.lineTo(step * cw, H - 12); g.stroke();
}

// ---------- 9. evolution ----------
async function drawEvolution() {
  const stats = await fetch('data/stats.json?t=' + Date.now()).then((r) => r.json()).catch(() => null);
  const rows = stats && stats[D.run] ? stats[D.run] : [];
  if (window.Chart && rows.length) new Chart($('sizec'), {
    type: 'line', data: { datasets: [
      { label: 'hidden neurons', data: rows.map((r) => ({ x: +r.gen, y: +r.hidden })), borderColor: col('--accent'), pointRadius: 0, yAxisID: 'y' },
      { label: 'connections', data: rows.map((r) => ({ x: +r.gen, y: +r.conns })), borderColor: col('--warn'), pointRadius: 0, yAxisID: 'y2' }] },
    options: { animation: false, maintainAspectRatio: false, scales: { x: { type: 'linear', title: { display: true, text: 'generation' } },
      y: { title: { display: true, text: 'hidden neurons' }, suggestedMin: 20 }, y2: { position: 'right', title: { display: true, text: 'connections' }, grid: { display: false } } } },
  });
  const ch = D.champions;
  $('champtable').innerHTML = '<tr><th>champion</th><th>gen</th><th>stage</th><th>fresh check: landed</th><th>score</th><th>hidden</th><th>connections</th><th>new neurons</th><th>new conns</th><th>avg weight change</th><th>biggest change</th></tr>' +
    ch.map((c) => `<tr><td>${esc(c.label)}</td><td>${c.gen}</td><td>${esc(c.stage)}</td><td>${c.check ? pct(c.check.landed) : '–'}</td><td>${c.check ? c.check.score.toFixed(0) : '–'}</td>
      <td>${c.net.nodes.filter((n) => n.k >= 4).length}</td><td>${c.net.conns.length}</td><td>${c.diff.new_nodes}</td><td>${c.diff.new_conns}</td><td>${f3(c.diff.mean_dw)}</td><td>${f2(c.diff.max_dw)}</td></tr>`).join('');
  if (window.Chart && ch.length) new Chart($('driftc'), {
    type: 'line', data: { datasets: [{ label: 'average |weight change| since the start network', data: ch.map((c) => ({ x: c.gen, y: c.diff.mean_dw })),
      borderColor: col('--good'), backgroundColor: col('--good'), pointRadius: 3 }] },
    options: { animation: false, maintainAspectRatio: false, scales: { x: { type: 'linear', title: { display: true, text: 'generation' } }, y: { beginAtZero: true } } },
  });
}

function fillConfig() {
  document.querySelectorAll('[data-cfg]').forEach((el) => {
    const v = D.config[el.dataset.cfg];
    el.textContent = el.hasAttribute('data-pct') ? (v * 100).toFixed(v < 0.01 ? 1 : 0) + '%' : v;
  });
}

let timer = null;
async function main() {
  D = await fetch('data/network.json?t=' + Date.now()).then((r) => r.json());
  const newest = D.champions.length ? D.champions[D.champions.length - 1] : null;
  NET = prepare(newest ? newest.net : D.seed, D.input_names.length);
  $('meta').textContent = `Run ${D.run}. Newest champion: ${newest ? `${newest.file} (generation ${newest.gen}, stage “${newest.stage}”)` : 'none yet, showing the start network'}. ` +
    `Recorded flight: seed ${D.trace.seed}, ${D.trace.steps.length} steps, ${D.trace.outcome}` +
    (D.trace.tries > 1 ? ` (the first of ${D.trace.tries} unseen flights that ${D.trace.landed ? 'landed' : 'were tried; none landed'}).` : '.');
  fillConfig();
  document.querySelectorAll('[data-net]').forEach((b) => b.onclick = () => {
    document.querySelectorAll('[data-net]').forEach((x) => x.classList.remove('on')); b.classList.add('on'); mode = b.dataset.net; drawNet();
  });
  $('thr').oninput = () => { thr = +$('thr').value; $('thrv').textContent = thr.toFixed(2); drawNet(); };
  $('tslider').max = D.trace.steps.length - 1;
  $('tslider').oninput = () => setStep(+$('tslider').value);
  $('play').onclick = () => {
    if (timer) { clearInterval(timer); timer = null; $('play').textContent = '▶ play'; return; }
    $('play').textContent = '❚❚ pause';
    timer = setInterval(() => { const n = step + 2; if (n >= D.trace.steps.length) { clearInterval(timer); timer = null; $('play').textContent = '▶ play'; return; } setStep(n); }, 40);
  };
  drawTanh();
  buildTry();
  setStep(D.presets.length > 2 ? D.presets[2].step : 0);
  drawEvolution();
}
main();
