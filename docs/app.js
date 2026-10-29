const $ = (id) => document.getElementById(id);
const pct = (x) => (x === '' || x == null || isNaN(x)) ? '–' : (100 * x).toFixed(1).replace(/\.0$/, '') + '%';
const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const get = (f) => fetch('data/' + f + '?t=' + Date.now()).then((r) => r.ok ? r.json() : null).catch(() => null);

function worldText(w = {}, ramp = {}) {
  const p = [];
  const wind = ramp.wind ? `${ramp.wind[0]}→${ramp.wind[1]}` : (w.wind ?? 15);
  const turb = ramp.turbulence ? `${ramp.turbulence[0]}→${ramp.turbulence[1]}` : (w.turbulence ?? 1.5);
  if (w.steady) p.push(`steady push ${wind} (one constant force, either way)`);
  else p.push(+wind === 0 ? 'calm' : `wind ${wind}`);
  if (!w.steady && +turb !== 0) p.push(`turbulence ${turb}`);
  if (w.boulder) p.push('boulder');
  return p.join(' + ');
}
function rewardText(r = {}) {
  const p = ['env score'];
  p.push(`+${r.landing_bonus ?? 100} for landing between the flags, + up to 50 for the centre`);
  if (r.timeout_penalty) p.push(`−${r.timeout_penalty} for a timeout (no hovering)`);
  if (r.boulder_penalty) p.push(`−${r.boulder_penalty} extra for touching the boulder (on top of the crash)`);
  if (r.clearance_bonus) p.push(`+ up to ${r.clearance_bonus} for keeping clear of it`);
  return p.join(', ');
}
function passText(p = {}) {
  const q = [];
  if (p.landed) q.push(`≥ ${pct(p.landed)} landed`);
  if (p.score) q.push(`score ≥ ${p.score}`);
  if (p.boulder === 0) q.push('no boulder hits');
  return q.join(', ');
}

async function main() {
  const [status, stats, log, cur] = await Promise.all([get('status.json'), get('stats.json'), get('log.json'), get('curricula.json')]);
  if (!status) { $('updated').textContent = 'No data yet: training is being set up.'; return; }
  $('updated').textContent = `Updated ${status.updated}. Deadline ${status.deadline}.`;
  const runs = status.runs || [];
  const main = runs.find((r) => r.alive) || runs[runs.length - 1];
  if (!main) return;
  const last = main.last || {};
  const c = cur[main.run];
  const stIdx = c.stages.findIndex((s) => s.name === last.stage);
  const best = main.best_check;
  $('cards').innerHTML = [
    ['Run', main.run, main.alive ? 'training now' : 'stopped'],
    ['Stage', `${stIdx + 1} / ${c.stages.length}`, last.stage || ''],
    ['Generation', last.gen ?? '–', `${last.secs ?? '–'} s per generation`],
    ['Champion (train+select)', pct(+last.champ_landed), `score ${last.champ_score ?? '–'}`],
    ['Best final-stage check', best ? pct(best.landed) : '–', best ? `100 fresh flights, score ${best.score.toFixed(0)}` : 'not reached yet'],
    ['Goal', '98%', '1,000 unseen flights'],
  ].map(([k, v, s]) => `<div class="card"><div class="k">${k}</div><div class="v">${esc(v)}</div><div class="s">${esc(s)}</div></div>`).join('');

  const evs = (log || []).filter((e) => e.run === main.run);
  $('stages').innerHTML = c.stages.map((s, i) => {
    const passed = evs.some((e) => e.kind === 'stage_pass' && e.stage_index === i);
    const capped = evs.some((e) => e.kind === 'stage_cap' && e.stage_index === i);
    const cls = passed ? 'passed' : (i === stIdx && main.alive) ? 'current' : capped ? 'capped' : '';
    const state = passed ? 'passed' : i === stIdx ? 'running' : capped ? 'cap reached, moved on' : i < stIdx ? 'done' : 'waiting';
    const sup = s.support ? `<dt>Training wheels</dt><dd>an upright-pulling torque, ${s.support[0]} → 0 over the ramp; checks fly without it</dd>` : '';
    return `<div class="stage ${cls}"><div class="n">${i + 1}</div><div>
      <h3>${esc(s.name)} <span class="muted small">· ${state} · cap ${s.cap} generations</span></h3>
      <dl><dt>World</dt><dd>${esc(worldText(s.world, s.ramp))}</dd>${sup}
      <dt>Reward</dt><dd>${esc(rewardText(s.reward))}</dd>
      <dt>Pass</dt><dd>${esc(passText(s.pass))}</dd>
      <dt>Expected</dt><dd>${esc(s.expect || '')}</dd></dl></div></div>`;
  }).join('');

  const rows = (stats && stats[main.run]) || [];
  const css = getComputedStyle(document.documentElement);
  const col = (v) => css.getPropertyValue(v).trim();
  const bands = [];
  rows.forEach((r, i) => { if (!i || r.stage !== rows[i - 1].stage) bands.push({ start: +r.gen, stage: r.stage }); });
  bands.forEach((b, i) => b.end = i + 1 < bands.length ? bands[i + 1].start : +rows[rows.length - 1].gen + 1);
  const bandPlugin = { id: 'bands', beforeDraw(ch) {
    const { ctx, chartArea: a, scales: { x } } = ch;
    bands.forEach((b, i) => {
      const x0 = x.getPixelForValue(b.start), x1 = x.getPixelForValue(b.end);
      if (i % 2 === 0) { ctx.fillStyle = col('--band'); ctx.fillRect(x0, a.top, x1 - x0, a.bottom - a.top); }
      ctx.fillStyle = col('--muted'); ctx.font = '11px system-ui'; ctx.fillText(b.stage, x0 + 4, a.top + 12);
    });
  } };
  // y-axis autoscale: from just below the lowest value shown (rounded down to 5) up to 100
  const ys = rows.flatMap((r) => [100 * r.champ_landed, r.check_landed === '' ? 100 : 100 * r.check_landed]).filter((v) => !isNaN(v));
  const yMin = ys.length ? Math.max(0, Math.floor((Math.min(...ys) - 3) / 5) * 5) : 0;
  if (window.Chart && rows.length) new Chart($('chart'), {
    type: 'line', plugins: [bandPlugin],
    data: { datasets: [
      { label: 'champion, train + selection flights', data: rows.map((r) => ({ x: +r.gen, y: 100 * r.champ_landed })),
        borderColor: col('--accent'), borderWidth: 1.5, pointRadius: 0, tension: .2 },
      { label: 'champion, 100 fresh check flights', data: rows.filter((r) => r.check_landed !== '').map((r) => ({ x: +r.gen, y: 100 * r.check_landed })),
        showLine: false, pointRadius: 3.5, backgroundColor: col('--good'), borderColor: col('--good') },
      { label: 'goal 98%', data: rows.length ? [{ x: +rows[0].gen, y: 98 }, { x: +rows[rows.length - 1].gen, y: 98 }] : [],
        borderColor: col('--bad'), borderDash: [5, 4], borderWidth: 1, pointRadius: 0 },
    ] },
    options: { animation: false, maintainAspectRatio: false, interaction: { mode: 'nearest', intersect: false },
      scales: { x: { type: 'linear', title: { display: true, text: 'generation', color: col('--muted') }, ticks: { color: col('--muted') }, grid: { color: col('--line') } },
                y: { min: yMin, max: 100, title: { display: true, text: '% landed', color: col('--muted') }, ticks: { color: col('--muted') }, grid: { color: col('--line') } } },
      plugins: { legend: { labels: { color: col('--ink'), boxWidth: 12 } } } },
  });

  $('log').innerHTML = (log || []).filter((e) => e.kind !== 'run_start' || true).map(eventHtml).join('') || '<p class="muted">Nothing yet.</p>';
}

function chk(c) { return c ? `${pct(c.landed)} landed, score ${c.score.toFixed(0)}, crashed ${pct(c.crashed)}, boulder ${pct(c.boulder)}, timeout ${pct(c.timeout)}` : ''; }

function eventHtml(e) {
  let title = '', body = '';
  switch (e.kind) {
    case 'journal': title = e.title; body = e.text; break;
    case 'run_start': title = `Run ${e.run} started`; body = e.note; break;
    case 'stage_start': title = `Stage started: ${e.stage}`; body = e.expect ? 'Expected: ' + e.expect : ''; break;
    case 'new_best': title = `New best champion in ${e.stage} (gen ${e.gen})`; body = 'On 100 fresh flights: ' + chk(e.check); break;
    case 'check_pass': title = `Check passed in ${e.stage}, confirming on a second 100`; body = `First: ${chk(e.check)}\nConfirm: ${chk(e.confirm)}`; break;
    case 'stage_pass': title = `✅ Stage passed: ${e.stage} (gen ${e.gen})`; body = `Check: ${chk(e.check)}\nConfirm: ${chk(e.confirm)}`; break;
    case 'stage_cap': title = `Stage ${e.stage} hit its generation cap; moving on`; break;
    case 'override': title = `Manual change: ${e.what}`; body = e.reason; break;
    case 'curriculum_done': title = 'Curriculum finished'; break;
    case 'goal_check': title = `🎯 Goal check: ${pct(e.landed)} landed on ${e.n} unseen flights`; body = `95% interval ${pct(e.ci[0])} – ${pct(e.ci[1])}. Score ${e.score.toFixed(0)}. ${e.note || ''}`; break;
    default: title = e.kind;
  }
  const img = e.replay ? `<img loading="lazy" src="${e.replay}" alt="replay of the champion at this moment">` : '';
  return `<div class="ev"><div class="body"><div class="t">${esc(e.t)}${e.run ? ' · run ' + esc(e.run) : ''}</div>
    <h4><span class="tag ${e.kind}">${e.kind.replace('_', ' ')}</span>${esc(title)}</h4><p>${esc(body)}</p></div>${img}</div>`;
}
main();
