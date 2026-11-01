(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const ids = ['rpm', 'poles', 'fsw', 'offset', 'ud', 'uq', 'm'];
  const controls = Object.fromEntries(ids.map(id => [id, $(id)]));
  const colors = { ua: '#2266aa', ub: '#18835b', uc: '#8651a5', ud: '#18835b', uq: '#a36a32' };
  let playing = true, position = 0, last = 0, config, samples;
  const tau = 2 * Math.PI;
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  function rebuild() {
    const n = id => Number(controls[id].value);
    if (ids.some(id => controls[id].value === '' || !Number.isFinite(n(id)))) return;
    config = { rpm: clamp(n('rpm'), 1, 100000), poles: Math.round(clamp(n('poles'), 1, 100)), fsw: clamp(n('fsw'), 1, 1000000), offset: n('offset') * Math.PI / 180, ud: n('ud'), uq: n('uq'), m: n('m') };
    const c = config;
    c.fel = c.rpm / 60 * c.poles;
    c.vdc = Math.hypot(c.ud, c.uq) * Math.sqrt(6) / Math.max(c.m, 1e-6);
    c.linear = c.vdc / Math.sqrt(3); c.radius = 2 * c.vdc / 3;
    c.request = Math.hypot(c.ud, c.uq) * Math.SQRT2;
    c.step = tau * c.fel / c.fsw;
    c.lead = Math.atan2(-c.uq, c.ud);
    // Bound the display workload while retaining the actual PWM sample positions.
    const totalSteps = Math.ceil(6 * tau / c.step);
    c.stride = Math.max(1, Math.ceil(totalSteps / 2000));
    const count = Math.ceil(totalSteps / c.stride);
    samples = Array.from({ length: count + 1 }, (_, i) => signal(Math.min(i * c.stride * c.step, 6 * tau)));
    $('m-value').textContent = c.m.toFixed(3);
    render();
  }
  function signal(progress) {
    const c = config, theta = progress + c.offset;
    const sector = ((theta % (Math.PI / 3)) + Math.PI / 3) % (Math.PI / 3) - Math.PI / 6;
    const applied = Math.min(c.request, c.linear / Math.cos(sector));
    const alpha = applied * Math.cos(theta), beta = applied * Math.sin(theta);
    const ratio = c.request > 1e-12 ? applied / c.request : 0;
    return { progress, theta, alpha, beta, applied, ua: alpha, ub: -alpha / 2 + Math.sqrt(3) * beta / 2, uc: -alpha / 2 - Math.sqrt(3) * beta / 2, ud: c.ud * ratio, uq: c.uq * ratio };
  }
  function line(ctx, points, color, width = 2, dash = []) {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.beginPath();
    points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke(); ctx.setLineDash([]);
  }
  function dot(ctx, x, y, color) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, 5, 0, tau); ctx.fill(); }
  function hexagon(current, index) {
    const canvas = $('hex'), ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height, cx = w / 2, cy = h / 2;
    ctx.clearRect(0, 0, w, h); const scale = Math.min(w, h) * .4 / Math.max(config.radius, 1);
    const xy = (x, y) => [cx + x * scale, cy - y * scale];
    line(ctx, [[25, cy], [w - 25, cy]], '#d5dde2', 1); line(ctx, [[cx, 20], [cx, h - 20]], '#d5dde2', 1);
    const polygon = Array.from({ length: 7 }, (_, i) => xy(config.radius * Math.cos(i * tau / 6), config.radius * Math.sin(i * tau / 6)));
    line(ctx, polygon, '#183b50', 3);
    line(ctx, Array.from({ length: 121 }, (_, i) => xy(config.linear * Math.cos(i * tau / 120), config.linear * Math.sin(i * tau / 120))), '#8c9ca6', 2, [6, 5]);
    line(ctx, samples.slice(Math.max(0, index - 40), index + 1).map(s => xy(s.alpha, s.beta)), '#d78118', 2);
    const rotor = xy(current.applied * Math.cos(current.theta + config.lead), current.applied * Math.sin(current.theta + config.lead));
    line(ctx, [[cx, cy], rotor], '#697783', 2, [6, 5]);
    const tip = xy(current.alpha, current.beta); line(ctx, [[cx, cy], tip], '#c73335', 4); dot(ctx, ...tip, '#c73335');
    ctx.fillStyle = '#526372'; ctx.font = '16px sans-serif'; ctx.fillText('uα', w - 30, cy - 8); ctx.fillText('uβ', cx + 8, 22);
  }
  function plot(id, keys, current, index) {
    const canvas = $(id), ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height, left = 55, right = w - 18, top = 15, bottom = h - 38;
    ctx.clearRect(0, 0, w, h);
    const values = samples.flatMap(s => keys.map(k => s[k]));
    const max = Math.max(1, ...values.map(Math.abs)) * 1.12;
    const x = s => left + s.progress / (6 * tau) * (right - left), y = v => (top + bottom) / 2 - v / max * (bottom - top) / 2;
    ctx.font = '13px sans-serif'; ctx.fillStyle = '#687986';
    [-1, 0, 1].forEach(v => { const yy = y(v * max); line(ctx, [[left, yy], [right, yy]], '#e3e9ed', 1); ctx.fillText((v * max).toFixed(0), 2, yy + 4); });
    for (let i = 0; i <= 6; i++) { const xx = left + i / 6 * (right - left); line(ctx, [[xx, top], [xx, bottom]], '#edf1f3', 1); ctx.fillText(id === 'phase' ? `${i * 360}°` : `${(i / config.fel * 1000).toFixed(1)}`, xx - 10, h - 15); }
    ctx.fillText(id === 'phase' ? 'Electrical angle from offset' : 'Time [ms]', left, h - 1);
    keys.forEach(key => {
      const draw = (arr, color, width) => { const pts = []; arr.forEach((s, i) => { if (i) pts.push([x(s), y(arr[i - 1][key])]); pts.push([x(s), y(s[key])]); }); line(ctx, pts, color, width); };
      ctx.globalAlpha = .25; draw(samples, colors[key], 1.5); ctx.globalAlpha = 1; draw(samples.slice(0, index + 1), colors[key], 2.5); dot(ctx, x(current), y(current[key]), colors[key]);
    });
    line(ctx, [[x(current), top], [x(current), bottom]], '#687986', 1, [5, 4]);
  }
  function render() {
    if (!config || !samples.length) return;
    const index = clamp(Math.floor(position / (config.step * config.stride)), 0, samples.length - 1), s = samples[index], c = config;
    $('angle').value = position * 180 / Math.PI; $('angle-value').textContent = `${(s.progress * 180 / Math.PI).toFixed(1)}°`;
    $('readouts').innerHTML = `<span>DC link <strong>${c.vdc.toFixed(1)} V</strong></span><span>Electrical frequency <strong>${c.fel.toFixed(1)} Hz</strong></span><span>PWM angle step <strong>${(c.step * 180 / Math.PI).toFixed(2)}°</strong></span><span>Time <strong>${(s.progress / tau / c.fel * 1000).toFixed(3)} ms</strong></span><span>Applied m <strong>${(c.linear ? s.applied / c.linear : 0).toFixed(3)}</strong></span><span>Limited <strong>${c.request > s.applied + 1e-9 ? 'yes' : 'no'}</strong></span>`;
    hexagon(s, index); plot('phase', ['ua', 'ub', 'uc'], s, index); plot('dq', ['ud', 'uq'], s, index);
  }
  ids.forEach(id => controls[id].addEventListener('input', rebuild));
  $('angle').addEventListener('input', e => { position = Number(e.target.value) * Math.PI / 180; render(); });
  $('play').addEventListener('click', () => { playing = !playing; $('play').textContent = playing ? 'Pause' : 'Play'; last = 0; });
  $('reset').addEventListener('click', () => { position = 0; last = 0; render(); });
  function tick(now) { if (playing && last) { position = (position + (now - last) / 1000 * tau / 5) % (6 * tau); render(); } last = now; requestAnimationFrame(tick); }
  rebuild(); requestAnimationFrame(tick);
})();
