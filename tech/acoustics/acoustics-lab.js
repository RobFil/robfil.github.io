(() => {
  "use strict";
  const canvas = document.getElementById("acoustics-canvas");
  const ctx = canvas.getContext("2d");
  const controls = {
    fundamental: document.getElementById("fundamental"), amplitude: document.getElementById("amplitude"),
    fifth: document.getElementById("fifth"), fifthPhase: document.getElementById("fifth-phase"),
    seventh: document.getElementById("seventh"), seventhPhase: document.getElementById("seventh-phase"),
    angle: document.getElementById("angle"), compensation: document.getElementById("compensation")
  };
  const outputs = {
    fundamental: document.getElementById("fundamental-value"), amplitude: document.getElementById("amplitude-value"),
    fifth: document.getElementById("fifth-value"), fifthPhase: document.getElementById("fifth-phase-value"),
    seventh: document.getElementById("seventh-value"), seventhPhase: document.getElementById("seventh-phase-value"), angle: document.getElementById("angle-value")
  };
  const colours = { u: "#1d4ed8", v: "#059669", w: "#c2410c", d: "#7c3aed", q: "#db2777", field: "#0f9f64", grid: "#cbd5e1", ink: "#0f172a", muted: "#64748b", card: "#ffffff" };
  const tau = Math.PI * 2;

  function state() {
    const i1 = Number(controls.amplitude.value); const i5 = i1 * Number(controls.fifth.value) / 100; const i7 = i1 * Number(controls.seventh.value) / 100;
    return { f: Number(controls.fundamental.value), i1, i5, i7, angle: Number(controls.angle.value) * Math.PI / 180, compensated: controls.compensation.checked, phi5: Number(controls.fifthPhase.value) * Math.PI / 180, phi7: Number(controls.seventhPhase.value) * Math.PI / 180 };
  }
  function updateLabels() { const s = state(); outputs.fundamental.textContent = `${s.f} Hz`; outputs.amplitude.textContent = `${s.i1.toFixed(1)} A`; outputs.fifth.textContent = `${controls.fifth.value} %`; outputs.fifthPhase.textContent = `${controls.fifthPhase.value}°`; outputs.seventh.textContent = `${controls.seventh.value} %`; outputs.seventhPhase.textContent = `${controls.seventhPhase.value}°`; outputs.angle.textContent = `${controls.angle.value}°`; }
  function line(x1, y1, x2, y2, color, width = 1, dash = []) { ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.restore(); }
  function text(value, x, y, size = 12, color = colours.ink, align = "left", weight = "400") { ctx.fillStyle = color; ctx.font = `${weight} ${size}px Inter, system-ui, sans-serif`; ctx.textAlign = align; ctx.textBaseline = "middle"; ctx.fillText(value, x, y); }
  function card(x, y, w, h, title, detail = "") { ctx.fillStyle = colours.card; ctx.strokeStyle = colours.grid; ctx.lineWidth = 1; ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h); text(title, x + 14, y + 17, 13, colours.ink, "left", "700"); if (detail) text(detail, x + w - 12, y + 17, 9, colours.muted, "right", "700"); }
  function vector(cx, cy, x, y, color, label) { line(cx, cy, x, y, color, 3); const a = Math.atan2(y - cy, x - cx); const h = 8; ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - h * Math.cos(a - .48), y - h * Math.sin(a - .48)); ctx.lineTo(x - h * Math.cos(a + .48), y - h * Math.sin(a + .48)); ctx.closePath(); ctx.fill(); if (label) text(label, x + 8, y - 8, 10, color, "left", "700"); }
  function current(phase, theta, s) {
    const i5 = s.compensated ? 0 : s.i5; const i7 = s.compensated ? 0 : s.i7;
    return s.i1 * Math.cos(theta + phase) + i5 * Math.cos(5 * (theta + phase) + s.phi5) + i7 * Math.cos(7 * (theta + phase) + s.phi7);
  }
  function drawWave(x, y, w, h, s) {
    card(x, y, w, h, "Three-phase currents iuvw", s.compensated ? "ideal cancellation on" : "instantaneous values at t = 0");
    const left = x + 42, right = x + w - 16, top = y + 38, bottom = y + h - 27, mid = (top + bottom) / 2, limit = 1.08 * (s.compensated ? s.i1 : s.i1 + s.i5 + s.i7), scale = (bottom - top) / (2 * limit);
    line(left, mid, right, mid, colours.grid, 1, [4, 3]); text("t = 0 ms", left, y + h - 10, 9, colours.muted); text("20 ms", right, y + h - 10, 9, colours.muted, "right");
    [[0, colours.u, "u"], [-2 * Math.PI / 3, colours.v, "v"], [2 * Math.PI / 3, colours.w, "w"]].forEach(([phase, color, label]) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
      for (let px = left; px <= right; px += 2) { const local = s.angle + tau * s.f * .02 * (px - left) / (right - left); const py = mid - scale * current(phase, local, s); if (px === left) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
      ctx.stroke(); const value = current(phase, s.angle, s); text(`${label}: ${value >= 0 ? "+" : ""}${value.toFixed(2)} A`, right - 2, top + 12 + ["u", "v", "w"].indexOf(label) * 16, 9, color, "right");
    });
    text(`±${limit.toFixed(1)}`, x + 7, top, 8, colours.muted); text(`−${limit.toFixed(1)}`, x + 7, bottom, 8, colours.muted);
  }
  function drawSpaceVector(x, y, w, h, s) {
    card(x, y, w, h, "αβ space vectors", "instantaneous at t = 0"); const cx = x + w / 2, cy = y + h / 2 + 12, r = Math.min(w, h) * .30;
    ctx.strokeStyle = colours.grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, r, 0, tau); ctx.stroke(); line(cx - r - 12, cy, cx + r + 12, cy, colours.grid); line(cx, cy - r - 12, cx, cy + r + 12, colours.grid);
    const a1 = s.angle, a5 = -5 * s.angle - s.phi5, a7 = 7 * s.angle + s.phi7;
    const ratio = r / Math.max(s.i1 + s.i5 + s.i7, 1); const vectors = [[s.i1, a1, colours.u, "1st +ω"], [s.compensated ? 0 : s.i5, a5, colours.q, "5th −5ω"], [s.compensated ? 0 : s.i7, a7, "#d97706", "7th +7ω"]];
    let sx = 0, sy = 0; vectors.forEach(([mag, a, color, label]) => { const vx = mag * Math.cos(a) * ratio, vy = -mag * Math.sin(a) * ratio; sx += vx; sy += vy; if (mag > 0) vector(cx, cy, cx + vx, cy + vy, color, label); }); vector(cx, cy, cx + sx, cy + sy, colours.field, "resultant"); text("α", cx + r + 18, cy + 3, 10, colours.muted); text("β", cx - 4, cy - r - 18, 10, colours.muted);
  }
  function drawDq(x, y, w, h, s) {
    card(x, y, w, h, "Current-synchronous dq frame", `harmonic ripple at |6 f1| = ${6 * s.f} Hz`); const left = x + 42, right = x + w - 16, top = y + 38, bottom = y + h - 27, mid = (top + bottom) / 2, scale = (bottom - top) / (2 * Math.max(s.i5 + s.i7, 1)); line(left, mid, right, mid, colours.grid, 1, [4, 3]);
    const draw = (axis, color, label) => { if (s.compensated || (s.i5 === 0 && s.i7 === 0)) return; ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath(); for (let px = left; px <= right; px += 2) { const time = .02 * (px - left) / (right - left); const harmonicAngle = tau * 6 * s.f * time + 6 * s.angle; const value = axis === "d" ? s.i5 * Math.cos(harmonicAngle + s.phi5) + s.i7 * Math.cos(harmonicAngle + s.phi7) : -s.i5 * Math.sin(harmonicAngle + s.phi5) + s.i7 * Math.sin(harmonicAngle + s.phi7); const py = mid - value * scale; if (px === left) ctx.moveTo(px, py); else ctx.lineTo(px, py); } ctx.stroke(); text(label, right - 4, top + (axis === "d" ? 12 : 28), 9, color, "right"); };
    draw("d", colours.d, "d harmonic"); draw("q", colours.q, "q harmonic"); text("d/q harmonic ripple", left, y + h - 10, 9, colours.muted); text("t = 0 ms", left + 150, y + h - 10, 9, colours.muted); text("20 ms", right, y + h - 10, 9, colours.muted, "right"); if (s.compensated) text("Only DC fundamental remains", x + w / 2, mid, 12, colours.field, "center", "700");
  }
  function drawMsrft(x, y, w, h, s) {
    card(x, y, w, h, "MSRFT harmonic channels", "harmonics become DC"); const channels = [["5th in −6ω", s.compensated ? 0 : s.i5, s.phi5, -1, colours.q], ["7th in +6ω", s.compensated ? 0 : s.i7, s.phi7, 1, "#d97706"]];
    channels.forEach(([label, mag, phase, sign, color], index) => { const cx = x + 125 + index * 185, cy = y + 105, r = 43; ctx.strokeStyle = colours.grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, r, 0, tau); ctx.stroke(); line(cx - r - 6, cy, cx + r + 6, cy, colours.grid); line(cx, cy - r - 6, cx, cy + r + 6, colours.grid); const d = mag * Math.cos(phase), q = sign * mag * Math.sin(phase); const scale = r / Math.max(s.i1 * .35, 1); vector(cx, cy, cx + d * scale, cy - q * scale, color, "DC"); text(label, cx, y + 45, 11, colours.ink, "center", "700"); text(`d: ${d.toFixed(2)} A`, cx, y + 166, 10, colours.muted, "center"); text(`q: ${q.toFixed(2)} A`, cx, y + 184, 10, colours.muted, "center"); text(`|I|: ${mag.toFixed(2)} A`, cx, y + 202, 10, color, "center", "700"); });
    text(s.compensated ? "Counter-currents cancel both harmonic components." : "Rotate into ±6ω frames to obtain constant d/q values.", x + w / 2, y + h - 16, 10, colours.muted, "center");
  }
  function draw() { const s = state(); ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.fillStyle = "#f8fafc"; ctx.fillRect(0, 0, canvas.width, canvas.height); drawWave(25, 25, 610, 275, s); drawSpaceVector(655, 25, 370, 275, s); drawDq(25, 325, 610, 270, s); drawMsrft(655, 325, 370, 270, s); text(`f₁ = ${s.f} Hz   |   I₁ = ${s.i1.toFixed(1)} A   |   I₅ = ${s.i5.toFixed(2)} A   |   I₇ = ${s.i7.toFixed(2)} A`, 25, 640, 12, colours.ink, "left", "700"); text("Learning model: ideal balanced three-phase currents and ideal MSRFT extraction.", 25, 665, 10, colours.muted); }
  Object.values(controls).forEach(control => control.addEventListener("input", () => { updateLabels(); draw(); })); updateLabels(); draw();
})();
