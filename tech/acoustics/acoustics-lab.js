(() => {
  "use strict";
  const canvas = document.getElementById("acoustics-canvas");
  const detailCanvas = document.getElementById("detail-canvas");
  let ctx = canvas.getContext("2d");
  const mainContext = ctx;
  const detailContext = detailCanvas.getContext("2d");
  let activeView = "harmonics";
  const controls = {
    fundamental: document.getElementById("fundamental"), amplitude: document.getElementById("amplitude"), fundamentalPhase: document.getElementById("fundamental-phase"),
    fifth: document.getElementById("fifth"), fifthPhase: document.getElementById("fifth-phase"),
    seventh: document.getElementById("seventh"), seventhPhase: document.getElementById("seventh-phase"),
    cursorAngle: document.getElementById("cursor-angle"),
    showPhaseU: document.getElementById("show-phase-u"), showPhaseV: document.getElementById("show-phase-v"), showPhaseW: document.getElementById("show-phase-w")
  };
  const outputs = {
    fundamental: document.getElementById("fundamental-value"), amplitude: document.getElementById("amplitude-value"), fundamentalPhase: document.getElementById("fundamental-phase-value"),
    fifth: document.getElementById("fifth-value"), fifthPhase: document.getElementById("fifth-phase-value"), fifthVectorAngle: document.getElementById("fifth-vector-angle"),
    seventh: document.getElementById("seventh-value"), seventhPhase: document.getElementById("seventh-phase-value"), seventhVectorAngle: document.getElementById("seventh-vector-angle"), cursorAngle: document.getElementById("cursor-angle-value")
  };
  const colours = { u: "#1d4ed8", v: "#059669", w: "#c2410c", d: "#7c3aed", q: "#db2777", field: "#0f9f64", grid: "#cbd5e1", ink: "#0f172a", muted: "#64748b", card: "#ffffff" };
  const tau = Math.PI * 2;
  const sequenceOrder = { fundamental: 1, fifth: -5, seventh: 7 };
  const phases = [
    { key: "showPhaseU", label: "U", offset: 0, color: colours.u },
    { key: "showPhaseV", label: "V", offset: -tau / 3, color: colours.v },
    { key: "showPhaseW", label: "W", offset: tau / 3, color: colours.w }
  ];

  function state() {
    const i1 = Number(controls.amplitude.value); const i5 = i1 * Number(controls.fifth.value) / 100; const i7 = i1 * Number(controls.seventh.value) / 100;
    return { f: Number(controls.fundamental.value), i1, i5, i7, phi1: Number(controls.fundamentalPhase.value) * Math.PI / 180, cursorAngle: Number(controls.cursorAngle.value) * Math.PI / 180, gamma5: Number(controls.fifthPhase.value) * Math.PI / 180, gamma7: Number(controls.seventhPhase.value) * Math.PI / 180, shownPhases: phases.filter(phase => controls[phase.key].checked) };
  }
  function formatVectorAngle(angle) { const raw = Math.round(angle * 180 / Math.PI); const signed = ((raw + 180) % 360 + 360) % 360 - 180; const full = (signed + 360) % 360; return `${signed}° (${full}°)`; }
  function phaseAngleAtRotor(s, rotorAngle = s.cursorAngle) {
    let angle = rotorAngle + Math.PI / 2 + s.phi1;
    while (angle < -Math.PI) angle += tau;
    while (angle > 3 * Math.PI) angle -= tau;
    return angle;
  }
  function angleToX(angle, left, right) { return left + (right - left) * (angle + Math.PI) / (2 * tau); }
  function alphaBetaAngle(sequence, gamma, phaseAngle) { return sequence * phaseAngle + gamma - Math.PI / 2; }
  function updateLabels() { const s = state(); outputs.fundamental.textContent = `${s.f} Hz`; outputs.amplitude.textContent = `${s.i1.toFixed(1)} A`; outputs.fundamentalPhase.textContent = `${controls.fundamentalPhase.value}°`; outputs.fifth.textContent = `${controls.fifth.value} %`; outputs.fifthPhase.textContent = `${controls.fifthPhase.value}°`; outputs.fifthVectorAngle.textContent = formatVectorAngle(alphaBetaAngle(sequenceOrder.fifth, s.gamma5, phaseAngleAtRotor(s, 0))); outputs.seventh.textContent = `${controls.seventh.value} %`; outputs.seventhPhase.textContent = `${controls.seventhPhase.value}°`; outputs.seventhVectorAngle.textContent = formatVectorAngle(alphaBetaAngle(sequenceOrder.seventh, s.gamma7, phaseAngleAtRotor(s, 0))); outputs.cursorAngle.textContent = `${controls.cursorAngle.value}°`; }
  function line(x1, y1, x2, y2, color, width = 1, dash = []) { ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.restore(); }
  function text(value, x, y, size = 12, color = colours.ink, align = "left", weight = "400") { ctx.fillStyle = color; ctx.font = `${weight} ${size}px Inter, system-ui, sans-serif`; ctx.textAlign = align; ctx.textBaseline = "middle"; ctx.fillText(value, x, y); }
  function card(x, y, w, h, title, detail = "") { ctx.fillStyle = colours.card; ctx.strokeStyle = colours.grid; ctx.lineWidth = 1; ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h); text(title, x + 14, y + 17, 13, colours.ink, "left", "700"); if (detail) text(detail, x + w - 12, y + 17, 9, colours.muted, "right", "700"); }
  function drawCurrentAxis(left, right, top, bottom, limit, digits = 1) {
    const levels = [1, .5, 0, -.5, -1], mid = (top + bottom) / 2;
    line(left, top, left, bottom, colours.muted, 1);
    ctx.save(); ctx.translate(left - 48, mid); ctx.rotate(-Math.PI / 2); text("Current [A]", 0, 0, 8, colours.muted, "center"); ctx.restore();
    levels.forEach(level => {
      const py = mid - level * (bottom - top) / 2, value = level * limit;
      line(left - 4, py, left + 4, py, colours.muted, 1);
      line(left + 4, py, right, py, colours.grid, 1, level === 0 ? [4, 3] : [2, 4]);
      const label = level === 0 ? "0" : `${value > 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
      text(label, left - 7, py, 8, colours.muted, "right");
    });
  }
  function vector(cx, cy, x, y, color, label) { line(cx, cy, x, y, color, 3); const a = Math.atan2(y - cy, x - cx); const h = 8; ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - h * Math.cos(a - .48), y - h * Math.sin(a - .48)); ctx.lineTo(x - h * Math.cos(a + .48), y - h * Math.sin(a + .48)); ctx.closePath(); ctx.fill(); if (label) text(label, x + 8, y - 8, 10, color, "left", "700"); }
  function harmonicCurrent(amplitude, sequence, gamma, phase, theta) { return amplitude * Math.sin(sequence * theta + gamma + phase); }
  function dqComponents(amplitude, sequence, gamma, rotorAngle, phi1) {
    const phaseAngle = rotorAngle + Math.PI / 2 + phi1;
    const angle = alphaBetaAngle(sequence, gamma, phaseAngle) - rotorAngle;
    return { d: amplitude * Math.cos(angle), q: amplitude * Math.sin(angle) };
  }
  function current(phase, theta, s) {
    return s.i1 * Math.sin(theta + phase) + harmonicCurrent(s.i5, sequenceOrder.fifth, s.gamma5, phase, theta) + harmonicCurrent(s.i7, sequenceOrder.seventh, s.gamma7, phase, theta);
  }
  function drawWave(x, y, w, h, s) {
    card(x, y, w, h, "Three-phase currents iuvw", "two periods; values at cursor");
    const left = x + 60, right = x + w - 16, top = y + 40, bottom = y + h - 27, mid = (top + bottom) / 2, limit = 1.08 * (s.i1 + s.i5 + s.i7), scale = (bottom - top) / (2 * limit);
    const zeroX = left + (right - left) / 4, periodEndX = left + 3 * (right - left) / 4, cursorX = angleToX(phaseAngleAtRotor(s), left, right);
    drawCurrentAxis(left, right, top, bottom, limit); line(zeroX, top, zeroX, bottom, colours.grid, 1, [3, 3]); line(periodEndX, top, periodEndX, bottom, colours.grid, 1, [3, 3]); line(cursorX, top, cursorX, bottom, colours.field, 2); text("θ1 = −180°", left, y + h - 10, 9, colours.muted); text("θ1 = 0°", zeroX, y + h - 10, 9, colours.muted, "center"); text("θ1 = 360°", periodEndX, y + h - 10, 9, colours.muted, "center"); text("θ1 = +540°", right, y + h - 10, 9, colours.muted, "right"); text(`rotor θr = ${Math.round(s.cursorAngle * 180 / Math.PI)}°`, cursorX, top + 10, 9, colours.field, "center", "700");
    phases.forEach(({ offset: phase, color, label }, phaseIndex) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
      for (let px = left; px <= right; px += 2) { const local = -Math.PI + 2 * tau * (px - left) / (right - left); const py = mid - scale * current(phase, local, s); if (px === left) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
      ctx.stroke(); const value = current(phase, phaseAngleAtRotor(s), s); text(`${label}: ${value >= 0 ? "+" : ""}${value.toFixed(2)} A`, right - 2, top + 12 + phaseIndex * 16, 9, color, "right");
    });
  }
  function drawUPhaseHarmonics(x, y, w, h, s) {
    const harmonic5 = s.i5, harmonic7 = s.i7;
    card(x, y, w, h, "Selected phase harmonic components", "U blue, V green, W orange; solid −5th, dashed +7th");
    const selected = s.shownPhases, left = x + 60, right = x + w - 16, top = y + 72, bottom = y + h - 27, mid = (top + bottom) / 2, limit = Math.max(.1, 1.08 * (harmonic5 + harmonic7)), scale = (bottom - top) / (2 * limit), zeroX = left + (right - left) / 4, periodEndX = left + 3 * (right - left) / 4, cursorX = angleToX(phaseAngleAtRotor(s), left, right);
    drawCurrentAxis(left, right, top, bottom, limit, 2); line(zeroX, top, zeroX, bottom, colours.grid, 1, [3, 3]); line(periodEndX, top, periodEndX, bottom, colours.grid, 1, [3, 3]); line(cursorX, top, cursorX, bottom, colours.field, 2);
    const cursorValues = selected.map(phase => {
      const phaseAngle = phaseAngleAtRotor(s);
      const value5 = harmonicCurrent(harmonic5, sequenceOrder.fifth, s.gamma5, phase.offset, phaseAngle);
      const value7 = harmonicCurrent(harmonic7, sequenceOrder.seventh, s.gamma7, phase.offset, phaseAngle);
      return { phase, value5, value7 };
    });
    cursorValues.forEach(({ phase, value5, value7 }, index) => text(`${phase.label}: −5th ${value5.toFixed(2)} A, +7th ${value7.toFixed(2)} A`, left, y + 38 + index * 12, 8, phase.color));
    selected.forEach(phase => {
      [[sequenceOrder.fifth, s.gamma5, harmonic5, []], [sequenceOrder.seventh, s.gamma7, harmonic7, [5, 3]]].forEach(([order, gamma, amplitude, dash]) => {
        if (amplitude === 0) return;
        ctx.strokeStyle = phase.color; ctx.lineWidth = 2; ctx.setLineDash(dash); ctx.beginPath();
        for (let px = left; px <= right; px += 2) { const theta = -Math.PI + 2 * tau * (px - left) / (right - left), value = harmonicCurrent(amplitude, order, gamma, phase.offset, theta), py = mid - scale * value; if (px === left) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
        ctx.stroke(); ctx.setLineDash([]);
      });
    });
    text("θ1 = −180°", left, y + h - 10, 9, colours.muted); text("θ1 = 0°", zeroX, y + h - 10, 9, colours.muted, "center"); text("θ1 = 360°", periodEndX, y + h - 10, 9, colours.muted, "center"); text("θ1 = +540°", right, y + h - 10, 9, colours.muted, "right");
  }
  function drawSpaceVector(x, y, w, h, s) {
    card(x, y, w, h, "αβ space vectors", "at current rotor position"); const cx = x + w / 2, cy = y + h / 2 + 12, r = Math.min(w, h) * .30;
    const vectorLimit = Math.max(s.i1 + s.i5 + s.i7, 1);
    ctx.strokeStyle = colours.grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, r, 0, tau); ctx.stroke(); line(cx - r - 12, cy, cx + r + 12, cy, colours.grid); line(cx, cy - r - 12, cx, cy + r + 12, colours.grid);
    [-1, 0, 1].forEach(level => { const tickY = cy - level * r; line(cx - 4, tickY, cx + 4, tickY, colours.muted); text(level === 0 ? "0" : `${level > 0 ? "+" : "−"}${vectorLimit.toFixed(1)}`, cx - 8, tickY, 7, colours.muted, "right"); });
    const phaseAngle = phaseAngleAtRotor(s);
    const a1 = alphaBetaAngle(sequenceOrder.fundamental, 0, phaseAngle), a5 = alphaBetaAngle(sequenceOrder.fifth, s.gamma5, phaseAngle), a7 = alphaBetaAngle(sequenceOrder.seventh, s.gamma7, phaseAngle);
    const ratio = r / vectorLimit; const vectors = [[s.i1, a1, colours.u, "1st (+ω)"], [s.i5, a5, colours.q, "−5th (−5ω)"], [s.i7, a7, "#d97706", "+7th (+7ω)"]];
    const rotorRadius = r * .84; line(cx, cy, cx + rotorRadius * Math.cos(s.cursorAngle), cy - rotorRadius * Math.sin(s.cursorAngle), "#475569", 2, [5, 4]); text("rotor d", cx + (rotorRadius + 16) * Math.cos(s.cursorAngle), cy - (rotorRadius + 16) * Math.sin(s.cursorAngle), 9, "#475569", "center", "700"); let sx = 0, sy = 0; vectors.forEach(([mag, a, color, label]) => { const vx = mag * Math.cos(a) * ratio, vy = -mag * Math.sin(a) * ratio; sx += vx; sy += vy; if (mag > 0) vector(cx, cy, cx + vx, cy + vy, color, label); }); vector(cx, cy, cx + sx, cy + sy, colours.field, "resultant"); text("α [A]", cx + r + 18, cy + 3, 9, colours.muted); text("β [A]", cx - 4, cy - r - 18, 9, colours.muted);
  }
  function drawDq(x, y, w, h, s) {
    const harmonic5 = s.i5, harmonic7 = s.i7;
    card(x, y, w, h, "Rotor-synchronous dq currents", `−5th: −6f₁; +7th: +6f₁ (${6 * s.f} Hz); d solid, q dashed`);
    const left = x + 60, right = x + w - 16, top = y + 52, bottom = y + h - 27, mid = (top + bottom) / 2, limit = 1.08 * (s.i1 + harmonic5 + harmonic7), scale = (bottom - top) / (2 * limit), zeroX = left + (right - left) / 4, periodEndX = left + 3 * (right - left) / 4, cursorX = angleToX(phaseAngleAtRotor(s), left, right), h5AtCursor = dqComponents(harmonic5, sequenceOrder.fifth, s.gamma5, s.cursorAngle, s.phi1), h7AtCursor = dqComponents(harmonic7, sequenceOrder.seventh, s.gamma7, s.cursorAngle, s.phi1), currentD = s.i1 * Math.cos(s.phi1) + h5AtCursor.d + h7AtCursor.d, currentQ = s.i1 * Math.sin(s.phi1) + h5AtCursor.q + h7AtCursor.q;
    drawCurrentAxis(left, right, top, bottom, limit); line(zeroX, top, zeroX, bottom, colours.grid, 1, [3, 3]); line(periodEndX, top, periodEndX, bottom, colours.grid, 1, [3, 3]); line(cursorX, top, cursorX, bottom, colours.field, 2); text(`At cursor: d = ${currentD.toFixed(2)} A, q = ${currentQ.toFixed(2)} A`, left, y + 36, 9, colours.muted);
    const draw = (axis, color, label) => { ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(axis === "q" ? [5, 3] : []); ctx.beginPath(); for (let px = left; px <= right; px += 2) { const phaseAngle = -Math.PI + 2 * tau * (px - left) / (right - left), rotorAngle = phaseAngle - Math.PI / 2 - s.phi1, h5 = dqComponents(harmonic5, sequenceOrder.fifth, s.gamma5, rotorAngle, s.phi1), h7 = dqComponents(harmonic7, sequenceOrder.seventh, s.gamma7, rotorAngle, s.phi1), value = axis === "d" ? s.i1 * Math.cos(s.phi1) + h5.d + h7.d : s.i1 * Math.sin(s.phi1) + h5.q + h7.q, py = mid - value * scale; if (px === left) ctx.moveTo(px, py); else ctx.lineTo(px, py); } ctx.stroke(); ctx.setLineDash([]); text(label, right - 4, top + (axis === "d" ? 12 : 28), 9, color, "right"); };
    draw("d", colours.d, "d current"); draw("q", colours.q, "q current"); text("θ1 = −180°", left, y + h - 10, 9, colours.muted); text("θ1 = 0°", zeroX, y + h - 10, 9, colours.muted, "center"); text("θ1 = 360°", periodEndX, y + h - 10, 9, colours.muted, "center"); text("θ1 = +540°", right, y + h - 10, 9, colours.muted, "right");
  }
  function drawHarmonicFrameComponents(x, y, w, h, s) {
    const harmonic5 = s.i5, harmonic7 = s.i7;
    card(x, y, w, h, "dq sixth-frequency sum (harmonics only)", `−5th: −6f₁; +7th: +6f₁ (${6 * s.f} Hz); d solid, q dashed`);
    const left = x + 60, right = x + w - 16, top = y + 72, bottom = y + h - 27, mid = (top + bottom) / 2, limit = Math.max(.1, 1.08 * (harmonic5 + harmonic7)), scale = (bottom - top) / (2 * limit), zeroX = left + (right - left) / 4, periodEndX = left + 3 * (right - left) / 4, cursorX = angleToX(phaseAngleAtRotor(s), left, right);
    drawCurrentAxis(left, right, top, bottom, limit, 2); line(zeroX, top, zeroX, bottom, colours.grid, 1, [3, 3]); line(periodEndX, top, periodEndX, bottom, colours.grid, 1, [3, 3]); line(cursorX, top, cursorX, bottom, colours.field, 2);
    const at5 = dqComponents(harmonic5, sequenceOrder.fifth, s.gamma5, s.cursorAngle, s.phi1), at7 = dqComponents(harmonic7, sequenceOrder.seventh, s.gamma7, s.cursorAngle, s.phi1);
    text(`At cursor: d₆ = ${(at5.d + at7.d).toFixed(2)} A, q₆ = ${(at5.q + at7.q).toFixed(2)} A`, left, y + 42, 9, colours.muted);
    [["d", colours.d, []], ["q", colours.q, [5, 3]]].forEach(([axis, color, dash]) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash); ctx.beginPath();
      for (let px = left; px <= right; px += 2) {
        const theta1 = -Math.PI + 2 * tau * (px - left) / (right - left), rotorAngle = theta1 - Math.PI / 2 - s.phi1;
        const h5 = dqComponents(harmonic5, sequenceOrder.fifth, s.gamma5, rotorAngle, s.phi1);
        const h7 = dqComponents(harmonic7, sequenceOrder.seventh, s.gamma7, rotorAngle, s.phi1);
        const value = h5[axis] + h7[axis], py = mid - scale * value;
        if (px === left) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke(); ctx.setLineDash([]);
      text(`${axis}₆ current`, right - 4, top + (axis === "d" ? 12 : 28), 9, color, "right");
    });
    text("θ1 = −180°", left, y + h - 10, 9, colours.muted); text("θ1 = 0°", zeroX, y + h - 10, 9, colours.muted, "center"); text("θ1 = 360°", periodEndX, y + h - 10, 9, colours.muted, "center"); text("θ1 = +540°", right, y + h - 10, 9, colours.muted, "right");
  }
  function draw() {
    const s = state();
    ctx = mainContext; ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.fillStyle = "#f8fafc"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawWave(25, 5, 610, 220, s); drawSpaceVector(655, 5, 370, 220, s);
    ctx = detailContext; ctx.clearRect(0, 0, detailCanvas.width, detailCanvas.height); ctx.fillStyle = "#f8fafc"; ctx.fillRect(0, 0, detailCanvas.width, detailCanvas.height);
    if (activeView === "harmonics") drawUPhaseHarmonics(25, 5, 610, 220, s);
    if (activeView === "dq") drawDq(25, 5, 610, 220, s);
    if (activeView === "dq6") drawHarmonicFrameComponents(25, 5, 610, 220, s);
  }
  const tabs = { harmonics: document.getElementById("tab-harmonics"), dq: document.getElementById("tab-dq"), dq6: document.getElementById("tab-dq6") };
  function selectView(view) {
    activeView = view;
    Object.entries(tabs).forEach(([key, tab]) => { const selected = key === view; tab.setAttribute("aria-selected", String(selected)); tab.tabIndex = selected ? 0 : -1; });
    document.getElementById("phase-visibility").hidden = view !== "harmonics";
    detailCanvas.setAttribute("aria-labelledby", tabs[view].id);
    draw();
  }
  const viewOrder = Object.keys(tabs);
  Object.entries(tabs).forEach(([view, tab]) => {
    tab.addEventListener("click", () => selectView(view));
    tab.addEventListener("keydown", event => {
      const direction = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
      if (!direction) return;
      event.preventDefault();
      const next = viewOrder[(viewOrder.indexOf(view) + direction + viewOrder.length) % viewOrder.length];
      selectView(next); tabs[next].focus();
    });
  });
  Object.values(controls).forEach(control => control.addEventListener("input", () => { updateLabels(); draw(); }));
  selectView("harmonics"); updateLabels(); draw();
})();
