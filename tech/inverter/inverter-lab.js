(() => {
  "use strict";

  const WIDTH = 1050;
  const HEIGHT = 660;
  const phaseNames = ["U", "V", "W"];
  const phaseAngles = [0, (2 * Math.PI) / 3, (4 * Math.PI) / 3];
  const phaseColours = ["#1d4ed8", "#059669", "#c2410c"];
  const canvases = {
    coil: document.getElementById("coil-canvas"),
    test: document.getElementById("test-canvas"),
    motor: document.getElementById("motor-canvas"),
  };
  const state = {
    activeTab: "coil",
    coil: { voltage: 0, current: 0, history: [{ u: 0, i: 0, b: 0 }] },
    test: { active: [false, false, false], theta: 0, speed: 0, torque: 0, history: [0] },
    motor: { speed: 0, amplitude: 50, phase: 90, theta: 0 },
  };

  function prepare(canvas) {
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const pixelWidth = Math.round(WIDTH * ratio);
    const pixelHeight = Math.round(HEIGHT * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, WIDTH, HEIGHT);
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(0, 0, WIDTH, HEIGHT);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    return ctx;
  }

  function font(ctx, size, weight = "400") {
    ctx.font = `${weight} ${size}px "Segoe UI", Arial, sans-serif`;
  }

  function text(ctx, value, x, y, size = 12, colour = "#0f172a", align = "left", weight = "400") {
    ctx.fillStyle = colour;
    ctx.textAlign = align;
    ctx.textBaseline = "middle";
    font(ctx, size, weight);
    ctx.fillText(value, x, y);
  }

  function line(ctx, x1, y1, x2, y2, colour = "#334155", width = 2, dash = []) {
    ctx.save();
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.restore();
  }

  function arrow(ctx, x1, y1, x2, y2, colour, width = 3) {
    const angle = Math.atan2(y2 - y1, x2 - x1);
    line(ctx, x1, y1, x2, y2, colour, width);
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 11 * Math.cos(angle - Math.PI / 6), y2 - 11 * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x2 - 11 * Math.cos(angle + Math.PI / 6), y2 - 11 * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  }

  function card(ctx, x, y, width, height, title) {
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1;
    ctx.fillRect(x, y, width, height);
    ctx.strokeRect(x, y, width, height);
    text(ctx, title, x + 12, y + 17, 12, "#0f172a", "left", "700");
  }

  function clamp(value, low, high) { return Math.min(high, Math.max(low, value)); }
  function wrapAngle(angle) { return ((angle + Math.PI) % (2 * Math.PI)) - Math.PI; }
  function degrees(angle) { return ((angle * 180) / Math.PI + 360) % 360; }

  function drawTrace(ctx, x, y, width, height, title, unit, values, limit, colour) {
    card(ctx, x, y, width, height, title);
    const left = x + 55;
    const right = x + width - 18;
    const top = y + 34;
    const bottom = y + height - 18;
    const mid = (top + bottom) / 2;
    line(ctx, left, mid, right, mid, "#94a3b8", 1, [3, 3]);
    text(ctx, `+${limit}`, x + 9, top, 9, "#64748b");
    text(ctx, `-${limit}`, x + 9, bottom, 9, "#64748b");
    text(ctx, unit, right, y + height - 8, 9, "#64748b", "right");
    if (!values.length) return;
    ctx.save();
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((value, index) => {
      const px = left + ((right - left) * index) / Math.max(1, values.length - 1);
      const py = mid - ((bottom - top) * 0.43 * clamp(value, -limit, limit)) / limit;
      if (index === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.restore();
    const value = values[values.length - 1];
    const markerY = mid - ((bottom - top) * 0.43 * clamp(value, -limit, limit)) / limit;
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(right, markerY, 4, 0, 2 * Math.PI);
    ctx.fill();
    text(ctx, `${value >= 0 ? "+" : ""}${value.toFixed(2)} ${unit}`, right - 4, y + 17, 9, colour, "right", "700");
  }

  function drawCoil() {
    const ctx = prepare(canvases.coil);
    const { voltage, current, history } = state.coil;
    const magneticField = current * 0.05;
    text(ctx, "Coil with battery", 25, 25, 18, "#0f172a", "left", "700");
    text(ctx, `U_bat = ${voltage >= 0 ? "+" : ""}${voltage.toFixed(1)} V   |   i = ${current >= 0 ? "+" : ""}${current.toFixed(2)} A   |   B = ${magneticField >= 0 ? "+" : ""}${magneticField.toFixed(3)} T`, 25, 55, 12, "#166534", "left", "700");
    drawTrace(ctx, 25, 95, 610, 195, "Coil current i", "A", history.map(sample => sample.i), 10, "#d97706");
    drawTrace(ctx, 25, 310, 610, 195, "Magnetic field B", "T", history.map(sample => sample.b), 0.5, "#16a34a");
    drawTrace(ctx, 25, 525, 610, 90, "Battery voltage U_bat", "V", history.map(sample => sample.u), 10, "#dc2626");

    card(ctx, 655, 75, 370, 400, "Single coil");
    text(ctx, "Current creates a magnetic field", 840, 117, 9, "#475569", "center");
    const coilLeft = 710;
    const coilRight = 970;
    const coilY = 276;
    for (let turn = 0; turn < 8; turn += 1) {
      const turnX = coilLeft + 20 + turn * 31;
      ctx.strokeStyle = "#7c3aed";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(turnX, coilY, 12, 59, 0, 0, 2 * Math.PI);
      ctx.stroke();
    }
    line(ctx, coilLeft - 18, coilY - 59, coilRight - 8, coilY - 59, "#7c3aed", 2);
    line(ctx, coilLeft + 8, coilY + 59, coilRight + 18, coilY + 59, "#7c3aed", 2);
    const currentLength = 12 + 112 * Math.min(Math.abs(current) / 10, 1);
    if (Math.abs(current) > 0.02) {
      const direction = Math.sign(current);
      arrow(ctx, 830 - (direction * currentLength) / 2, 154, 830 + (direction * currentLength) / 2, 154, "#d97706", 4);
    } else line(ctx, 824, 154, 836, 154, "#64748b", 2);
    text(ctx, `i = ${current >= 0 ? "+" : ""}${current.toFixed(2)} A`, 830, 178, 11, "#b45309", "center", "700");
    const fieldLength = 18 + 205 * Math.min(Math.abs(magneticField) / 0.5, 1);
    if (Math.abs(magneticField) > 0.002) {
      const direction = Math.sign(magneticField);
      arrow(ctx, 840 - (direction * fieldLength) / 2, coilY, 840 + (direction * fieldLength) / 2, coilY, "#16a34a", 4);
      const leftPole = direction > 0 ? "S" : "N";
      const rightPole = direction > 0 ? "N" : "S";
      text(ctx, leftPole, coilLeft - 25, coilY, 18, leftPole === "N" ? "#dc2626" : "#2563eb", "center", "700");
      text(ctx, rightPole, coilRight + 25, coilY, 18, rightPole === "N" ? "#dc2626" : "#2563eb", "center", "700");
    } else line(ctx, 832, coilY, 848, coilY, "#64748b", 2);
    text(ctx, `B = ${magneticField >= 0 ? "+" : ""}${magneticField.toFixed(3)} T`, 840, 384, 12, "#15803d", "center", "700");
    drawCoilEsb(ctx, voltage, current);
  }

  function drawCoilEsb(ctx, voltage, current) {
    card(ctx, 655, 493, 370, 153, "Simplified equivalent circuit");
    text(ctx, "Battery - R - L", 1013, 510, 9, "#475569", "right");
    const top = 548;
    const bottom = 603;
    line(ctx, 700, top, 765, top); line(ctx, 824, top, 856, top); line(ctx, 937, top, 978, top);
    line(ctx, 978, top, 978, bottom); line(ctx, 978, bottom, 700, bottom); line(ctx, 700, bottom, 700, 574); line(ctx, 700, 556, 700, top);
    const positiveAtTop = voltage >= 0;
    const topColour = positiveAtTop ? "#dc2626" : "#2563eb";
    const bottomColour = positiveAtTop ? "#2563eb" : "#dc2626";
    line(ctx, 684, 556, 716, 556, topColour, 3); line(ctx, 690, 574, 710, 574, bottomColour, 3);
    text(ctx, positiveAtTop ? "+" : "-", 672, 556, 12, topColour, "center", "700");
    text(ctx, positiveAtTop ? "-" : "+", 672, 574, 12, bottomColour, "center", "700");
    text(ctx, `U_bat ${voltage >= 0 ? "+" : ""}${voltage.toFixed(1)} V`, 700, 622, 8, "#0f172a", "center", "700");
    const zigzag = [[765, top], [773, top - 8], [782, top + 8], [791, top - 8], [800, top + 8], [809, top - 8], [816, top + 8], [824, top]];
    ctx.strokeStyle = "#475569"; ctx.lineWidth = 2; ctx.beginPath(); zigzag.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke();
    text(ctx, "R", 794, 531, 10, "#475569", "center", "700");
    ctx.strokeStyle = "#7c3aed"; ctx.lineWidth = 2; ctx.beginPath();
    for (let point = 0; point <= 32; point += 1) { const progress = point / 32; const x = 856 + 81 * progress; const y = top - 11 * Math.sin(8 * Math.PI * progress); if (point) ctx.lineTo(x, y); else ctx.moveTo(x, y); } ctx.stroke();
    text(ctx, "L", 897, 531, 10, "#6d28d9", "center", "700");
    if (Math.abs(current) > 0.02) { const length = 10 + 36 * Math.min(Math.abs(current) / 10, 1); const direction = Math.sign(current); arrow(ctx, 736 - direction * length / 2, 531, 736 + direction * length / 2, 531, "#d97706", 3); }
    text(ctx, `i ${current >= 0 ? "+" : ""}${current.toFixed(2)} A`, 736, 517, 8, "#b45309", "center", "700");
  }

  function testField() {
    let x = 0; let y = 0;
    state.test.active.forEach((active, index) => { if (active) { x += Math.cos(phaseAngles[index]); y += Math.sin(phaseAngles[index]); } });
    const strength = Math.hypot(x, y);
    return { angle: strength > 1e-9 ? Math.atan2(y, x) : 0, strength };
  }

  function drawTest() {
    const ctx = prepare(canvases.test);
    const field = testField();
    const activeNames = phaseNames.filter((_, index) => state.test.active[index]).join(", ") || "none";
    const result = field.strength > 1e-9 ? `S_res at ${degrees(field.angle).toFixed(0)} degrees` : "no resultant stator pole";
    const rpm = state.test.speed * 60 / (2 * Math.PI);
    text(ctx, "Motor experiment: switch stator poles manually", 25, 25, 18, "#0f172a", "left", "700");
    text(ctx, `Active: ${activeNames}   |   ${result}   |   T_em = ${state.test.torque >= 0 ? "+" : ""}${state.test.torque.toFixed(2)}   |   n = ${rpm >= 0 ? "+" : ""}${rpm.toFixed(1)} rpm`, 25, 55, 12, field.strength ? "#166534" : "#475569", "left", "700");
    drawTestPhasePanel(ctx); drawTestFieldPanel(ctx, field); drawTestRotorPanel(ctx); drawTestMachine(ctx, field); drawTestEsb(ctx);
  }

  function drawTestPhasePanel(ctx) {
    card(ctx, 25, 95, 610, 195, "Phases: 0 V or +Udc"); text(ctx, "Q / S / E", 623, 112, 9, "#475569", "right", "700");
    const keys = ["E", "Q", "S"];
    [0, 1, 2].forEach(index => {
      const y = 150 + index * 42; const active = state.test.active[index];
      ctx.fillStyle = active ? "#dcfce7" : "#f8fafc"; ctx.strokeStyle = active ? "#16a34a" : "#94a3b8"; ctx.lineWidth = 2; ctx.fillRect(49, y - 14, 94, 28); ctx.strokeRect(49, y - 14, 94, 28);
      text(ctx, `${keys[index]}: ${phaseNames[index]}`, 96, y, 10, "#0f172a", "center", "700");
      line(ctx, 165, y, 480, y, active ? "#16a34a" : "#cbd5e1", active ? 5 : 2);
      text(ctx, active ? "+Udc  ->  S" : "0 V", 525, y, 11, active ? "#166534" : "#475569", "center", "700");
      text(ctx, `${index * 120}°`, 610, y, 9, "#64748b", "right");
    });
  }

  function drawTestFieldPanel(ctx, field) {
    card(ctx, 25, 310, 610, 195, "Resultant stator pole");
    const cx = 170; const cy = 410; ctx.strokeStyle = "#94a3b8"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, 68, 0, 2 * Math.PI); ctx.stroke();
    [0, 120, 240].forEach(angle => text(ctx, `${angle}°`, cx + 68 * Math.cos(angle * Math.PI / 180), cy - 68 * Math.sin(angle * Math.PI / 180), 8, "#64748b", "center"));
    if (field.strength > 1e-9) {
      arrow(ctx, cx, cy, cx + 68 * Math.cos(field.angle), cy - 68 * Math.sin(field.angle), "#16a34a", 5);
      text(ctx, `S_res = ${degrees(field.angle).toFixed(0)}°`, 290, 382, 14, "#166534", "left", "700");
      text(ctx, "T_em comes from the angle error to the rotor north pole.", 290, 414, 10, "#475569");
    } else { text(ctx, "No resultant pole", 290, 393, 14, "#64748b", "left", "700"); text(ctx, "With U + V + W, the fields cancel.", 290, 423, 10, "#475569"); }
    text(ctx, `Field strength (normalized): ${field.strength.toFixed(1)}`, 290, 460, 10, "#475569", "left", "700");
  }

  function drawTestRotorPanel(ctx) {
    card(ctx, 25, 525, 610, 90, "Rotor angle");
    text(ctx, `${degrees(state.test.theta).toFixed(1)} degrees`, 623, 542, 9, "#b45309", "right");
    const left = 80; const right = 617; const top = 555; const bottom = 602; line(ctx, left, bottom, right, bottom, "#94a3b8", 1, [3, 3]); text(ctx, "360", 33, top, 8, "#64748b"); text(ctx, "0", 43, bottom, 8, "#64748b");
    ctx.strokeStyle = "#d97706"; ctx.lineWidth = 2; ctx.beginPath(); state.test.history.forEach((angle, index) => { const x = left + (right - left) * index / Math.max(1, state.test.history.length - 1); const y = bottom - (bottom - top) * degrees(angle) / 360; if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.stroke();
  }

  function drawTestMachine(ctx, field) {
    const cx = 830; const cy = 280; const radius = 182; text(ctx, "Three-phase machine", cx, 66, 12, "#0f172a", "center", "700");
    ctx.fillStyle = "#e2e8f0"; ctx.strokeStyle = "#334155"; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(cx, cy, radius, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    phaseAngles.forEach((angle, index) => {
      const active = state.test.active[index]; const rx = Math.cos(angle); const ry = -Math.sin(angle); const tx = -Math.sin(angle); const ty = -Math.cos(angle); const px = cx + 150 * rx; const py = cy + 150 * ry;
      ctx.strokeStyle = active ? "#16a34a" : "#64748b"; ctx.lineWidth = 3; ctx.beginPath();
      for (let point = 0; point <= 24; point += 1) { const progress = point / 24; const along = -23 + 46 * progress; const offset = 7 * Math.sin(6 * Math.PI * progress); const x = px + along * tx + offset * rx; const y = py + along * ty + offset * ry; if (point) ctx.lineTo(x, y); else ctx.moveTo(x, y); } ctx.stroke();
      text(ctx, `${phaseNames[index]} ${active ? "+Udc" : "0 V"}`, cx + 174 * rx, cy + 174 * ry, 8, active ? "#166534" : "#64748b", "center", "700");
      if (active) { const nx = cx + 118 * rx; const ny = cy + 118 * ry; ctx.fillStyle = "#2563eb"; ctx.strokeStyle = "#1d4ed8"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(nx, ny, 16, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); text(ctx, "S", nx, ny, 12, "#ffffff", "center", "700"); }
    });
    if (field.strength > 1e-9) arrow(ctx, cx, cy, cx + 142 * Math.cos(field.angle), cy - 142 * Math.sin(field.angle), "#16a34a", 4);
    drawRotor(ctx, cx, cy, state.test.theta, 67, 29, 78);
    text(ctx, "T_em from angle error moves the rotor", cx, 468, 9, "#475569", "center");
  }

  function drawRotor(ctx, cx, cy, theta, halfLength, halfWidth, radius) {
    const dx = Math.cos(theta); const dy = -Math.sin(theta); const px = -dy; const py = dx;
    const corners = [[-1, -1], [-1, 1], [1, 1], [1, -1]].map(([longitudinal, lateral]) => [cx + longitudinal * halfLength * dx + lateral * halfWidth * px, cy + longitudinal * halfLength * dy + lateral * halfWidth * py]);
    ctx.fillStyle = "#fef3c7"; ctx.strokeStyle = "#334155"; ctx.lineWidth = 2; ctx.beginPath(); corners.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.closePath(); ctx.fill(); ctx.stroke();
    line(ctx, cx, cy, cx + halfLength * dx, cy + halfLength * dy, "#dc2626", halfWidth * 2); line(ctx, cx, cy, cx - halfLength * dx, cy - halfLength * dy, "#2563eb", halfWidth * 2);
    text(ctx, "N", cx + 40 * dx, cy + 40 * dy, 13, "#ffffff", "center", "700"); text(ctx, "S", cx - 40 * dx, cy - 40 * dy, 13, "#ffffff", "center", "700");
    ctx.strokeStyle = "#334155"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, radius, 0, 2 * Math.PI); ctx.stroke();
    arrow(ctx, cx, cy, cx + 110 * dx, cy + 110 * dy, "#000000", 2); text(ctx, "RLG 0°", cx + 138 * dx, cy + 138 * dy, 9, "#000000", "center", "700");
  }

  function drawTestEsb(ctx) {
    card(ctx, 655, 493, 370, 153, "Simplified inverter"); text(ctx, "0 V / +Udc only", 1013, 510, 9, "#475569", "right");
    line(ctx, 694, 527, 895, 527, "#dc2626", 2); line(ctx, 694, 582, 895, 582, "#64748b", 2); text(ctx, "+Udc", 680, 527, 8, "#dc2626", "right", "700"); text(ctx, "0 V", 680, 582, 8, "#475569", "right", "700");
    [746, 803, 860].forEach((x, index) => { const active = state.test.active[index]; const colour = phaseColours[index]; line(ctx, x, 527, x, 541, "#475569", 1); ctx.fillStyle = active ? "#16a34a" : "#ffffff"; ctx.strokeStyle = active ? "#166534" : "#64748b"; ctx.lineWidth = 2; ctx.fillRect(x - 17, 541, 34, 16); ctx.strokeRect(x - 17, 541, 34, 16); text(ctx, active ? "+Udc" : "0 V", x, 549, 7, active ? "#ffffff" : "#475569", "center", "700"); line(ctx, x, 557, x, 567, colour, 2); line(ctx, x, 567, 935, 567, colour, 2); text(ctx, phaseNames[index], x, 603, 9, colour, "center", "700"); });
    ctx.fillStyle = "#f8fafc"; ctx.strokeStyle = "#334155"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(970, 565, 30, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); text(ctx, "M", 970, 560, 14, "#0f172a", "center", "700"); text(ctx, "PSM", 970, 577, 8, "#475569", "center");
  }

  function drawMotor() {
    const ctx = prepare(canvases.motor); const motor = state.motor; const currentAngle = motor.theta + motor.phase * Math.PI / 180; const iq = motor.amplitude * Math.sin(motor.phase * Math.PI / 180); const torque = 0.12 * iq; const voltage = 0.5 * motor.amplitude;
    text(ctx, "Three-phase system with rotating magnetic field", 25, 25, 18, "#0f172a", "left", "700"); text(ctx, `Electromagnetic torque T_e = ${torque >= 0 ? "+" : ""}${torque.toFixed(2)} Nm`, 25, 55, 13, "#166534", "left", "700");
    drawMotorWave(ctx, 25, 95, "Phase currents iabc", "A", motor.amplitude, 100, currentAngle); drawMotorWave(ctx, 25, 310, "Phase voltages uabc", "V", voltage, 50, currentAngle); drawMotorRotorPanel(ctx);
    drawMotorMachine(ctx, currentAngle); drawMotorEsb(ctx, voltage, currentAngle);
  }

  function drawMotorWave(ctx, x, y, title, unit, amplitude, limit, angle) {
    card(ctx, x, y, 610, 195, title); const left = x + 55; const right = x + 592; const top = y + 42; const bottom = y + 167; const mid = (top + bottom) / 2; line(ctx, left, mid, right, mid, "#94a3b8", 1, [3, 3]); text(ctx, `+${limit}`, x + 9, top, 9, "#64748b"); text(ctx, `-${limit}`, x + 9, bottom, 9, "#64748b"); text(ctx, unit, right, y + 186, 9, "#64748b", "right");
    phaseAngles.forEach((offset, phase) => { const colour = phaseColours[phase]; ctx.strokeStyle = colour; ctx.lineWidth = 2; ctx.beginPath(); for (let px = left; px <= right; px += 3) { const historyAngle = angle - 2 * Math.PI * (right - px) / (right - left); const py = mid - (bottom - top) * .43 * (amplitude / limit) * Math.cos(historyAngle + offset); if (px === left) ctx.moveTo(px, py); else ctx.lineTo(px, py); } ctx.stroke(); const instantaneous = amplitude * Math.cos(angle + offset); const markerY = mid - (bottom - top) * .43 * (amplitude / limit) * Math.cos(angle + offset); ctx.fillStyle = colour; ctx.beginPath(); ctx.arc(right, markerY, 4, 0, 2 * Math.PI); ctx.fill(); text(ctx, `${"abc"[phase]}: ${instantaneous >= 0 ? "+" : ""}${instantaneous.toFixed(1)} ${unit}`, right - 2, top + 12 + phase * 18, 9, colour, "right"); });
  }

  function drawMotorRotorPanel(ctx) {
    card(ctx, 25, 525, 610, 90, "Rotor angle");
    const degreesValue = degrees(state.motor.theta);
    text(ctx, `${degreesValue.toFixed(1)} degrees`, 623, 542, 9, "#b45309", "right");
    const left = 80; const right = 617; const top = 555; const bottom = 602;
    line(ctx, left, bottom, right, bottom, "#94a3b8", 1, [3, 3]);
    text(ctx, "360", 33, top, 8, "#64748b"); text(ctx, "0", 43, bottom, 8, "#64748b");
    ctx.strokeStyle = "#d97706"; ctx.lineWidth = 2; ctx.beginPath();
    for (let x = left; x <= right; x += 3) {
      const historyAngle = state.motor.theta - 2 * Math.PI * (right - x) / (right - left);
      const y = bottom - (bottom - top) * degrees(historyAngle) / 360;
      if (x === left) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    const markerY = bottom - (bottom - top) * degreesValue / 360;
    ctx.fillStyle = "#d97706"; ctx.beginPath(); ctx.arc(right, markerY, 4, 0, 2 * Math.PI); ctx.fill();
  }

  function drawMotorMachine(ctx, currentAngle) {
    const cx = 830; const cy = 280; const outerRadius = 205; const rotorRadius = 84;
    text(ctx, "Three-phase machine", cx, 66, 12, "#0f172a", "center", "700");
    ctx.fillStyle = "#e2e8f0"; ctx.strokeStyle = "#334155"; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(cx, cy, outerRadius, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();

    phaseAngles.forEach((angle, index) => {
      const radialX = Math.cos(angle); const radialY = -Math.sin(angle);
      const tangentX = -Math.sin(angle); const tangentY = -Math.cos(angle);
      const coilX = cx + 157 * radialX; const coilY = cy + 157 * radialY;
      const current = state.motor.amplitude * Math.cos(currentAngle + phaseAngles[index]);
      const colour = phaseColours[index];

      ctx.strokeStyle = colour; ctx.lineWidth = 3; ctx.beginPath();
      for (let point = 0; point <= 24; point += 1) {
        const progress = point / 24; const along = -24 + 48 * progress; const offset = 8 * Math.sin(6 * Math.PI * progress);
        const x = coilX + along * tangentX + offset * radialX; const y = coilY + along * tangentY + offset * radialY;
        if (point) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      }
      ctx.stroke();
      line(ctx, coilX - 33 * tangentX, coilY - 33 * tangentY, coilX - 24 * tangentX, coilY - 24 * tangentY, colour, 3);
      line(ctx, coilX + 24 * tangentX, coilY + 24 * tangentY, coilX + 33 * tangentX, coilY + 33 * tangentY, colour, 3);

      const relativeCurrent = Math.abs(current) / Math.max(state.motor.amplitude, 1);
      const arrowLength = 10 + 46 * relativeCurrent; const arrowX = coilX - 25 * radialX; const arrowY = coilY - 25 * radialY;
      if (Math.abs(current) < 0.05) line(ctx, arrowX - 5 * tangentX, arrowY - 5 * tangentY, arrowX + 5 * tangentX, arrowY + 5 * tangentY, "#64748b", 2);
      else {
        const direction = Math.sign(current);
        arrow(ctx, arrowX - direction * arrowLength / 2 * tangentX, arrowY - direction * arrowLength / 2 * tangentY, arrowX + direction * arrowLength / 2 * tangentX, arrowY + direction * arrowLength / 2 * tangentY, colour, 3);
      }
      text(ctx, `${phaseNames[index]}  ${current >= 0 ? "+" : ""}${current.toFixed(0)} A`, cx + 104 * radialX, cy + 104 * radialY, 9, colour, "center", "700");
    });

    const fieldX = Math.cos(currentAngle); const fieldY = -Math.sin(currentAngle); const fieldPerpX = -fieldY; const fieldPerpY = fieldX;
    line(ctx, cx - 148 * fieldX, cy - 148 * fieldY, cx + 148 * fieldX, cy + 148 * fieldY, "#94a3b8", 1, [4, 4]);
    const fieldStrength = Math.min(state.motor.amplitude / 100, 1);
    if (fieldStrength > 0.001) {
      const halfRadial = 6 + 18 * fieldStrength; const halfLateral = 8 + 34 * fieldStrength;
      [["N", -1, "#dc2626"], ["S", 1, "#2563eb"]].forEach(([name, sign, colour]) => {
        const poleX = cx + sign * 122 * fieldX; const poleY = cy + sign * 122 * fieldY;
        const corners = [[-1, -1], [-1, 1], [1, 1], [1, -1]].map(([radialSign, lateralSign]) => [poleX + radialSign * halfRadial * fieldX + lateralSign * halfLateral * fieldPerpX, poleY + radialSign * halfRadial * fieldY + lateralSign * halfLateral * fieldPerpY]);
        ctx.fillStyle = colour; ctx.strokeStyle = "#334155"; ctx.lineWidth = 1; ctx.beginPath(); corners.forEach(([x, y], point) => point ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.closePath(); ctx.fill(); ctx.stroke();
        text(ctx, name, poleX, poleY, Math.max(7, Math.round(8 + 6 * fieldStrength)), "#ffffff", "center", "700");
      });
    }
    drawRotor(ctx, cx, cy, state.motor.theta, 72, 32, rotorRadius);
  }

  function drawMotorEsb(ctx, voltage, currentAngle) {
    card(ctx, 655, 493, 370, 153, "Simplified equivalent circuit");
    text(ctx, "green: HS | red: LS", 1013, 510, 8, "#475569", "right");
    ctx.fillStyle = "#eff6ff"; ctx.strokeStyle = "#2563eb"; ctx.lineWidth = 2; ctx.fillRect(670, 522, 68, 53); ctx.strokeRect(670, 522, 68, 53);
    text(ctx, "µC", 704, 540, 12, "#1d4ed8", "center", "700"); text(ctx, "PWM", 704, 555, 8, "#1d4ed8", "center", "700");
    ctx.fillStyle = "#f5f3ff"; ctx.strokeStyle = "#7c3aed"; ctx.lineWidth = 2; ctx.fillRect(775, 515, 96, 66); ctx.strokeRect(775, 515, 96, 66);
    arrow(ctx, 738, 548, 775, 548, "#2563eb", 2); text(ctx, "PWM", 756, 539, 7, "#2563eb", "center");
    const batteryX = 758; const plusRail = 521; const minusRail = 576;
    line(ctx, batteryX, plusRail, 863, plusRail, "#dc2626", 2); line(ctx, batteryX, minusRail, 863, minusRail, "#2563eb", 2);
    line(ctx, batteryX, plusRail, batteryX, 533, "#dc2626", 2); line(ctx, batteryX - 14, 533, batteryX + 14, 533, "#dc2626", 3); line(ctx, batteryX - 8, 563, batteryX + 8, 563, "#2563eb", 3); line(ctx, batteryX, 563, batteryX, minusRail, "#2563eb", 2);
    text(ctx, "+", batteryX - 20, 533, 10, "#dc2626", "center", "700"); text(ctx, "-", batteryX - 20, 563, 12, "#2563eb", "center", "700");
    const sensors = [];
    [799, 827, 855].forEach((legX, index) => {
      const phaseY = 535 + index * 15; const colour = phaseColours[index]; const phaseVoltage = voltage * Math.cos(currentAngle + phaseAngles[index]); const highSide = phaseVoltage >= 0;
      const hsY = phaseY - 6; const lsY = phaseY + 6;
      line(ctx, legX, plusRail, legX, hsY - 4, "#475569", 1); ctx.fillStyle = highSide ? "#16a34a" : "#ffffff"; ctx.strokeStyle = "#166534"; ctx.lineWidth = 1; ctx.fillRect(legX - 8, hsY - 4, 16, 8); ctx.strokeRect(legX - 8, hsY - 4, 16, 8); text(ctx, "HS", legX, hsY, 6, highSide ? "#ffffff" : "#166534", "center", "700");
      line(ctx, legX, hsY + 4, legX, phaseY, "#475569", 1); ctx.fillStyle = colour; ctx.beginPath(); ctx.arc(legX, phaseY, 2, 0, 2 * Math.PI); ctx.fill(); line(ctx, legX, phaseY, legX, lsY - 4, "#475569", 1);
      ctx.fillStyle = !highSide ? "#dc2626" : "#ffffff"; ctx.strokeStyle = "#991b1b"; ctx.fillRect(legX - 8, lsY - 4, 16, 8); ctx.strokeRect(legX - 8, lsY - 4, 16, 8); text(ctx, "LS", legX, lsY, 6, !highSide ? "#ffffff" : "#991b1b", "center", "700"); line(ctx, legX, lsY + 4, legX, minusRail, "#475569", 1);
      line(ctx, legX, phaseY, 954, phaseY, colour, 2); text(ctx, phaseNames[index], 877, phaseY - 8, 7, colour, "center", "700");
      ctx.fillStyle = "#ffffff"; ctx.strokeStyle = colour; ctx.lineWidth = 1; ctx.fillRect(886, phaseY - 7, 35, 14); ctx.strokeRect(886, phaseY - 7, 35, 14); text(ctx, `I_${phaseNames[index].toLowerCase()}`, 903.5, phaseY, 7, colour, "center", "700"); sensors.push([903.5, phaseY]);
    });
    const motorX = 982; const motorY = 550; ctx.fillStyle = "#f8fafc"; ctx.strokeStyle = "#334155"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(motorX, motorY, 27, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); text(ctx, "M", motorX, motorY - 5, 14, "#0f172a", "center", "700"); text(ctx, "3~", motorX, motorY + 12, 8, "#475569", "center");
    const feedbackY = 594; sensors.forEach(([x, y]) => line(ctx, x, y + 7, x, feedbackY, "#64748b", 1, [2, 2])); line(ctx, sensors[0][0], feedbackY, 704, feedbackY, "#64748b", 1, [2, 2]); arrow(ctx, 704, feedbackY, 704, 566, "#64748b", 1); text(ctx, "iabc", 790, feedbackY + 8, 8, "#475569", "center");
    ctx.fillStyle = "#f9fafb"; ctx.strokeStyle = "#111827"; ctx.lineWidth = 2; ctx.fillRect(916, 616, 58, 24); ctx.strokeRect(916, 616, 58, 24); text(ctx, "RLG", 945, 628, 9, "#111827", "center", "700"); line(ctx, motorX, motorY + 27, motorX, 616, "#111827", 1, [3, 2]); line(ctx, 916, 628, 720, 628, "#111827", 1, [3, 2]); arrow(ctx, 720, 628, 720, 556, "#111827", 1); text(ctx, "angle", 815, 619, 8, "#111827", "center");
  }

  function updateTestButtons() { document.querySelectorAll(".phase-button").forEach(button => { const index = Number(button.dataset.phase); const active = state.test.active[index]; const key = index === 1 ? "Q" : index === 2 ? "S" : "E"; button.textContent = `${key}: ${phaseNames[index]} = ${active ? "+Udc" : "0 V"}`; button.classList.toggle("is-active", active); }); }
  function setTestPhase(index, active) { if (state.test.active[index] === active) return; state.test.active[index] = active; updateTestButtons(); }

  function switchTab(tab) { state.activeTab = tab; document.querySelectorAll(".lab-tab").forEach(button => { const active = button.dataset.tab === tab; button.classList.toggle("is-active", active); button.setAttribute("aria-selected", String(active)); }); document.querySelectorAll(".lab-panel").forEach(panel => { panel.hidden = panel.dataset.panel !== tab; }); }

  document.querySelectorAll(".lab-tab").forEach(button => button.addEventListener("click", () => switchTab(button.dataset.tab)));
  document.querySelectorAll(".phase-button").forEach(button => { const index = Number(button.dataset.phase); button.addEventListener("pointerdown", event => { event.preventDefault(); setTestPhase(index, true); }); ["pointerup", "pointercancel", "pointerleave"].forEach(type => button.addEventListener(type, () => setTestPhase(index, false))); });
  document.getElementById("test-reset").addEventListener("click", () => { state.test.theta = 0; state.test.speed = 0; state.test.torque = 0; state.test.history = [0]; });
  document.getElementById("coil-reset").addEventListener("click", () => { state.coil.voltage = 0; document.getElementById("coil-voltage").value = "0"; document.getElementById("coil-voltage-value").textContent = "+0.0 V"; });
  document.getElementById("coil-voltage").addEventListener("input", event => { state.coil.voltage = Number(event.target.value); document.getElementById("coil-voltage-value").textContent = `${state.coil.voltage >= 0 ? "+" : ""}${state.coil.voltage.toFixed(1)} V`; });
  [["speed", "motor-speed", "motor-speed-value", "rpm"], ["amplitude", "motor-amplitude", "motor-amplitude-value", "A"], ["phase", "motor-phase", "motor-phase-value", "°"]].forEach(([name, inputId, outputId, unit]) => document.getElementById(inputId).addEventListener("input", event => { state.motor[name] = Number(event.target.value); const value = state.motor[name]; document.getElementById(outputId).textContent = `${name === "phase" && value >= 0 ? "+" : ""}${value} ${unit}`; }));
  const keyPhase = { q: 1, s: 2, e: 0 };
  window.addEventListener("keydown", event => { const key = event.key.toLowerCase(); if (state.activeTab === "test" && key in keyPhase && !event.repeat) { event.preventDefault(); setTestPhase(keyPhase[key], true); } });
  window.addEventListener("keyup", event => { const key = event.key.toLowerCase(); if (key in keyPhase) { setTestPhase(keyPhase[key], false); } });
  window.addEventListener("blur", () => state.test.active.forEach((_, index) => setTestPhase(index, false)));

  let previousTime = performance.now();
  function animate(time) {
    const dt = Math.min(0.05, Math.max(0, (time - previousTime) / 1000)); previousTime = time;
    const coil = state.coil; coil.current += ((coil.voltage - coil.current) / 0.25) * dt; coil.history.push({ u: coil.voltage, i: coil.current, b: coil.current * 0.05 }); if (coil.history.length > 240) coil.history.splice(0, coil.history.length - 240);
    const field = testField(); const rotorNorth = state.test.theta; const error = wrapAngle(field.angle - rotorNorth); state.test.torque = 2.2 * field.strength * Math.sin(error); const acceleration = (state.test.torque - 1.1 * state.test.speed) / 0.35; state.test.speed += acceleration * dt; state.test.theta = (state.test.theta + state.test.speed * dt) % (2 * Math.PI); if (state.test.theta < 0) state.test.theta += 2 * Math.PI; state.test.history.push(state.test.theta); if (state.test.history.length > 240) state.test.history.splice(0, state.test.history.length - 240);
    state.motor.theta = (state.motor.theta + 2 * Math.PI * Math.min(state.motor.speed / 60, 8) * dt) % (2 * Math.PI);
    if (state.activeTab === "coil") drawCoil(); else if (state.activeTab === "test") drawTest(); else drawMotor();
    requestAnimationFrame(animate);
  }

  updateTestButtons();
  requestAnimationFrame(animate);
})();
