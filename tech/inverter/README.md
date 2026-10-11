# Dreiphasen-Inverter: einfache GUI

Start im Repository-Hauptordner:

```powershell
py tech/inverter/inverter_visualizer.py
```

Die Anwendung benötigt nur Python mit der Standardbibliothek (`tkinter`).

Sie startet maximiert mit `0 rpm`. Die Lehrgrafik wird automatisch an die verfügbare Fensterfläche skaliert.

## Was die Visualisierung zeigt

- `iabc`: drei um 120° versetzte Phasenströme
- `uabc`: drei um 120° versetzte Phasenspannungen
- Rotorwinkelverlauf unter den Strom- und Spannungsdiagrammen
- ein vereinfachtes Dreiphasensystem mit drei repräsentativen Wicklungen A/B/C und einem rotierenden N/S-Magneten
- schwarze, mit dem Rotor mitdrehende Referenzachse `RLG 0°`
- Drehzahl-, Stromamplituden- und Stromwinkel-Schieberegler
- numerische Eingabefelder direkt neben jedem Schieberegler (Enter oder Feld verlassen zum Uebernehmen)
- ein vereinfachtes ESB mit µC, 3-Phasen-Inverter, drei Stromsensoren, Motor und RLG-Rueckfuehrung
- interaktive Halbbruecken mit Batteriesymbol und Zwischenkreis-Schienen: HS wird bei positiver, LS bei negativer Phasenspannung gruen bzw. rot hervorgehoben

Das ist ein Anschauungsmodell einer oberflaechenmontierten Permanentmagnet-Synchronmaschine (SPMSM), kein Motorregler. Der Stromamplituden-Schieberegler setzt den Spitzenwert des dreiphasigen Stroms. Der Stromwinkel `phi` dreht das daraus entstehende Statorfeld relativ zum Rotor.

Als zentraler Output wird das elektromagnetische Moment gezeigt. Es wird aus dem einfachen SPMSM-Zusammenhang berechnet:

`T_e = 3/2 * p * psi_PM * i_q`

Die feste Annahme ist ein Polpaar (`p = 1`) und `psi_PM = 0,08 Wb`. Deshalb sind mechanischer und elektrischer Rotorwinkel in dieser Darstellung gleich. Dabei gilt `i_q = I_spitze * sin(phi)`: bei `phi = 0 Grad` entsteht kein Moment, bei `+90 Grad` das maximale positive und bei `-90 Grad` das maximale negative Moment.

Die Statorfeld-Amplitude wird ebenso bewusst einfach und linear aus der Stromamplitude gebildet: `B_stator = 0,01 T/A * I_spitze`. Die N-/S-Polschuhe im Luftspalt wachsen und schrumpfen mit der eingestellten Stromamplitude.

Die Drehzahl ist bewusst auf den Bereich `0 bis 100 rpm` begrenzt, damit die langsame Rotation und die Phasenlage gut sichtbar bleiben.

Die Winkel folgen der mathematischen Standardkonvention: `theta = 0 Grad` liegt auf der positiven x-Achse (rechts); ein steigender Winkel dreht gegen den Uhrzeigersinn. Die schwarze Referenz `RLG 0°` liegt daher bei der Startposition rechts vom Rotor.

Die drei Wicklungen repraesentieren die Phasen A, B und C. Jede Phase wird bewusst nur als einfaches Spulensymbol mit zwei kurzen Anschluessen dargestellt. Daneben liegt jeweils ein einzelner, klarer Stromzeiger als Referenz fuer `i_a`, `i_b` bzw. `i_c`; seine Laenge ist proportional zum Betrag und seine Richtung zum Vorzeichen des Phasenstroms.

Im vereinfachten ESB ist jede PMSM-Phase als Reihenschaltung aus Statorwiderstand `R_s`, Statorinduktivitaet `L_s` und drehzahlabhaengiger Gegen-EMK `e` gekennzeichnet.

Die Spannungsamplitude ist fuer diese einfache Darstellung direkt an die Stromamplitude gekoppelt: `u_spitze = 0,5 V/A * I_spitze`. Bei 100 A beträgt die dargestellte Phasenspannung daher 50 V.
