# Interactive Inverter Lab

Browser version: [robfil.github.io/tech/inverter/](https://robfil.github.io/tech/inverter/)

Three compact learning views:

- **Coil** - adjustable DC voltage, RL current response, and magnetic field.
- **Motor experiment** - hold `Q`, `S`, or `E` to apply `+Udc` to phases `V`, `W`, or `U`. Active phases create stator south poles; the rotor north pole follows the resultant field.
- **Motor** - simplified SPMSM currents, voltages, rotor angle, and electromagnetic torque.

The browser version is static HTML, CSS, and JavaScript, so it runs directly on GitHub Pages. `inverter_visualizer.py` is the local Tkinter version.
