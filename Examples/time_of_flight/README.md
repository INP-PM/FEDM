**Case study I - Time of flight model**

In the first case study, the modelling of the time-of-flight experiment is presented. The spatiotemporal evolution of the electron number density is described by the drift-diffusion equation, for which the analytic solution exists. So, by calculating the time and mesh order-of-accuracy, verification of the code is rigorously carried out. To simplify the problem, the simplest plane parallel geometry with constant electric field between the electrodes at the atmospheric pressure is assumed (pressure $p_0 = 760$ Torr, gas temperature $T_g = 300$ K). The initial time step is  $t=2.5$ ns  and calculations are carried out until $T = 3$ ns.  The drift velocity, diffusion coefficient and effective ionization coefficient are pre-calculated and imported in the model as the parameters. To obtain the order of accuracy, the relative difference between numerical and analytical solution is calculated for different time-step and mesh sizes. Since only one equation is solved, the native FEniCS functions are mostly used. For verification purpose the structured mesh is required, so built-in mesh-generating function is used. The FEDM function are used for:

- creating the output files using the `output_files()` function
- printing the number of elements, minimal and maximal cell diameter using the `mesh_statistics()` function
- logging the particle properties, the simulation conditions, the mesh statistics and the time using the `log()` function
- definition of the variational formulation using the `weak_form_balance_equation_log_representation()` function

The code can be executed by running the following in the terminal:

```bash
python3 fedm-tof.py
```

or in parallel using MPI

```bash
mpirun -np 8 python3 fedm-tof.py
```
