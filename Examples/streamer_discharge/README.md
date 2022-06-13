**Case study II - Streamer discharge model**

In the second case study, the positive streamer in air at atmospheric pressure is modelled. The number density profiles and electric potential are calculated and used for the code verification by benchmarking, where results from Bagheri et al. Plasma Sources Sci. Technol. 27 (2018) 095002 are used as the benchmark. The model consists of Poisson's and the balance equations for electrons and positive ions, which are solved in axisymmetric geometry, assuming the local field approximation. The simulation conditions are defined in the model as parameters (pressure $p_0 = 760 \, Torr$, working voltage $U_w = 18750 \, V$, gas temperature $T_g = 300 \, K$). The problem is defined mostly using FEDM functions:

- the particle species list are read from the input file stored in `file_input` folder using `read_speclist()` function, which returns the number of species, their names and corresponding file names
- the particle properties are read from the input files stored in `file_input` folder using `read_particle_properties()`, which returns the mass and the charge of the particles
- the transport coefficients are read from the input files using `reading_transport_coefficients()`, noting that since the coefficients are described as the functions in this example, they should be expressed as the python code in the input files
- the number of equations are determined based on the type of approximation used in the model (LFA or LMEA) by `approximation_type()` function
- the output files are created using `output_files()` function
- the boundaries are marked with boundary marking function `Marking_boundaries()`, returning the mesh function that is used for redefinition of the surface integral measure `ds`
- the logging of the particle properties, the simulation conditions, the mesh statistics and the time is done using `log()` function
- the mixed function space is defined using `Mixed_element_list()` which creates the list of function elements of the desired order
- the list of functions (or trial and test functions for segregated approach) for given number of particle species is done using `Function_definition()`
- the variational form of Poisson's equation is defined using `weak_form_Poisson_equation()`
- the fluxes are defined using `Flux()` of `Flux_log()` (for the logarithmic formulation)
- the variational forms of the balance equations for the particle species are defined using `weak_form_balance_equation()` or `weak_form_balance_equation_log_representation()` (for the logarithmic formulation)
- the boundary conditions are defined using `Boundary_flux()` function
- solving of the equations in adaptive manner requires two functions, `adaptive_solver()` and `adaptive_timestep()`
- the results are written into file using `file_output()`

The code can be executed by running the following in the terminal:

```bash
python3 fedm-streamer.py
```

or in parallel using MPI

```bash
mpirun -np 8 python3 fedm-streamer.py
```
