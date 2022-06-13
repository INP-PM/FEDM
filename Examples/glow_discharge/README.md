**Case study III - Glow discharge model**

In the last case study, the glow discharge in argon at low pressure is modelled. The aim of this example is to demonstrate the use of the code for the case when multiple particle species with many processes are considered in the model and when the local mean energy approximation is assumed. The model describes spatiotemporal evolution of the number density of particle species, the potential and the mean electron energy. In order to simplify the problem, the simplest plan-parallel configuration in axisymmetric geometry is assumed. In comparison to the time-of-flight and the streamer examples, the FEDM functions  are used extensively. First the simulation conditions (pressure $p_0 = 1 \ Torr$, working voltage $U_w = 250 \ V$, gas temperature $T_g = 300 \ K$) and the geometry are specified manually. Then the model is set up in the following way:

- the particle species list are read from the input file stored in the `file_input` folder using the `read_speclist()` function, which returns the number of species, their names and corresponding file names
- the particle properties are read from the input files stored in the `file_input` folder using the `read_particle_properties()` function, which returns the mass and the charge of the particles
- the number of equations that nedd to be solved are determined based on the used type of approximation in the model (LFA or LMEA) using the `approximation_type()` function
- the output files are created using the `output_files()` function
- the logging of the particle properties, the simulation conditions, the mesh statistics and the time are done using the `log()` function
- the reaction matrices used to construct the rates and the source terms are read from the reaction scheme file using the `reaction_matrices()` function
- the names of files containing the rate coefficients are read using the `rate_coefficient_file_names()` function
- the energy loss for the given reaction are read from the file using the `energy_loss()` function
- the transport coefficients are read from the input files using the `reading_transport_coefficients()` function
- the rate coefficients are read from the input files using the `read_rate_coefficients()` function
- the values of the transport coefficient for the given energy, the reduced electric field or the temperature are obtained using the `Transport_coefficient_interpolation()` function
- the values of the rate coefficient for the given energy, the reduced electric field or the temperature are obtained using the `Rate_coefficient_interpolation()` function
- the source terms for all particle species in the model are obtained using the `Source_term()` function
- the energy source term is generated using the `Energy_Source_term()` function
- the boundaries are marked with the boundary marking function `Marking_boundaries()`, returning the mesh function which is used for the redefinition of the surface integral measure `ds`
- the list of the functions (or trial and test functions for segregated approach) for the given number of particle species is done using `Function_definition()`
- the variational form of the Poisson's equation is defined using `weak_form_Poisson_equation()`
- the fluxes are defined using `Flux()` or `Flux_log()` (for the logarithmic formulation)
- the variational forms of the balance equations for the particle species are defined using `weak_form_balance_equation()` or `weak_form_balance_equation_log_representation()` (for the logarithmic formulation)
- the boundary conditions are defined using `Boundary_flux()` function
- solving of the equation with adaptive time stepping  requires two functions, `adaptive_solver()` and `adaptive_timestep()`
- the results are written into file using `file_output()`

The code can be executed by running the following in terminal:

```bash
python3 fedm-gd.py
```

or in parallel using mpi

```bash
mpirun -np 8 python3 fedm-gd.py
```
