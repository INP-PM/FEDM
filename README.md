# FEDM

## Description

Finite Element electric Discharge Modelling (FEDM) code utilizes the FEniCS (https://fenicsproject.org) computing platform for fluid modelling of the electrical discharges operating under various conditions. The fluid model usually comprises the system of balance equations for particle species, coupled with Poisson's equation and, depending on the used approximation, the electron energy balance equation. In practice, it is often required to take into account a large number of particle species and processes, so the manual definition of the balance equations and the source terms becomes a time-consuming, tedious and error-prone process. This package automates this procedure by providing a set of functions that allows easy definition of the problem. In addition, the package offers a method for the time discretisation of the time-dependent equations using a variable time-step backward differentiation formula with appropriate time-step size control, which are not natively available in the FEniCS. This time-discretisation method allows one to tackle the problem of stiff equations that commonly occur in plasma modelling.

## Build status

Initial build - version 0.1

## Features

- automated function and variational form definition for an arbitrary number of particle species based on a predefined particle species list
- adaptive time step refinement using variable time-step Backward differentiation formula (BDF) of the second-order
- automatized source term definition based on reading the reaction scheme from the input file

## Installation

In order to run this code FEniCS version 2019.1.0. is required. The easiest way to obtain the desired version of FEniCS is by using a Docker image. First, it is required to install the Docker. On Ubuntu/Debian Linux systems, this can be done using the official repository in the following way:

```bash
sudo apt update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

After installing the Docker, the latest stable FEniCS image (currently FEniCS 2019.1.0) can be run using the following command in the terminal:

```bash
sudo docker run -ti \
    -v $(pwd):/home/fenics/shared \
    -v /tmp \
    quay.io/fenicsproject/stable
```

The following steps will also let you run Docker as a non-root user:

```bash
sudo groupadd docker # create docker group if none exists
sudo usermod -aG docker $USER # Add self to docker group
# ... log out and log back in, or call `newgrp docker` ...
docker run -ti \
    -v $(pwd):/home/fenics/shared \
    -v /tmp \
    quay.io/fenicsproject/stable
```

Here it is assumed that the FEDM code is located in the subdirectory `fedm` within the current directory. Now, switch to the shared volume mounted in the container to use FEDM:

```bash
cd shared/fedm
```

The code directory has the following structure:

```
fedm
|-- Examples
|   |-- glow_discharge
|   |   |-- fedm-gd.py
|   |	|-- file_input
|   |   |-- README.md
|   |-- streamer_discharge
|   |   |-- fedm-streamer.py
|   |	|-- file_input
|   |   |-- mesh.xml
|   |   |-- README.md
|   |-- time_of_flight
|   |   |-- fedm-tof.py
|   |   |-- README.md
|-- fedm
|   |-- file_io.py
|   |-- functions.py
|   |-- physical_constants.py
|-- LICENCE
|-- README.md
|-- setup.py
|-- setup.cfg
|-- pyproject.toml
```

FEDM can be installed within the Docker container using:

```bash
python3 -m pip install --user .
```

The directory `Examples` contains the code for the three case studies described in [ADD REFERENCE]. One can execute each example by running the following command in the corresponding directory (you may need to use `sudo`):

```bash
python3 fedm-name_of_example.py
```

or in parallel using MPI:

```bash
mpirun –np 8 python3 fedm-name_of_example.py
```

Note that the new experimental version FEniCSx 0.4 has been recently published, while the FEniCS 2019.1.0 is considered to be deprecated. Currently, there are no plans to update FEDM to be compatible with the newest version (at least until the stable FEniCSx version is published).

## Testing

Testing must be performed within the Docker container. The testing dependencies should 
be installed using:

```bash
python3 -m pip install --user .[tests]
```

The Python `vtk` library is needed to read output files, which in turn requires OpenGL,
so you may need to install this:

```bash
sudo apt update
sudo apt install libgl1
```

The tests are then run using pytest:

```bash
python3 -m pytest -v tests
```

The tests will run each of the integrated tests and store data in the container's `/tmp`
directory. This is why the container must be run with `-v /tmp`.

## How to use?

FEDM consists of the three modules

- `functions.py` storing the functions used for problem definition,

- `file_io.py` for input and output of the data and

- `physical_constants.py` for storing the physical constants.

The main script is used to set up and solve the problem using the functions stored in these modules. The solving of the problem is done in several steps.

1. Reading the input data. In this step, the configuration files for the given problem are read to obtain particle properties and the reaction scheme, and the transport and the reaction rate coefficients. In this way, the number of particles for which the problem is solved is imported into the code. The reaction matrices containing the partial reaction orders and the stoichiometric coefficients for the given species, required for the rate and the source term definitions, are read from the `reacscheme.cfg`. The file contains symbolically written reactions, dependences, energy losses or gains, and names of the files containing the reaction rate coefficient values. The transport and rate coefficients are read from the `.cfg` files and stored in the lists. Depending on the problem, they can be imported in form of a function (written as a string of python code in `.cfg` file) or in form of look-up tables. The files start with a commented header containing information about the coefficients, such as a reference from which the value is taken and the dependence. The transport and rate coefficients may depend on the reduced electric field, the mean electron energy, the electron and/or gas temperature, or may be a constant. The coefficients are stored as functions (written as a string of python or C++ code in the `.cfg` file) or in the form of look-up tables. The files start with a header, containing information about the coefficients such as a reference from which the coefficient value is taken and its dependence, and the coefficient values.  In case when the dependence is the constant or a function, only one value or function is stored in a file. Otherwise, data are stored in two columns, one representing the reduced electric field, the mean electron energy, or the electron temperature and the other representing values of the coefficients. Note that a single blank space must be used for separating the columns, instead of a tab. Also, all files should use utf8 encoding. The following functions are used for reading the input data:

   - `read_speclist()` is used to read the particle species list from the input file stored in the `file_input` folder, and obtain the number of species, their names and corresponding file names
   - `read_particle_properties()`is used to read the particle properties from the input files stored in the `file_input` folder. The function returns the mass and the charge of the particles
   - `reaction_matrices()`  is used to read the reaction matrices from the reaction scheme
   - `rate_coefficient_file_names()` are used to read the name of the files containing the rate coefficient from the reaction scheme file
   - `energy_loss()` is used to read the energy loss for the given reaction from the reaction scheme file
   - `reading_transport_coefficients()` is used to read the transport coefficients from the input files
   - `reading_rate_coefficients()` is used to read the rate coefficients from the input files

2. Mesh for FEM discretization. The mesh is either generated using a built-in function (structured mesh) or imported from the external (`.xml` or `.xdmf`) file.

3. Function spaces. In the examples presented here, a fully coupled solution approach is used, so the native mixed element functions are used for the function space definition. The mixed element list is obtained using the custom function `Mixed_element_list()` which creates a list of the function elements. Since the number of functions varies from problem to problem, the procedure of function definition is automatised by calling the custom function  `Function_definition()`.

4. Domain boundaries. Depending on the given problem, one may need to mark the specific boundaries. This can be done using the custom function `Marking_boundaries()`.

5. Evaluation of transport and rate coefficients. The imported transport and rate coefficients are interpolated in the code when needed.

6. Source term definition. Particle source term definition is done using the function `Source_term()`, while the energy source term definition is done using `Energy_Source_term()`, using the imported reaction matrices (described in step 1).

7. Fluxes. Depending on whether the transformation of the variables is done or not (e.g. logarithmic representation), the fluxes are defined using two functions:

   - For the case when the transformation of variables is not done, function `weak_form_balance_equation()` is used
   - For the logarithmic representation of the equation (where the problem is solved for the logarithm of the variable) `weak_form_balance_equation_log_representation()` is designed.

8. Weak formulation. For the small examples, the weak formulation can be done manually for each equation. However, for the cases where the problem is solved for multiple particle species, this can become tedious. In this case, one may use a set of functions to define the variational form of equations. For the balance equations of particle species, two functions exist:

   - For the case when the transformation of the variables is not done, function `weak_form_balance_equation()` is used
   - For the logarithmic representation of the equation (where the problem is solved for the logarithm of the variable) `weak_form_balance_equation_log_representation()` is used


      Both functions implement the backward differentiation formula (BDF) of the second order with variable time-step. In practice, the change of    order can be done without redefining the time-stepping scheme in the time-loop. Instead, by defining the previous time-step size `dt_old` as some large number, the adaptive BDF is reduced to the first order. Moreover, by defining that `dt_old = dt`, adaptive BDF can be reduced to BDF with a constant time-step size. Therefore, the seamless initiation of the formula and switching between two orders is possible.


   - For the electron energy balance equation, the same function may be used as for particle balance equations.
   - For Poisson's equation, a variational form may be defined manually or using the `weak_form_Poisson_equation()` function.

9. Boundary conditions. The boundary integrals (if required) are defined separately using the function `Boundary_flux()` and added to the weak formulation.

10. Solver. FEniCS has an option to access PETSc solvers available in the PETSc library. The function `PETScSNESSolver()` is used to set up the nonlinear solver. The advantage of this approach is that it is possible to tune the nonlinear and linear solver parameters, such as relative or absolute tolerance, line search type in SNES, the maximum number of iterations, etc. It should be noted that it is possible to define your own custom solvers (see Cahn-Hilliard equation demo in FEniCS repository).

11. The file output. The linear interpolation of the calculated results is used to determine the output value for the particular time step, which are then saved into the file using the function `file_output()`.

12. Adaptive time stepping. The time-step size control is based on the L2 norm difference between the previous and the current step. Depending on used approximation, relevant variable may be mean electron energy, electron number density or the whole solution vector. The new time step is obtained using the function `adaptive_timestep()` which utilizes PID controller, described in the article.

## Note from the author
The FEDM code was developed while the author was learning the Python language. Most of the functions are not written in a Pythonic way and many of them probably already exist. However, since the code has been verified and works quite well in its present form, we have no intention to change or update the functions (i.e., we follow the first rule of programming: "If it ain't broke, don't fix it").

## License

In agreement with the FEniCS licensing, FEDM is open source code developed under LGPLv3 (GNU Lesser General Public License version 3).

## Acknowledgment

The development of the FEDM is funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation)—project number 407462159. The authors wish to thank the users of [FEniCS forum](https://fenicsproject.discourse.group) for useful information and discussion.

## Citation

Please cite the paper [ADD REFERENCE] if you use the FEDM code for your research.

## Contact

[aleksandar.jovanovic@inp-greifswald.de](mailto:aleksandar.jovanovic@inp-greifswald.de)

[markus.becker@inp-greifswald.de](mailto:markus.becker@inp-greifswald.de)
