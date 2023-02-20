"""
This small example presents modelling of time-of-flight experiment and is used for verification purpose.
Namely, for this particular test case the exact analytical solution for the electron number density exists,
so it is used to verify the accuracy of the code using the method of exact solutions. L2 norm is used to
quantify the difference between the solutions and to determine the mesh and time order-of-accuracy of the
code for solving the balance equation.
"""
from dolfin import *
import numpy as np
from timeit import default_timer as timer
import time
from pathlib import Path
from argparse import ArgumentParser
from fedm.physical_constants import *
from fedm.file_io import *
from fedm.functions import *

def main(output_dir = None):
    if output_dir is not None:
        files.output_folder_path = Path(output_dir)

    # Optimization parameters.
    parameters["form_compiler"]["optimize"]     = True
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters['krylov_solver']['nonzero_initial_guess'] = True

    # Defining tye of used solver and its parameters.
    linear_solver = "mumps" # Setting linear solver: mumps | gmres | lu
    maximum_iterations = 50 # Setting up maximum number of nonlinear solver iterations
    relative_tolerance = 1e-10 # Setting up relative tolerance

    # ============================================================================
    # Definition of the simulation conditions, model and coordinates
    # ============================================================================
    model = 'Time_of_flight' # Model name
    coordinates = 'cylindrical' # Coordinates choice
    gas = 'Air'
    Tgas = 300.0 # Gas temperature in [K]
    p0 = 760.0 # Pressure in [Torr]
    N0 = p0*3.21877e22 # Number density in [m^-3]

    # ============================================================================
    # Defining number of species for which the problem is solved, their properties and creating output files.
    # ============================================================================
    number_of_species = 1
    particle_species_type = ['electrons', 'analytical solution'] # Defining particle species types, where the values are used as names for output files
    M = me
    charge = -elementary_charge
    equation_type = ['drift-diffusion-reaction'] # Defining the type of the equation (reaction | diffusion-reaction | drift-diffusion-reaction)
    wez = 1.7e5 # Electron drift velocity z component [m/s] 
    De = 0.12 # Electron Diffusion coefficient [m^2/s]
    alpha_e = 5009.51 # Effective ionization coefficient [1/m]

    log('properties', files.model_log, gas, model, particle_species_type, M, charge) # Writting particle properties into a log file
    vtkfile_u = output_files('pvd', 'number density', particle_species_type) # Setting-up output files

    # ============================================================================
    # Definition of the time variables.
    # ============================================================================
    t_old = None # Previous time step
    t0 = 2.5e-9 # Initial time step
    t = t0 # Current time step
    # Test changes: T_final is reduced from 3e-9 to 2.6e-9
    T_final = 2.6e-9 # Simulation time end [s]

    dt_init = 1e-12 # Initial time step size
    dt = Expression("time_step", time_step = dt_init, degree = 0) # Time step size [s]
    dt_old = Expression("time_step", time_step = 1e30, degree = 0) # Time step size [s] set up as a large value to reduce the order of BDF.

    t_output_step = 1e-9 # Time step intervals at which the results are printed to file
    # Test changes: from 3e-9 to 2.6e-9
    t_output = 2.6e-9 # Initial output time step

    # ============================================================================
    # Defining the geometry of the problem and corresponding boundaries.
    # ============================================================================
    if coordinates == 'cylindrical':
        r = Expression('x[0]', degree = 1)
        z = Expression('x[1]', degree = 1)

    # Gap length and width
    # Test changes: these values are halved
    box_width = 2.5e-4  # [m]
    box_height = 5e-4 # [m]

    boundaries = [['line', 0.0, 0.0, 0.0, box_width],\
                    ['line', box_height, box_height, 0.0, box_width],\
                    ['line', 0.0, box_height, 0.0, 0.0],\
                    ['line', 0.0, box_height, box_width, box_width]] # Defining list of boundaries lying between z0 and z1 and r0 and r1

    # ============================================================================
    # Mesh setup. Structured mesh is generated using built-in mesh generator.
    # ============================================================================
    # Test changes: mesh has half as many cells
    mesh = RectangleMesh(Point(0, 0), Point(box_width, box_height), 40, 40) # Generating structured mesh.
    mesh_statistics(mesh) # Prints number of elements, minimal and maximal cell diameter.
    h = MPI.max(MPI.comm_world, mesh.hmax()) # Maximuml cell size in mesh.

    log('conditions', files.model_log, dt.time_step, 'None', p0, box_height, N0, Tgas)
    log('mesh', files.model_log, mesh)
    log('initial time', files.model_log, t)

    # ============================================================================
    # Defining type of elements and function space, test functions, trial functions and functions for storing variables, and weak form
    # of equation.
    # ============================================================================
    V = FunctionSpace(mesh, 'P', 1) # Defining function space
    W = VectorFunctionSpace(mesh, 'P', 1) # Defining vector function space

    u = TrialFunction(V) # Defining trial function
    v = TestFunction(V) # Defining test function
    u_old = Function(V) # Defining function for storing the data at k-1 time step
    u_old1 = Function(V) # Defining function for storing the data at k-2 time step
    u_new = Function(V) # Defining function for storing the data at k time step

    u_analytical  = Expression('std::log(exp(-(pow(x[1]-w*t, 2)+pow(x[0], 2))/(4.0*D*t)+alpha*w*t)/pow(4*D*t*pi,1.5))', D = De, w = wez, alpha = alpha_e, t = t, pi=pi,  degree = 3) # Analytical solution of the particle balance equation.
    u_old.assign(interpolate(u_analytical , V)) # Setting up value at k-1 time step
    u_old1.assign(interpolate(u_analytical , V)) # Setting up value at k-2 time step

    w = interpolate(Constant(('0', wez)), W) # Electron drift velocity [m/s]
    D = interpolate(Constant(De), V) # Diffusion coefficient [m^2/s]
    alpha_eff = interpolate(Constant(alpha_e), V) # Effective ionization coefficient [1/m]

    Gamma = -grad(D*exp(u)) + w*exp(u) # Defining electron flux [m^{-2} s^{-1}]
    f = Expression('exp(-(pow(x[1]-w*t, 2)+pow(x[0], 2))/(4.0*D*t)+alpha*w*t)*(w*alpha)/(8*pow(pi,1.5)*pow(D*t, 1.5))', D = De, w = wez, alpha = alpha_e, t = t, pi=pi,  degree = 2) # Defining source term

    F = weak_form_balance_equation_log_representation(equation_type[0], dt, dt_old, dx, u, u_old, u_old1, v, f, Gamma, r) # Definition of variational formulation of the balance equation for the electrons

    u_new.assign(interpolate(Expression('std::log(exp(-(pow(x[1]-w*t, 2)+pow(x[0], 2))/(4.0*D*t)+alpha*w*t)/pow(4.0*D*t*pi,1.5) + DOLFIN_EPS)', D = De, w = wez, alpha = alpha_e, t = t, pi=pi,  degree = 2), V)) # Setting up initial guess for nonlinear solver

    # ============================================================================
    # Setting-up nonlinear solver
    # ============================================================================
    # Defining the problem
    F = action(F, u_new)
    J = derivative(F, u_new, u)
    problem = Problem(J, F, [])

    # Initializing nonlinear solver and setting up the parameters
    nonlinear_solver = PETScSNESSolver() # Nonlinear solver initialization
    nonlinear_solver.parameters['relative_tolerance'] = relative_tolerance # Setting up relative tolerance of the nonlinear solver
    nonlinear_solver.parameters["linear_solver"]= linear_solver # Setting up linear solver
    nonlinear_solver.parameters['maximum_iterations'] = maximum_iterations # Setting up maximum number of iterations
    # nonlinear_solver.parameters["preconditioner"]="hypre_amg" # Setting the preconditioner, uncomment if iterative solver is used

    while abs(t-T_final)/T_final > 1e-6:
        t_old = t # Updating old time steps
        u_old1.assign(u_old) # Updating variable value in k-2 time step
        u_old.assign(u_new) # Updating variable value in k-1 time step
        t += dt.time_step # Updating the new  time steps

        log('time', files.model_log, t) # Time logging
        print_time(t) # Printing out current time step

        f.t=t # Updating the source term for the current time step
        u_analytical.t = t # Updating the analytical solution for the current time step

        nonlinear_solver.solve(problem, u_new.vector()) # Solving the system of equations

        if abs(t-t_output)/t_output <= 1e-6:
            n_exact = project(exp(u_analytical), V, solver_type='mumps')
            n_num = project(exp(u_new), V, solver_type='mumps')
            relative_error = errornorm(n_num, n_exact, 'l2')/norm(n_exact, 'l2') # Calculating relative difference between exact analytical and numerical solution
            with open(files.error_file, "a") as f_err:
                f_err.write('h_max = ' + str(h) + '\t dt = ' + str(dt.time_step) + '\t relative_error = ' + str(relative_error) + '\n')
            if(MPI.rank(MPI.comm_world)==0):
                print(relative_error)
            vtkfile_u[0] << (n_num, t)
            vtkfile_u[1] << (n_exact, t)
            t_output += t_output_step

        if t > (t0 + dt_init):
            dt_old.time_step = dt.time_step # For initialization BDF1 is used and after initial step BDF2 is activated
            if(MPI.rank(MPI.comm_world)==0):
                print(str(dt_old.time_step) + '\t' + str(dt.time_step) +'\n')

if __name__ == "__main__":
    arg_parser = ArgumentParser(description="FEDM time of flight test")
    arg_parser.add_argument(
        "-o", "--output", help="output directory", type=Path, default=None
    )
    args = arg_parser.parse_args()
    main(output_dir=args.output)
