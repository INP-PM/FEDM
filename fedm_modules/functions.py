# Fluid modelling functions module

from dolfin import *
import numpy as np
import sys

from .physical_constants import elementary_charge, kB, kB_eV

def approximation_type(approx, number_of_spec, particle_spec = 0, mass = 0, charge = 0):
    """
    Depending on approximation used, the number of equations, charge
    and mass variables are modified.
    """
    if approx == 'LFA':
        number_of_eq = number_of_spec
        number_of_spec -= 1
        particle_spec.remove(particle_spec[0])
        mass.remove(mass[0])
        charge.remove(charge[0])
        return number_of_spec, number_of_eq, particle_spec, mass, charge
    elif approx == 'LMEA':
        number_of_eq = number_of_spec + 1
        return number_of_eq

def mesh_statistics(mesh):
    """
    Returns mesh size and, maximum and minimum element size.
    Input is mesh.
    """
    file_name = 'mesh'
    file_path = 'output/mesh/' + file_name + '.pvd'
    vtkfile_mesh = File(file_path)
    vtkfile_mesh << mesh
    n_element = MPI.sum(MPI.comm_world, mesh.num_cells())
    #measures the greatest distance between any two vertices of a cell
    hmax = MPI.max(MPI.comm_world, mesh.hmax())
    #measures the smallest distance between any two vertices of a cell
    hmin = MPI.min(MPI.comm_world, mesh.hmin())
    if(MPI.rank(MPI.comm_world)==0):
        mesh_information = open('output/mesh/mesh info.txt','w')
        print("Number of elements is:", int(n_element))
        print("Maximum element edge length is:", hmax)
        print("Minimum element edge length is:", hmin)
        mesh_information.write("Number of elements is: ")
        print("%.*g" % (5, n_element), file = mesh_information)
        mesh_information.write("\n" + "Maximum element edge length is: ")
        print("%.*g" % (5, hmax), file=mesh_information)
        mesh_information.write("\n" + "Minimum element edge length is: ")
        print("%.*g" % (5, hmin), file=mesh_information)
        mesh_information.write("\n")
        mesh_information.close()

def Marking_boundaries(mesh, boundary = [[]], submesh = 'No'):
    """
    Marking boundaries of a provided mesh. Currently, straight-line and circular
    boundaries are supported. First argument is the mesh, the second
    argument is a list of boundary properties (boundary type and coordinates).
    """

    tol=1e-8
    class circle(SubDomain):
        def inside(self, x, on_boundary):
            if submesh == 'No':
                if center_z <= 0:
                    if abs(pow((x[1] - center_z) , 2) + pow(x[0] - center_r, 2) - pow(radius, 2)) <= tol and x[1] <= 0.0 and on_boundary:
                        return True
                else:
                    if abs(pow((x[1] - center_z) , 2) + pow(x[0] - center_r, 2) - pow(radius, 2)) <= tol and x[1] >= gap_length and on_boundary:
                        return True
            else:
                if center_z <= 0:
                    if abs(pow((x[1] - center_z) , 2) + pow(x[0] - center_r, 2) - pow(radius, 2)) <= tol and x[1] <= 0.0:
                        return True
                else:
                    if abs(pow((x[1] - center_z) , 2) + pow(x[0] - center_r, 2) - pow(radius, 2)) <= tol and x[1] >= gap_length:
                        return True

    class line(SubDomain):
       def inside(self, x, on_boundary):
           return between(x[0], (r1, r2)) and between(x[1], (z1, z2)) and on_boundary

    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    i = 0
    while i < len(boundary):
        if(MPI.rank(MPI.comm_world)==0):
            print(boundary[i][0])
        if boundary[i][0] == 'circle':
            center_z = boundary[i][1]
            center_r = boundary[i][2]
            radius = boundary[i][3]
            bmark = circle()
            bmark.mark(boundary_markers, i)
        elif boundary[i][0] == 'line':
            z1 = boundary[i][1] - DOLFIN_EPS
            z2 = boundary[i][2] + DOLFIN_EPS
            r1 = boundary[i][3] - DOLFIN_EPS
            r2 = boundary[i][4] + DOLFIN_EPS
            bmark = line()
            bmark.mark(boundary_markers, i)
        i += 1
    return boundary_markers

def Mixed_element_list(number_of_equations, P):
    """
    Defines list of mixed elements. Input arguments are
    number of equations and element type.
    """
    element_list = []
    i = 0
    while i < number_of_equations:
        element_list.append(P)
        i += 1
    return element_list

def Function_space_list(number_of_equations, V):
    """
    Defines list of function spaces. Input arguments are
    number of equations and function space.
    """
    space_list = []
    i = 0
    while i < number_of_equations:
        space_list.append(V)
        i += 1
    return space_list

def Function_definition(F_space, f_type, eq_number = 1):
    """
    Defines list of desired function type (TrialFunction, TestFunction or ordinary Function).
    Input arguments are function space, type of desired function and number of equations,
    where the default value is one.
    """
    u_temp = []
    if f_type == 'TrialFunction':
        i = 0
        while i < eq_number:
            temp = TrialFunction(F_space)
            u_temp.append(temp)
            i += 1
    elif f_type == 'TestFunction':
        i = 0
        while i < eq_number:
            temp = TestFunction(F_space)
            u_temp.append(temp)
            i += 1
    elif f_type == 'Function':
        i = 0
        while i < eq_number:
            temp = Function(F_space)
            u_temp.append(temp)
            i += 1
    return u_temp

class Problem(NonlinearProblem):
    # """
    # Nonlinear problem definition. Input parameters are F - weak form of the equation
    # J - Jacobian and bcs -Dirichlet boundary conditions. The part of code is provided
    # by user Nate Sime from FEniCS forum:
    # https://fenicsproject.discourse.group/t/set-krylov-linear-solver-paramters-in-newton-solver/1070/3
    # """
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        # """
        # Linear form assembly
        # """
        assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        # """
        # Bilinear form assembly
        # """
        assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)

def Max(a, b):
    """
    Returns maximum value of a and b.
    """
    return (a+b+abs(a-b))/Constant(2.0)

def Min(a, b):
    """
    Returns minimum value of a and b.
    """
    return (a+b-abs(a-b))/Constant(2.0)

def Flux(sign, u, D, mu, E):
    """
    Defines particle flux using drift-diffusion approximation.
    Input arguments are particle charge, number density,
    diffusion coefficient, mobility and electric field.
    """
    return -grad(D*u) + sign*mu*E*u

def Flux_log(sign, u, D, mu, E):
    """
    Defines particle flux using drift-diffusion and logarithmic approximation.
    Input arguments are particle charge, number density,
    Diffusion coefficient, mobility and electric field.
    """
    return -grad(D*exp(u)) + sign*mu*E*exp(u)

def weak_form_balance_equation_log_representation(equation_type, dt, dt_old, dx, u, u_old, u_old1, v, f, Gamma, r = 0.5/pi, D = 0):
    """
    Returns the weak form of the particle balance equations for logarithmic representation. Input
    arguments are equation type (reaction | diffusion-reaction | drift-diffusion-reaction),
    current time-step size, old time-step size, dV, trial function, value of variable in current
    and previous time step, test function, source term, particle flux, r coordinate and
    diffusion coefficient, which is only required for the diffusion equation.
    """
    if (equation_type == 'reaction'):
        return 2.0*pi*exp(u)*((((1.0+2.0*dt/dt_old)/(1.0+dt/dt_old))*(u - (pow(1.0+dt/dt_old, 2.0)/(1.0+2.0*dt/dt_old))*u_old\
         + (pow(dt/dt_old, 2.0)/(1.0+2.0*dt/dt_old))*u_old1))*v/dt)*r*dx - 2.0*pi*f*v*r*dx
    elif (equation_type == 'diffusion-reaction'):
        return 2.0*pi*exp(u)*((((1.0+2.0*dt/dt_old)/(1.0+dt/dt_old))*(u - (pow(1.0+dt/dt_old, 2.0)/(1.0+2.0*dt/dt_old))*u_old\
         + (pow(dt/dt_old, 2.0)/(1.0+2.0*dt/dt_old))*u_old1))*v/dt)*r*dx - 2.0*pi*dot(-grad(D*exp(u)), grad(v))*r*dx - 2.0*pi*f*v*r*dx
    elif (equation_type == 'drift-diffusion-reaction'):
        return 2.0*pi*exp(u)*((((1.0+2.0*dt/dt_old)/(1.0+dt/dt_old))*(u - (pow(1.0+dt/dt_old, 2.0)/(1.0+2.0*dt/dt_old))*u_old\
         + (pow(dt/dt_old, 2.0)/(1.0+2.0*dt/dt_old))*u_old1))*v/dt)*r*dx - 2.0*pi*dot(Gamma, grad(v))*r*dx - 2.0*pi*f*v*r*dx

def weak_form_balance_equation(equation_type, dt, dt_old, dx, u, u_old, u_old1, v, f, Gamma, r = 0.5/pi, D = 0):
    """
    Returns the weak form of particle balance equations. Input arguments are
    equation type (reaction | diffusion-reaction | drift-diffusion-reaction),
    current time step size, old time step size, dV, trial function, value of
    variable in current and previous time step, test function, source term,
    particle flux, r coordinate and diffusion coefficient, which is only
    required for diffusion equation.
    """
    if (equation_type == 'reaction'):
        return 2.0*pi*((((1.0+2.0*dt/dt_old)/(1.0+dt/dt_old))*(u - (pow(1.0+dt/dt_old, 2.0)/(1.0+2.0*dt/dt_old))*u_old\
         + (pow(dt/dt_old, 2.0)/(1.0+2.0*dt/dt_old))*u_old1))*v/dt)*r*dx - 2.0*pi*f*v*r*dx
    elif (equation_type == 'diffusion-reaction'):
        return 2.0*pi*((((1.0+2.0*dt/dt_old)/(1.0+dt/dt_old))*(u - (pow(1.0+dt/dt_old, 2.0)/(1.0+2.0*dt/dt_old))*u_old\
         + (pow(dt/dt_old, 2.0)/(1.0+2.0*dt/dt_old))*u_old1))*v/dt)*r*dx - 2.0*pi*dot(-grad(D*u), grad(v))*r*dx - 2.0*pi*f*v*r*dx
    elif (equation_type == 'drift-diffusion-reaction'):
        return 2.0*pi*((((1.0+2.0*dt/dt_old)/(1.0+dt/dt_old))*(u - (pow(1.0+dt/dt_old, 2.0)/(1.0+2.0*dt/dt_old))*u_old\
         + (pow(dt/dt_old, 2.0)/(1.0+2.0*dt/dt_old))*u_old1))*v/dt)*r*dx - 2.0*pi*dot(Gamma, grad(v))*r*dx - 2.0*pi*f*v*r*dx

def weak_form_Poisson_equation(dx, u, v, f, r = 0.5/pi):
    """
    Returns a weak form of Poisson equation. Input arguments are dV,
    trial function, test function, surce term and r coordinate.
    """
    return 2.0*pi*inner(grad(u), grad(v))*r*dx - 2.0*pi*f*v*r*dx

def Boundary_flux(bc_type, type_of_equation, particle_type, sign, mu, E, normal, u, gamma, v, ds_temp, r = 0.5/pi, vth = 0.0, ref = 1.0, Ion_flux = 0.0):
    """
    Function defines boundary conditions for different equations.
    Input arguments are boundary condition type, type of equation,
    type of particle, particle charge sign, mobility, electric field,
    normal, trial function, secondary electron emmision coefficient,
    test function, ds,  r coordinate, thermal velocity, reflection
    coefficient for specified particle species and boundary, and
    flux of ions.
    """
    BF = 0.0
    if bc_type == 'zero flux':
        BF = 0.0
    elif bc_type == 'flux source':
        if type_of_equation == 'reaction':
            BF = 0.0
        elif type_of_equation == 'diffusion-reaction':
            BF = 2*pi*((1.0-ref)/(1.0+ref))*(0.5*vth *exp(u))*v*r*ds_temp
        elif type_of_equation == 'drift-diffusion-reaction':
            if particle_type  == 'Heavy':
                BF = 2*pi*((1.0-ref)/(1.0+ref))*(0.5*vth*exp(u) + abs(sign*mu * dot(E, normal)*exp(u)))*v*r*ds_temp
            elif particle_type == 'electrons':
                BF = 2*pi*(((1.0-ref)/(1.0+ref))*(0.5*vth*exp(u) + abs(mu * dot(E, normal)*exp(u))))*v*r*ds_temp - 2.0*pi*2.0/(1.0+ref)*gamma*Ion_flux*v*r*ds_temp
    elif bc_type == 'Neumann':
        if type_of_equation == 'reaction':
            BF = 0.0
        elif type_of_equation == 'diffusion-reaction':
            BF = 0.0
        elif type_of_equation == 'drift-diffusion-reaction':
            BF = 2.0*pi*dot(sign*mu * E, normal)*exp(u)*v*r*ds
    return BF

def Transport_coefficient_interpolation(status, dependence, N0, Tgas, k_coeff, kx, ky, energy, redfield, mu = 0.0, Te = 0):
    """
    Function for linear interpolation of transport coefficients. Input
    arguments are status (initial definition or update), dependence
    of varaible (const, energy, reduced electric field, ESR), gas
    number density, gas temperature, list of coefficient variables,
    look-up table data (k_x and k_y), energy, reduced electric field
    and mobility (for ESR).
    """
    num_par = len(k_coeff)
    if status == 'initial':
        i = 0
        while i < num_par:
            if dependence[i] == 'const':
                k_coeff[i].vector()[:] = ky[i]/N0
            elif dependence[i] == 'Umean':
                k_coeff[i].vector()[:] = np.interp(energy.vector()[:], kx[i], ky[i])/N0
            elif dependence[i] == 'E/N':
                k_coeff[i].vector()[:] = np.interp(redfield.vector()[:], kx[i], ky[i])/N0
            elif dependence[i] == 'ESR':
                k_coeff[i].vector()[:] = kB*Tgas*mu[i].vector()[:]/elementary_charge
            elif dependence[i] == 'Tgas':
                k_coeff[i].vector()[:] = np.interp(Tgas, kx[i], ky[i])/N0
            i += 1
    elif status == 'update':
        i = 0
        while i < num_par:
            if dependence[i] == 'Umean':
                k_coeff[i].vector()[:] = np.interp(energy.vector()[:], kx[i], ky[i])/N0
            elif dependence[i] == 'E/N':
                k_coeff[i].vector()[:] = np.interp(redfield.vector()[:], kx[i], ky[i])/N0
            elif dependence[i] == 'ESR':
                k_coeff[i].vector()[:] = kB*Tgas*mu[i].vector()[:]/elementary_charge
            elif dependence[i] == 'Tgas':
                k_coeff[i].vector()[:] = np.interp(Tgas, kx[i], ky[i])/N0
            i += 1

def Rate_coefficient_interpolation(status, dependence, k_coeff, kx, ky, energy, redfield, Te = 0, Tgas = 300):
    """
    Function for linear interpolation of rate coefficients. Input arguments
    are status (initial  or update), dependence of varaible (const, energy,
    reduced electric field, Te, Tgas), list of coefficient variables, look-up
    table, energy and reduced electric field.
    """
    nr = len(k_coeff)
    if status == 'initial':
        i = 0
        while i < nr:
            if dependence[i] == 'const':
                k_coeff[i].vector()[:] = ky[i]
            elif dependence[i] == 'Umean':
                k_coeff[i].vector()[:] = np.interp(energy.vector()[:], kx[i], ky[i])
            elif dependence[i] == 'E/N':
                k_coeff[i].vector()[:] = np.interp(redfield.vector()[:], kx[i], ky[i])
            elif dependence[i] == 'fun:Te,Tgas':
                k_coeff[i] = eval(ky[i])
            elif dependence[i] == 'fun:Tgas':
                k_coeff[i] = eval(ky[i])
            i += 1
    elif status == 'update':
        i = 0
        while i < nr:
            if dependence[i] == 'Umean':
                k_coeff[i].vector()[:] = np.interp(energy.vector()[:], kx[i], ky[i])
            elif dependence[i] == 'E/N':
                k_coeff[i].vector()[:] = np.interp(redfield.vector()[:], kx[i], ky[i])
            elif dependence[i] == 'Te':
                k_coeff[i].vector()[:] = np.interp(2*energy.vector()[:]/(3*kB_eV), kx[i], ky[i])
            i += 1

def semi_implicit_coefficients(dependence, mean_energy_new, mean_energy_old, coefficient, coefficient_diff):
    coefficient_si = []
    i = 0
    while i < len(dependence):
        if dependence[i] == "Umean":
            coefficient_si.append(coefficient[i] + coefficient_diff[i]*(mean_energy_new - mean_energy_old))
        else:
            coefficient_si.append(coefficient[i])
        i += 1
    return coefficient_si

def Source_term(coupling, approx, p_matrix, l_matrix, g_matrix, k_coeff, N0, u):
    """
    Defines source term for coupled or uncoupled approach, with
    LFA (counting particles from 0) or LMEA approximation
    (counting particle from 1 and the zeroth equation is energy).
    Function arguments are power, loss and gain matrices,
    the rate coeffiient, energy loss, mean energy, gas number
    density  and particle number density variable
    """
    if coupling == "coupled":
        if approx == "LFA":
            nr = len(k_coeff)
            neq = len(u)
            Rate = [0]*nr
            j = 0
            while j < nr:
                temp = 1
                i = 0
                while i < neq:
                    if i == 0:
                        temp *= np.power(N0, p_matrix[j, i])
                    else:
                        temp *= np.power(exp(u[i-1]), p_matrix[j, i])
                    i += 1
                Rate[j] = k_coeff[j]*temp
                j += 1
            f_temp = [0]*neq
            i = 0
            while i < neq:
                temp = 0
                j = 0
                while j < nr:
                    temp += (g_matrix[j, i] *Rate[j] - l_matrix[j, i] *Rate[j])
                    j += 1
                f_temp[i] = temp
                i += 1
        elif approx == "LMEA":
            nr = len(k_coeff)
            neq = len(u)-1
            Rate = [0]*nr
            j = 0
            while j < nr:
                temp = 1
                i = 0
                while i < neq:
                    if i == 0:
                        temp *= np.power(N0, p_matrix[j, i])
                    else:
                        temp *= np.power(exp(u[i]), p_matrix[j, i])
                    i += 1
                Rate[j] = k_coeff[j]*temp
                j += 1
            f_temp = [0]*neq
            i = 0
            while i < neq:
                temp = 0
                j = 0
                while j < nr:
                    temp += (g_matrix[j, i] *Rate[j] - l_matrix[j, i] *Rate[j])
                    j += 1
                f_temp[i] = temp
                i += 1
    elif coupling == "uncoupled":
        if approx == "LFA":
            nr = len(k_coeff)
            neq = len(u)
            Rate = [0]*nr
            j = 0
            while j < nr:
                temp = 1
                i = 0
                while i < neq:
                    if i == 0:
                        temp *= np.power(N0, p_matrix[j, i])
                    else:
                        temp *= np.power(exp(u[i]), p_matrix[j, i])
                    i += 1
                Rate[j] = k_coeff[j]*temp
                j += 1
            f_temp = [0]*neq
            i = 0
            while i < neq:
                temp = 0
                j = 0
                while j < nr:
                    temp += (g_matrix[j, i] *Rate[j] - l_matrix[j, i] *Rate[j])
                    j += 1
                f_temp[i] = temp
                i += 1
        elif approx == "LMEA":
            nr = len(k_coeff)
            neq = len(u)
            Rate = [0]*nr
            j = 0
            while j < nr:
                temp = 1
                i = 0
                while i < neq:
                    if i == 0:
                        temp *= np.power(N0, p_matrix[j, i])
                    else:
                        temp *= np.power(exp(u[i]), p_matrix[j, i])
                    i += 1
                Rate[j] = k_coeff[j]*temp
                j += 1
            f_temp = [0]*neq
            i = 0
            while i < neq:
                temp = 0
                j = 0
                while j < nr:
                    temp += (g_matrix[j, i] *Rate[j] - l_matrix[j, i] *Rate[j])
                    j += 1
                f_temp[i] = temp
                i += 1
    return f_temp

def Energy_Source_term(coupling, p_matrix, l_matrix, g_matrix, k_coeff, u_loss, mean_energy, N0, n, Ei = 0):
    """
    Defines energy source term for LMEA approximation. Function arguments
    are power, loss and gain matrices, rate coeffiients, energy losses for specific
    process, mean electron energy, gas number density and particle number density
    variable.
    """
    if coupling == "coupled":
        nr = len(k_coeff)
        neq = len(n) - 1
        Rate = [0]*nr
        j = 0
        while j < nr:
            temp = 1
            i = 0
            while i < neq:
                if i == 0:
                    temp *= np.power(N0, p_matrix[j, i])
                else:
                    temp *= np.power(exp(n[i]), p_matrix[j, i])
                i += 1
            if u_loss[j] > 7e77 and u_loss[j] < 8e77:
                Rate[j] = -(Ei - mean_energy)*k_coeff[j]*temp
            elif  u_loss[j] > 9e99 and u_loss[j] < 1e100:
                Rate[j] = -mean_energy*k_coeff[j]*temp
            else:
                Rate[j] = -u_loss[j]*k_coeff[j]*temp
            j += 1
        f_temp = 0
        i = 0
        while i < nr:
                f_temp +=  Rate[i]
                i += 1
    elif coupling == "uncoupled":
        nr = len(k_coeff)
        neq = len(n)
        Rate = [0]*nr
        j = 0
        while j < nr:
            temp = 1
            i = 0
            while i < neq:
                if i == 0:
                    temp *= np.power(N0, p_matrix[j, i])
                else:
                    temp *= np.power(exp(n[i]), p_matrix[j, i])
                i += 1
            if u_loss[j] > 7e77 and u_loss[j] < 8e77:
                Rate[j] = -(Ei - mean_energy)*k_coeff[j]*temp
            elif  u_loss[j] > 9e99 and u_loss[j] < 1e100:
                Rate[j] = -mean_energy*k_coeff[j]*temp
            else:
                Rate[j] = -u_loss[j]*k_coeff[j]*temp
            j += 1
        f_temp = 0
        i = 0
        while i < nr:
                f_temp +=  Rate[i]
                i += 1
    return f_temp

def adaptive_timestep(dt, error, tol = 1e-4, dt_min = 1e-13, dt_max = 1e-9):
    """
    Function calculates new time step based on a PID controller
    M.  Moeller,  Time  stepping  methods,  ATHENS  course:
    Introductioninto Finite Elements, Delft Institute of Applied
    Mathematics, TU Delft(2015). Input arguments are time step,
    error, tolerance for time step controller, minimal and maximal
    time step.
    """
    dt *= (error[1]/error[0])**0.075 * (tol/error[0])**0.175 * (error[1]**2/(error[0]*error[2]))**0.01
    dt = min(dt, dt_max)
    dt = max(dt, dt_min)
    return dt

def adaptive_timestep_PI34(dt, error, tol = 1e-4, dt_min = 1e-13, dt_max = 1e-9):
    """
    Function calculates new time step size using PI.3.4 controller
    (G. Soederlind Numerical Algorithms 31: 281-310, 2002). Input
    arguments are time-step size, error, tolerance for time step controller,
    minimal and maximal time-step size.
    """
    dt *= (0.8*tol/error[0])**(0.3/3)* (0.8*error[1]/error[0])**(0.4/3)
    dt = min(dt, dt_max)
    dt = max(dt, dt_min)
    return dt

def adaptive_timestep_H211b(dt, dt_old, error, tol = 1e-4, dt_min = 1e-13, dt_max = 1e-9):
    """
    Function calculates new time step size using H211b controller
    (G. Soederlind, Acm. T. Math. Software 29: 1-26, 2003). Input arguments are
    time step size, previous time step size, error, tolerance for time
    step controller, minimal and maximal time step.
    """
    dt *= (0.8*tol/error[0])**(1/12)* (0.8*tol/error[1])**(1/12)*(dt/dt_old)**(-1/4)
    dt = min(dt, dt_max)
    dt = max(dt, dt_min)
    return dt

def adaptive_solver(nonlinear_solver, problem, t, dt, dt_old, u_new, u_old, var_list_new, var_list_old, assigner, error, file_error, max_error, ttol, dt_min, time_dependent_arguments = [], approximation = 'LMEA'):
    '''
    This function is used for solving the problem when adaptive time stepping
    is used. Input arguments are solver (PETScSNESSolver or NonlinearSolver),
    problem, time step, time step size, previous time step size, new variables
    defined on mixed function space, old variables defined on mixed function
    space, list of new variable for postprocessing, list of old varaibles for
    postprocessing, assigner for assigning values between variables, error,
    file for error output, maximal error, time stepping tolerance, minimal
    time step, list of functions that need to be updated (time dependent
    expresions and type of used approximation (by default it is LMEA)).
    '''
    if (MPI.rank(MPI.comm_world) == 0):
        print('Attempting to solve the equation for t = ' + str(t) + ' with dt = ' + str(dt.time_step), flush = True)
    try_except = False
    while try_except == False:
        try:
            t += dt.time_step # Updating time step

            # Updating time dependent expressions, if there are any.
            if isinstance(time_dependent_arguments, list) == True:
                i = 0
                while i < len(time_dependent_arguments):
                    time_dependent_arguments[i].t = t
                    i += 1

            num_vars = len(var_list_new)

            nonlinear_solver.solve(problem, u_new.vector()) # solving  the equation
            try_except = True

            assigner.assign(var_list_new, u_new) # assigning newly calculated values to post-processing variablables

            # Error estimation. Depending on the used approximation, it is determined from electron energy density, electron number density or, if nothing is specified as an argument, from all the variables solved for.
            if approximation == 'LMEA':
                error[0] = norm(var_list_new[0].vector()-var_list_old[0].vector()+DOLFIN_EPS)/norm(var_list_old[0].vector()+DOLFIN_EPS)#l2_norm(t, dt.time_step, we_newV, we_oldV)
            elif approximation == 'LFA':
                error[0] = norm(var_list_new[num_vars-2].vector()-var_list_old[num_vars-2].vector()+DOLFIN_EPS)/norm(var_list_old[num_vars-2].vector()+DOLFIN_EPS)#l2_norm(t, dt.time_step, we_newV, we_oldV)
            else:
                error[0] = norm(u_new.vector()-u_old.vector()+DOLFIN_EPS)/norm(u_old.vector()+DOLFIN_EPS)#l2_norm(t, dt.time_step, we_newV, we_oldV)

            file_error.write("{:<23}".format(str(error[0])) + '  ')
            file_error.write("{:<23}".format(str(dt_old.time_step)) + '  ' + "{:<23}".format(str(dt.time_step)) + '\n') # writting relative error to file
            file_error.flush()
            max_error[0] = max(error) # finding maximum error
            if error[0] >= ttol: # if maximum error is greater than the time stepping tolerance, the variables are reset to previous time step and calculations are repeated with the reduced time step size
                t -= dt.time_step # reseting time step to the previous time step
                dt.time_step *= (0.5*ttol/max_error[0]) # Reducing time-step size
                u_new.assign(u_old) # reseting variables to the previous time step
                assigner.assign(var_list_new, u_new) # assigning reset values to post-processing variablables
                try_except = False
                if (MPI.rank(MPI.comm_world) == 0):
                    print('Residual is greater than the prescribed tolerance. Reducing time-step size and repeating calculation. \n')
        except:
            t -= dt.time_step # reseting time step to the previous time step
            dt.time_step *= 0.5 # Reducing time-step size
            u_new.assign(u_old) # reseting variables to the previous time step
            assigner.assign(var_list_new, u_new) # assigning reset values to post-processing variablables
            try_except = False
            if (dt.time_step < dt_min):
                print('Solver failed. Reducing time-step size and repeating calculation. \n')
        if (dt.time_step < dt_min):
            sys.exit('Minimum time-step size reached, program is terminating.')
    return t # updating time step

def Normal_vector(mesh):
    W = VectorFunctionSpace(mesh, "CG", 1)

    # Projection of the normal vector on P1 space
    u = TrialFunction(W)
    v = TestFunction(W)
    n = FacetNormal(mesh)
    a = inner(u, v)*ds
    L = inner(n, v)*ds
    # Solve system
    A = assemble(a, keep_diagonal=True)
    b = assemble(L)
    A.ident_zeros()
    n = Function(W)
    solve(A, n.vector(), b, 'mumps')
    return n

def Poisson_solver(A, L, b, bcs, u, solver_type = 'mumps', preconditioner = 'hypre_amg'):
    b = assemble(L, tensor = b)
    [bc.apply(b) for bc in bcs]
    if solver_type == 'mumps':
        solve(A, u.vector(), b, solver_type)
    else:
        solve(A, u.vector(), b, solver_type, preconditioner)

def BoundaryGradient(var, zeroDomain, source_term, ds_extract, epsilon = 8.854187817e-12):
    """
    The function is an adaptation of the code snippet by D. Kamensky from https://fenicsproject.discourse.group/t/compute-gradient-of-scalar-field-on-boundarymesh/1172/2.
    It is used for the accurate calculation of the flux (in this case the electric field) across the specific boundary. Input parameters are variable whose gradient needs to be
    determined, marker of the whole domain except the boundary on which flux is calculated, the source term of the equation, list of ds of the specific boundaries
    (irrelevant boundaries should be marked as zero).
    """
    V = var.ufl_function_space()
    antiBCs = [DirichletBC(V, Constant(0.0), zeroDomain())]
    n = FacetNormal(V.mesh())

    E = TrialFunction(V)    # Electric field trial function
    v_phi = TestFunction(V)  # Potential test function
    En = Function(V)

    res = epsilon*inner(grad(var), grad(v_phi))*dx - source_term*v_phi*dx   # Poisson equation residual
    consistencyTerm = epsilon*inner(-grad(var),n)*v_phi*ds_extract[0]
    F = 0
    for i, ds in enumerate(ds_extract):
        if i == 0:
            F += -consistencyTerm
        else:
            F += epsilon*E*v_phi*ds

    F += epsilon*inner(grad(var), grad(v_phi))*dx - source_term*v_phi*dx

    a, L = lhs(F), rhs(F)

    A = assemble(a, keep_diagonal=True)
    [bc.apply(A) for bc in antiBCs]
    b = assemble(L)
    [bc.apply(b) for bc in antiBCs]
    solve(A, En.vector(), b, 'gmres', 'hypre_amg')
    return En
