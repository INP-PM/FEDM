# Fluid modelling functions module

import dolfin as df
from dolfin import *
from typing import List, Tuple, Any, Optional
from pathlib import Path
import numpy as np
from numpy import pi
import sys

from .physical_constants import elementary_charge, kB, kB_eV

def quoted(strings: List[str]) -> List[str]:
    """
    Utility function, takes a list of strings and returns the same list with each
    string starting and ending with a single-quote character.
    """
    return [f"'{string}'" for string in strings]

def modify_approximation_vars(
    approximation_type: str,
    number_of_species: int,
    particle_species: List[str],
    masses: List[float],
    charges: List[float],
) -> Tuple[int, int, List[str], List[float], List[float]]:
    """
    Depending on approximation used, the number of equations, charge and mass variables
    are modified. Returns number species, number of equations, particle species,
    masses, and charges. particle_species, masses, and charges may be modified.
    """
    approximation_types = ["LFA", "LMEA"]
    if approximation_type not in approximation_types:
        raise ValueError(
            "fedm.modify_approximation_vars: The approximation type "
            f"'{approximation_type}' is not recognised. Must be one of "
            f"{', '.join(quoted(approximation_types))}."
        )
    # IF LFA, remove the first species from each list
    if approximation_type == 'LFA':
        number_of_species -= 1
        particle_species.pop(0)
        masses.pop(0)
        charges.pop(0)
    # Number of equations should be 1 more than the number of species in each case
    number_of_eq = number_of_species + 1
    return number_of_species, number_of_eq, particle_species, masses, charges


def mesh_statistics(mesh: df.Mesh) -> None:
    """
    Returns mesh size and, maximum and minimum element size.
    Input is mesh.
    """
    mesh_dir = Path("output/mesh")
    vtkfile_mesh = df.File(str(mesh_dir / "mesh.pvd"))
    vtkfile_mesh.write(mesh)
    n_element = MPI.sum(MPI.comm_world, mesh.num_cells())
    #measures the greatest distance between any two vertices of a cell
    hmax = MPI.max(MPI.comm_world, mesh.hmax())
    #measures the smallest distance between any two vertices of a cell
    hmin = MPI.min(MPI.comm_world, mesh.hmin())
    if(MPI.rank(MPI.comm_world)==0):
        info_str = (
            f"Number of elements is: {int(n_element)}\n"
            f"Maximum element edge length is: {hmax:.5g}\n"
            f"Minimum element edge length is: {hmin:.5g}"
        )
        print(info_str)
        with open(mesh_dir / "mesh info.txt",'w') as mesh_information:
            mesh_information.write(info_str + '\n')

class CircleSubDomain(df.SubDomain):

    def __init__(
        self,
        center_z: float,
        center_r: float,
        radius: float,
        gap_length: float,
        submesh: bool = False,
        tol : float = 1e-8
    ):
        super().__init__()
        self._center_z = float(center_z)
        self._center_r = float(center_r)
        self._radius = float(radius)
        self._submesh = bool(submesh)
        self._tol = float(tol)

    def inside(self, x: List[float], on_boundary: bool) -> bool:
        r, z = x[0], x[1]
        dist_from_center_squared = (r-self._center_r)**2 + (z-self._center_z)**2
        within_tol = abs(dist_from_center_squared - self._radius**2) <= self._tol
        z_correct = z <= 0 if center_z <= 0 else z >= self._gap_length
        return within_tol and z_correct and (on_boundary or self._submesh)


class LineSubDomain(df.SubDomain):

    def __init__(self, r_range: Tuple[float,float], z_range: Tuple[float,float]):
        super().__init__()
        self._r_range = r_range
        self._z_range = z_range

    def inside(self, x: List[float], on_boundary: bool) -> bool:
        r, z = x[0], x[1]
        in_r_range = df.between(r, self._r_range)
        in_z_range = df.between(z, self._z_range)
        return in_r_range and in_z_range and on_boundary

def Marking_boundaries(
    mesh: df.Mesh,
    boundaries: List[List[Any]],
    submesh: bool = False,
    gap_length: float = 0.01
) -> df.MeshFunction:
    """
    Marking boundaries of a provided mesh. Currently, straight-line and circular
    boundaries are supported. First argument is the mesh, the second argument is a list
    of boundary properties (boundary type and coordinates).
    """

    boundary_markers = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    for idx, boundary in enumerate(boundaries):
        boundary_type = boundary[0]

        if MPI.rank(MPI.comm_world) == 0:
            print(boundary_type)

        if boundary_type == 'circle':
            center_z, center_r, radius = boundary[1:4]
            bmark = CircleSubDomain(center_z, center_r, radius, gap_length, submesh)
        elif boundary_type == 'line':
            eps = df.DOLFIN_EPS
            z1, z2 = boundary[1] - eps, boundary[2] + eps
            r1, r2 = boundary[3] - eps, boundary[4] + eps
            bmark = LineSubDomain((r1, r2), (z1, z2))
        else:
            raise ValueError(
                f"fedm.Marking_boundaries: Invalid boundary_type '{boundary_type}'. "
                "Possible values are 'circle', 'line'."
            )

        bmark.mark(boundary_markers, idx)

    return boundary_markers

def Mixed_element_list(number_of_equations: int, element: df.FiniteElement) -> List[df.FiniteElement]:
    """
    Defines list of mixed elements. Input arguments are number of equations and element
    type.
    """
    # WARNING: All elements of the list refer to the same object!
    return [element] * number_of_equations

def Function_space_list(number_of_equations: int, function_space: df.FunctionSpace) -> List[df.FunctionSpace]:
    """
    Defines list of function spaces. Input arguments are number of equations and
    function space.
    """
    # WARNING: All elements of the list refer to the same object!
    return [function_space] * number_of_equations

def Function_definition(
    function_space: df.FunctionSpace,
    function_type: str, 
    eq_number: int = 1
) -> List[Any]:
    """
    Defines list of desired function type (TrialFunction, TestFunction or ordinary Function).
    Input arguments are function space, type of desired function and number of equations,
    where the default value is one.
    """
    functions = {
        "TrialFunction" : df.TrialFunction,
        "TestFunction" : df.TestFunction,
        "Function" : df.Function,
    }
    if function_type not in functions:
        raise ValueError(
            f"fedm.Function_definition: Invalid function_type '{function_type}'. "
            f"Possible values are {', '.join(quoted(Functions))}."
        )
    function = functions[function_type]
    return [function(function_space) for _ in range(eq_number)]


class Problem(df.NonlinearProblem):
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
        df.NonlinearProblem.__init__(self)

    def F(self, b, x):
        """
        Linear form assembly
        """
        df.assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        """
        Bilinear form assembly
        """
        df.assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)

def Max(a, b):
    """
    Returns maximum value of a and b.
    """
    return (a + b + abs(a - b)) / df.Constant(2.0)

def Min(a, b):
    """
    Returns minimum value of a and b.
    """
    return (a + b - abs(a - b)) / df.Constant(2.0)

def Flux(sign, u, D, mu, E):
    """
    Defines particle flux using drift-diffusion approximation.
    Input arguments are particle charge, number density,
    diffusion coefficient, mobility and electric field.
    """
    return -df.grad(D * u) + sign * mu * E * u

def Flux_log(sign, u, D, mu, E):
    """
    Defines particle flux using drift-diffusion and logarithmic approximation.
    Input arguments are particle charge, number density,
    Diffusion coefficient, mobility and electric field.
    """
    return -df.grad(D * df.exp(u)) + sign * mu * E * df.exp(u)


def weak_form_balance_equation(
    equation_type: str, 
    dt: df.Expression, 
    dt_old: df.Expression,
    dx: df.Measure,  # Can use built-in dolfin.dx here
    u: Any, # obtain by indexing result of df.TrialFunction
    u_old: Any, # obtain by indexing result of df.Function
    u_old1: Any, # obtain by indexing result of df.Function
    v: Any, # obtain by indexing result of df.TestFunctions
    f: df.Function, # obtain by indexing result of df.Function_definition
    Gamma: df.Function, # obtain by indexing result of df.Function_definition
    r: float = 0.5 / df.pi,
    D: Optional[df.Function] = None, # get by indexing result of df.Function_definition
    log_representation: bool = False,
) -> df.Form:
    """
    Returns the weak form of the particle balance equations.

    Parameters
    ----------
    equation_type : str
        Type of equation to solve. Options are 'reaction', 'diffusion-reaction', or
        'drift-diffusion-reaction'.
    dt : df.Expression
        Current time-step size
    dt_old : df.Expression
        Previous time-step size
    dx : df.Measure
        dV, used to build integrals. Recommended to use dolfin.dx.
    u
        Trial function
    u_old
        Value of variable in current time step
    u_old1
        Value of variable in previous time step
    v
        Test function
    f : df.Function
        Source term
    Gamma : df.Function
        particle flux
    r : float, default 0.5 / pi
        r coordinate
    D : df.Function, default None
        diffusion coefficient, only required for the diffusion equation.
    log_representation : bool, default False
        Use logarithmic representation.

    Returns
    -------
    df.Form

    Raises
    ------
    ValueError
        If equation_type is not recognised, or if D is not supplied when solving the
        diffusion-reaction equation.
    """
    equation_types = ["reaction", "diffusion-reaction", "drift-diffusion-reaction"]
    if equation_type not in equation_types:
        raise ValueError(
            "fedm.weak_form_balance_equation_log_representation: The equation type "
            f"'{equation_type}' is not recognised. Must be one of "
            f"{', '.join(quoted(equation_types))}."
        )
    if equation_type == 'diffusion-reaction' and D is None:
        raise ValueError(
            "fedm.weak_form_balance_equation_log_representation: When 'equation_type' "
            "is diffusion-reaction, must also supply the diffusion coefficient 'D'."
        )
    # tr = timestep_ratio
    tr = dt / dt_old
    trp1 = tr + 1.0
    tr2p1 = 2.0 * tr + 1.0
    # If logarithmic, we include a factor of exp(u) in the integral
    expu_or_1 = df.exp(u) if log_representation else 1.0
    # Standard part
    result = v * expu_or_1 * (
        ( u * tr2p1 - u_old * trp1**2.0 + u_old1 * tr**2.0) / (trp1 * dt)
    )
    # Source terms
    result -= v * f
    # Diffusion terms
    if equation_type == "diffusion-reaction":
        expu_or_u = df.exp(u) if log_representation else u
        result -= df.dot(-df.grad(D * expu_or_u), df.grad(v))
    if equation_type == "drift-diffusion-reaction":
        result -= df.dot(Gamma, df.grad(v))
    # Return with integral bits
    return 2.0 * df.pi * r * result * dx

def weak_form_balance_equation_log_representation(*args, **kwargs) -> df.Form:
    """
    Convenience function, calls weak_form_balance_equation with log_representation set
    to True.
    """
    return weak_form_balance_equation(*args, **kwargs, log_representation=True)

def weak_form_Poisson_equation(dx, u, v, f, r = 0.5 / df.pi) -> df.Form:
    """
    Returns a weak form of Poisson equation. Input arguments are dV, trial function,
    test function, source term and r coordinate.

    Parameters
    ----------
    dx : df.Measure
        dV, used to build integrals. Recommended to use dolfin.dx.
    u
        Trial function
    v
        Test function
    f
        Source term
    r
        r coordinate

    Returns
    -------
    df.Form
    """
    return 2.0 * df.pi * r * (df.inner(df.grad(u), df.grad(v)) - f * v) * dx

def Boundary_flux(
    bc_type: str,
    equation_type: str,
    particle_type: str,
    sign: float,
    mu, # TODO types unknown
    E,
    normal,
    u,
    gamma,
    v,
    ds_temp,
    r: float = 0.5/pi,
    vth: float = 0.0,
    ref: float = 1.0,
    Ion_flux: float = 0.0,
):
    """
    Function defines boundary conditions for different equations.

    Parameters
    ----------
    bc_type : str
        Type of boundary condition. Options are 'zero flux', 'flux source' or 'Neumann'
    equation_type : str
        Choices are 'reaction', 'diffusion-reaction' or 'drift-diffusion-reaction'
    particle_type : str
        Choices are 'Heavy' or 'electrons'.
    sign: float
        Particle charge sign
    mu
        Mobility
    E
        Electric field
    normal
        Normal
    u
        Trial function
    gamma
        Secondary electron emission coefficient
    v
        Test function
    ds_temp : df.Measure
        ds, surface element used to build integrals (??)
    r: float, default 0.5/pi
        r coordinate
    vth: float, default 0.0
        Thermal velocity
    ref: float, default 1.0
        Reflection coefficient for specified particles species and boundary.
    Ion_flux: float, default 0.0
        Flux of ions

    Returns
    -------
    df.Form
        If bc_type is 'flux source' and there is a diffusion term in equation_type.
        Also if the user combines Neumann boundaries with a drift-diffusion-reaction
        equation.
    float
        Otherwise.

    Raises
    ------
    ValueError
        If bc_type is not recognised. If bc_type is not 'zero flux', also raised if
        equation_type is not recognised. Furthermore, if equation_type is
        'drift-diffusion-reaction', raises if particle_type is not recognised.
    """
    bc_types = ["zero flux", "flux source", "Neumann"]
    equation_types = ["reaction", "diffusion-reaction", "drift-diffusion-reaction"]
    particle_types = ["Heavy", "electrons"]

    # If provided bc_type has an underscore instead of a space, correct it
    bc_type = bc_type.replace('_', ' ')

    if bc_type not in bc_types:
        raise ValueError(
            f"fedm.Boundary_flux: boundary condition type '{bc_type}' not recognised. "
            f"Must be one of {', '.join(quoted(bc_types))}."
        )

    # Only raise error on bad 'equation_type' if the value is needed
    if bc_type != "zero flux" and equation_type not in equation_types:
        raise ValueError(
            f"fedm.Boundary_flux: equation type '{equation_type}' not recognised. "
            f"Must be one of {', '.join(quoted(equation_types))}."
        )

    # Only raise error on bad 'particle_type' if the value is needed
    if bc_type == "flux source" and equation_type == "diffusion-reaction" and particle_type not in particle_types:
        raise ValueError(
            f"fedm.Boundary_flux: particle type '{particle_type}' not recognised. "
            f"Must be one of {', '.join(quoted(particle_types))}."
        )

    if bc_type == 'flux source' and equation_type != 'reaction':
        result = ((1.0 - ref) / (1.0 + ref))
        if equation_type == 'diffusion-reaction':
            result *= 0.5 * vth * df.exp(u)
        if equation_type == 'drift-diffusion-reaction' and particle_type == 'Heavy':
            result *= (0.5 * vth + abs(sign * mu * df.dot(E, normal)))* df.exp(u)
        if equation_type == 'drift-diffusion-reaction' and particle_type == 'electrons':
            result *= (0.5 * vth + abs(mu * df.dot(E, normal)))* df.exp(u)
            result -= 2.0 * gamma * Ion_flux / (1.0 + ref)
        return 2.0 * df.pi * result * v * r * ds_temp

    if bc_type == 'Neumann' and equation_type == 'drift-diffusion-reaction':
        # Note: Here we use the built-in dolfin.ds, not the ds_temp passed in.
        return 2.0 * df.pi * df.dot(sign * mu * E, normal) * df.exp(u) * v * r * df.ds

    # default, returned if bc_type is 'zero_flux', or if no other conditions are met.
    return 0.0

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
