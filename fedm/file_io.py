# Input and output module

import warnings
import re
import itertools
from pathlib import Path
from textwrap import dedent
from typing import List, Any

from .utils import comma_separated, print_rank_0, mesh_info

import numpy as np
import pandas as pd
import dolfin as df

# TODO This could be set via a 'verbose' argument to a solver.
level = 40  # False | 10 | 13 | 16 | 20 | 30| 40 | 50
df.set_log_level(level)

# Functions for managing input/output files

_input_dir = None
_output_dir = None
_error_file = None
_model_log = None

def input_folder_path() -> Path:
    global _input_dir
    return _input_dir if _input_dir is not None else Path.cwd() / "file_input"

def set_input_folder_path(path: Path) -> None:
    global _input_dir
    path = Path(path)
    if not path.is_dir():
        raise RuntimeError(f"fedm.set_input_folder_path: '{path}' is not a directory.")
    _input_dir = path

def output_folder_path() -> Path:
    global _output_dir
    return _output_dir if _output_dir is not None else Path.cwd() / "output"

def set_output_folder_path(path: Path) -> None:
    global _output_dir
    path = Path(path)
    if not path.is_dir():
        path.mkdir()
    _output_dir = path

def error_file() -> Path:
    global _error_file
    # If we haven't set the global _error_file yet, or the output folder path has
    # changed, create a new error file.
    if _error_file is None or _error_file.parent != output_folder_path():
        _error_file = output_folder_path() / "relative error.log"
        # truncate file
        with open(_error_file, 'w') as _:
            pass
    
    return _error_file

def model_log() -> Path:
    global _model_log
    if _model_log is None or _model_log.parent != output_folder_path():
        _model_log = output_folder_path() / "model.log"
        # truncate file
        with open(_model_log, 'w') as _:
            pass
    return _model_log


# Utilities for use in this module

def no_convert(x: Any) -> Any:
    """
    This utility function is used in functions that may optionally try to convert
    inputs to a given type. 'no_convert' may be used in place of 'float', 'str', etc
    to prevent the function from converting types.
    """
    return x


def decomment(lines: List[str]) -> str:
    """
    Removes comment at end of each line, denoted with '#'. If the line is blank, or the
    line starts with a '#', returns an empty string. Works as a generator function.
    Code snippet addapted from:
    https://stackoverflow.com/questions/14158868/python-skip-comment-lines-marked-with-in-csv-dictreader
    """
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if line:
            yield line

# file_io functions

def output_files(
    file_type: str, type_of_output: str, output_file_names: List[str]
) -> List[Any]:
    """
    Creates desired number of output files.

    Parameters
    ----------
    file_type: str
        'pvd' or 'xdmf'
    type_of_output: str
        Name of folder where files should be saved
    output_file_names
        List of files to create
    """
    if file_type not in ["pvd", "xdmf"]:
        err_msg = dedent(
            f"""
            fedm.output_files: file type '{file_type}' is not valid. Options are
            'pvd' or 'xdmf'.
            """
        )
        raise ValueError(err_msg.rstrip().replace("\n", " "))

    FileType = df.File if file_type == "pvd" else df.XDMFFile
    output_dir = output_folder_path() / type_of_output

    file_paths = [
        FileType(str(output_dir / file_name / f"{file_name}.{file_type}"))
        for file_name in output_file_names
    ]

    if file_type == "xdmf":
        for f in file_paths:
            f.parameters["flush_output"] = True

    return file_paths


def read_single_value(file_name, convert=no_convert):
    """
    Reads file containing only constant.
    Input parameter is file name.
    """
    with open(file_name, "r", encoding="utf8") as f_input:
        # Finds first non-comment and non-blank line, passes through convert function
        return convert(next(decomment(f_input)))
    raise RuntimeError(f"fedm.read_single_value: No value found in file '{file_name}'")


def read_single_float(file_name, convert=no_convert):
    """
    Reads file containing only constant.
    Input parameter is file name.
    """
    return read_single_value(file_name, convert=float)


def read_single_string(file_name):
    """
    Reads file containing one column.
    Input parameter is file name.
    """
    return read_single_value(file_name, convert=str)


def read_and_decomment(file_name: str) -> List[str]:
    """
    Reads file, returns list of strings. Comment lines and blank lines are removed.
    """
    with open(file_name, "r") as f_input:
        return [line for line in decomment(f_input)]


def read_two_columns(file_name):
    """
    Reads two column files.
    Input parameter is file name.
    """
    data = pd.read_csv(file_name, comment="#", header=None, sep=r"\s+", dtype=float)
    # TODO test if we can avoid list conversions
    return list(data[0]), list(data[1])


def flatten(input_list: List[List[Any]]) -> List[Any]:
    """
    Reduces 2D list to 1D list.
    """
    return itertools.chain.from_iterable(input_list)


def flatten_float(input_list: List[List[Any]]) -> List[float]:
    """
    Reduces 2D list to 1D list and converts elements to float.
    """
    return [float(x) for x in flatten(input_list)]


def read_speclist(file_path):
    """
    Function reads list of species from 'speclist.cfg' file
    and extracts species names and species properties filename.
    """

    file_name = Path(file_path) / "speclist.cfg"

    # Get all lines from the file which have 'file:' in them
    # Remove the "file:" from the line, and split by whitespace
    lines = [line for line in read_and_decomment(file_name) if "file:" in line]

    # remove "file:" from each line and split by whitespace
    lines = [line.replace("file:", "").split() for line in lines]

    # Get data to return
    species_names = [line[0] for line in lines]
    species_properties_file_names = [line[1] for line in lines]
    species_name_tc = [line[1].split(".")[0] for line in lines]
    p_num = len(species_names)

    return p_num, species_names, species_properties_file_names, species_name_tc


def reaction_matrices(path: str, species: List[str]):
    """
    Reads the reaction scheme from "reacscheme.cfg" file and creates power, loss and
    gain matrices.
    """

    file_name = Path(path) / "reacscheme.cfg"

    reactions = [line.partition(" Type:")[0] for line in read_and_decomment(file_name)]
    loss = [reaction.partition(" -> ")[0].rstrip() for reaction in reactions]
    gain = [reaction.partition(" -> ")[2].rstrip() for reaction in reactions]

    l_matrix = np.empty([len(reactions), len(species)], dtype=int)
    g_matrix = np.empty([len(reactions), len(species)], dtype=int)
    for i, j in itertools.product(range(len(reactions)), range(len(species))):
        l_matrix[i, j] = loss[i].count(species[j])
        g_matrix[i, j] = gain[i].count(species[j])

    power_matrix = l_matrix
    lg_matrix = l_matrix - g_matrix
    loss_matrix = np.where(lg_matrix > 0, lg_matrix, 0)
    gain_matrix = np.where(lg_matrix < 0, -lg_matrix, 0)

    return power_matrix, loss_matrix, gain_matrix


def rate_coefficient_file_names(path):
    """
    Reads names of reaction coefficient files from "reacscheme.cfg" file.
    Input parameter is the path to the folder.
    """

    file_name = Path(path) / "reacscheme.cfg"
    rate_coefficient_dir_path = Path(path) / "rate_coefficients"

    lines = read_and_decomment(file_name)
    regex = re.compile(r"kfile: ([A-Za-z0-9_]+.[A-Za-z0-9_]+)")
    rcfns = flatten([regex.findall(line) for line in lines])
    return [rate_coefficient_dir_path / rcfn for rcfn in rcfns]


def energy_loss(path):
    """
    Reads energy loss values from "reacscheme.cfg" file.
    Input argument is the path to the folder.
    """

    file_name = Path(path) / "reacscheme.cfg"
    lines = read_and_decomment(file_name)

    regex = re.compile(r"Uin:\s?([+-]?\d+.\d+[eE]?[-+]?\d+|0|1.0)")
    energy_loss_value = flatten([regex.findall(line) for line in lines])
    energy_loss_value = [float(x) for x in energy_loss_value]

    print_rank_0(energy_loss_value)
    return energy_loss_value


def read_dependence(file_name: str):
    """
    Reads dependence of rate coefficients from the corresponding file.
    """
    with open(file_name, "r", encoding="utf8") as f_input:
        for line in f_input:
            if "Dependence:" in line:
                return line.split()[2]
    raise RuntimeError(
        f"fedm.read_dependence: Did not find dependence in file '{file_name}'"
    )


def dependence(file_names: List[str]):
    """
    Reads dependence of rate coefficients from a list of corresponding files.
    """
    return [read_dependence(file_name) for file_name in file_names]


def read_rate_coefficients(rc_file_names: List[str], k_dependences: List[str]):
    """
    Reading rate coefficients from files. Input
    are file names and dependences.
    """
    if len(rc_file_names) != len(k_dependences):
        raise ValueError(
            "fedm.read_rate_coefficients: rc_file_names and k_dependences should be "
            "the same length."
        )

    float_deps = ["const"]
    str_deps = ["fun:Te,Tgas", "fun:Tgas"]
    two_col_deps = ["Umean", "E/N", "ElecDist"]
    all_deps = float_deps + str_deps + two_col_deps
    for dependence in k_dependences:
        if dependence not in all_deps:
            raise ValueError(
                f"fedm.read_rate_coefficients: The dependence '{dependence}' is not "
                f"recognised. Options are {comma_separated(all_deps)}."
            )

    k_xs, k_ys = [], []
    for dependence, rc_file_name in zip(k_dependences, rc_file_names):
        print_rank_0(rc_file_name)
        if dependence in two_col_deps:
            k_x, k_y = read_two_columns(rc_file_name)
        else:
            convert = float if dependence in float_deps else str
            k_x, k_y = 0.0, read_single_value(rc_file_name, convert=convert)
        k_xs.append(k_x)
        k_ys.append(k_y)
    return k_xs, k_ys


def read_transport_coefficients(
    particle_names: List[str], transport_type: str, model: str
):
    """
    Reading transport coefricients from files. Input are
    particle names, type of transport coefficient (diffusion or mobility)
    and model.
    """
    path = input_folder_path() / model / "transport_coefficients"
    if not path.is_dir():
        raise RuntimeError(
            f"fedm.read_transport_coefficients: Transport coeff dir '{path}' not found."
        )

    float_deps = ["const"]
    str_deps = ["fun:Te,Tgas", "fun:E"]
    two_col_deps = ["Umean", "E/N", "Tgas", "Te"]
    dependences = float_deps + str_deps + two_col_deps
    if transport_type == "Diffusion":
        dependences += "ESR"

    k_xs = []
    k_ys = []
    k_dependence = []
    for particle in particle_names:
        # Get file name
        file_suffix = "_ND.dat" if transport_type == "Diffusion" else "_Nb.dat"
        file_name = path / (particle + file_suffix)
        if not file_name.is_file():
            raise RuntimeError(
                f"fedm.read_transport_coefficients: file '{file_name}' not found."
            )
        print_rank_0(file_name)

        # Get dependence from file
        dependence = read_dependence(file_name)

        if dependence == "const.":
            warnings.warn(
                "fedm.read_transport_coefficients: 'const' dependence should be "
                "written 'const', not 'const.'"
            )
            dependence = "const"

        if dependence not in dependences:
            err_msg = dedent(
                f"""\
                fedm.read_transport_coefficients: Dependence '{dependence}' not
                recognised. For the transport type '{transport_type}', the possible
                options are {comma_separated(dependences)}.
                """
            )
            raise RuntimeError(err_msg.rstrip().replace("\n", " "))

        # Set kx and ky
        if dependence in two_col_deps:
            k_x, k_y = read_two_columns(file_name)
        elif dependence == "ESR":
            k_x, k_y = 0.0, 0.0
        else:
            convert = float if dependence in float_deps else str
            k_x, k_y = 0.0, read_single_value(file_name, convert=convert)

        if dependence == "fun:Te,Tgas":
            try:
                k_y_str = k_y  # save reference in case we need it in an error message
                k_y = eval(k_y)
            except Exception as exc:
                raise RuntimeError(
                    f"fedm.read_transport_coefficients: k_y eval failed, '{k_y_str}'"
                ) from exc

        k_xs.append(k_x)
        k_ys.append(k_y)
        k_dependence.append(dependence)

    return k_xs, k_ys, k_dependence


def read_particle_properties(file_names: List[str], model: str):
    """
    Reads particle properties from input file.
    Input are file names and model.
    """
    path = input_folder_path() / model / "species"

    masses, charges = [], []
    regex_mass = re.compile(r"Mass\s?=\s?([+-]?\d+.\d+[eE]?[-+]?\d+|0|1.0)")
    regex_charge = re.compile(r"Z\s+?=\s+?([+-]?\d+)")

    for file_name in file_names:
        # Get full file name, test if file exists
        file_name = path / file_name
        if not file_name.is_file():
            raise RuntimeError(
                f"fedm.read_particle_properties: File '{file_name}' not found."
            )
        print_rank_0(file_name)

        # Read file
        lines = read_and_decomment(file_name)

        # Get mass and charge from file
        mass_found, charge_found = False, False
        for line in lines:
            print_rank_0(line)
            mass, charge = regex_mass.findall(line), regex_charge.findall(line)
            if mass:
                mass_found = True
                masses.append(float(mass[0]))
            if charge:
                charge_found = True
                charges.append(float(charge[0]))
        if not mass_found:
            raise RuntimeError(
                f"fedm.read_particle_properties: No mass found in file '{file_name}'."
            )
        if not charge_found:
            raise RuntimeError(
                f"fedm.read_particle_properties: No charge found in file '{file_name}'."
            )

    return masses, charges


def print_time_step(dt):
    """
    Prints time step.
    """
    print_rank_0("Time step is dt =", dt)


def print_time(t):
    """
    Prints time.
    """
    print_rank_0("t =", t)


def file_output(
    t,
    t_old,
    t_out,
    step,
    t_out_list,
    step_list,
    file_type,
    output_file_list,
    particle_name,
    u_old,
    u_old1,
    unit="s",
):
    """
    Writing value of desired variable to file. Value is calculated by linear
    interpolation.  Input arguments are current time step, previous time step, current
    time step length, previous time step length, list of output times, list of steps
    (which needs have same size), list of output files, list of variable values in
    current and previous time steps, and (optional) units of time.
    """
    units = {
        "ns": 1e9,
        "us": 1e6,
        "ms": 1e3,
        "s": 1.0,
    }

    try:
        scale = units[unit]
    except KeyError:
        err_msg = dedent(
            f"""\
            fedm.file_output: unit '{unit}' not valid.
            Options are {comma_separated(list(units))}.'
            """
        )
        raise ValueError(err_msg.rstrip().replace("\n", " "))

    if t > max(t_out_list):
        index = len(t_out_list) - 1
    else:
        index = next(x for x, val in enumerate(t_out_list) if val > t)


    while t_out <= t:
        for i in range(len(output_file_list)):
            temp = df.Function(u_old1[i].function_space())
            temp.vector()[:] = u_old1[i].vector()[:] + (t_out - t_old) * (
                u_old[i].vector()[:] - u_old1[i].vector()[:]
            ) / (t - t_old)
            if file_type[i] == "pvd":
                temp.rename(particle_name[i], str(i))
                # TODO find functional version of this, avoid C++ operator overloads
                output_file_list[i] << (temp, t_out * scale)
            elif file_type[i] == "xdmf":
                # appending to file
                output_file_list[i].write_checkpoint(
                    temp,
                    particle_name[i],
                    t_out * scale,
                    df.XDMFFile.Encoding.HDF5,
                    True,
                )
            else:
                err_msg = dedent(
                    f"""\
                    fedm.file_output: file type '{file_type}' not recognised.
                    Options are 'pvd' and 'xdmf'.
                    """
                )
                raise ValueError(err_msg.rstrip().replace("\n", " "))

        if t_out >= 0.999 * t_out_list[index - 1] and t_out < 0.999 * t_out_list[index]:
            step = step_list[index - 1]
        elif t_out >= 0.999*t_out_list[index]:
            step = step_list[index]
        # FIXME undefined if t_out < 0.999 * t_out_list[index - 1], need else statement
        t_out += step
    return t_out, step


def mesh_statistics(mesh: df.Mesh) -> None:
    """
    Returns mesh size and, maximum and minimum element size.
    Input is mesh.
    """
    mesh_dir = output_folder_path() / "mesh"
    vtkfile_mesh = df.File(str(mesh_dir / "mesh.pvd"))
    vtkfile_mesh.write(mesh)
    info_str = mesh_info(mesh)
    if df.MPI.rank(df.MPI.comm_world) == 0:
        print(info_str.rstrip())
        with open(mesh_dir / "mesh info.txt", "w") as mesh_information:
            mesh_information.write(info_str)


def numpy_2d_array_to_str(x: np.ndarray) -> str:
    # Remove '[' and  ']' from str representation
    no_brackets = str(np.asarray(x)).replace("[", "").replace("]", "")
    # Remove extra whitespace on each line, return
    return "\n".join([y.strip() for y in no_brackets.split("\n")])


def log(log_type, log_file_name, *args):
    """
    The function is used to log model data and its progress.
    The input arguments vary depending on the type of logging.
    For type = properties, input are gas, model, particle species
    names, particle mass, and charge.
    For type = conditions, input arguments are time step length,
    working voltage, pressure, gap length, gas number density,
    and gas temperature.
    For type = matrices, input arguments are gain, loss and power
    matrices.
    For type = initial_time, input argument is time.
    For type = time, input argument is time.
    For type = mesh, input argument is mesh.
    """

    if df.MPI.rank(df.MPI.comm_world) != 0:
        return

    if log_type == "properties":
        gas, model, particle_species_file_names, M, charge = args
        log_str = dedent(
            f"""\
            Gas:\t{gas}

            model:\t{model}

            Particle names:
            {particle_species_file_names}

            Mass:
            {M}

            Charge:
            {charge}
            """
        )
    elif log_type == "conditions":
        dt_var, U_w, p0, gap_length, N0, Tgas = args
        log_str = dedent(
            f"""\
            dt = {dt_var} s,
            U_w = {U_w} V,
            p_0 = {p0} Torr,
            d = {gap_length} m,
            N_0 = {N0} m^-3,
            T_gas = {Tgas} K
            """
        )
        log_str = log_str.rstrip().replace("\n", "\t ")
        log_str = f"Simulation conditions:\n{log_str}\n"
    elif log_type == "matrices":
        gain, loss, power = args
        log_str = dedent(
            f"""\
            Gain matrix:
            {numpy_2d_array_to_str(gain)}

            Loss matrix:
            {numpy_2d_array_to_str(loss)}

            Power matrix:
            {numpy_2d_array_to_str(power)}
            """
        )
    elif log_type == "initial time":
        log_str = f"Time:\n{args[0]}"
    elif log_type == "time":
        log_str = str(args[0])
    elif log_type == "mesh":
        log_str = mesh_info(args[0])
    else:
        err_msg = dedent(
            f"""\
            fedm.log: log_type '{log_type}' not recognised. Options are 'properties',
            'conditions', 'matrices', 'initial time', 'time', or 'mesh'
            """
        )
        raise ValueError(err_msg.rstrip().replace("\n", " "))

    with open(log_file_name, "a") as log_file:
        log_file.write(log_str)
        log_file.write("\n")
        log_file.flush()
