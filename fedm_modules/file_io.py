# Input and output module

from dolfin import *
import numpy as np
import csv
import re
import os
import gc
import sys

level = 40 #False | 10 | 13 | 16 | 20 | 30| 40 | 50
set_log_level(level)

output_folder_path = os.path.join(os.getcwd(), 'output')
os.makedirs(output_folder_path, exist_ok=True)
file_error = open('output/relative error.log', 'w')
model_logging = open('output/model.log', 'w')

def output_files(file_type, Type_of_output, output_file_names):
    """
    Creates desired number of output files.
    Input parameters are Type of output, which is used
    as the name of folder where the files are saved, and
    output_file_names which are used to create subfolder
    and vtk files of the same name.
    """
    number_of_output_files = len(output_file_names)
    if file_type == 'pvd':
        f_out = []
        i = 0
        while (i < number_of_output_files):
            file_path = 'output/' + Type_of_output + '/' + output_file_names[i] + '/' + output_file_names[i] + '.pvd'
            f_out.append(File(file_path))
            i += 1
    elif file_type == 'xdmf':
        f_out = []
        i = 0
        while (i < number_of_output_files):
            file_path = 'output/' + Type_of_output + '/' + output_file_names[i] + '/' + output_file_names[i] + '.xdmf'
            f_out.append(XDMFFile(file_path))
            f_out[i].parameters['flush_output'] = True
            i += 1
    return f_out

def decomment(csvfile):
    """
    Removes comments from input files.
    csvfile is the file name
    Code snippet addapted from:
    https://stackoverflow.com/questions/14158868/python-skip-comment-lines-marked-with-in-csv-dictreader
    """
    for row in csvfile:
        raw = row.split('#')[0].strip()
        if raw: yield raw

def Num_of_columns(file_name):
    """
    Returns number of columns.
    Input parameter is file name.
    """
    with open(file_name, 'r', encoding = 'utf8') as f_input:
        csv_input = csv.reader(decomment(f_input), delimiter=' ', skipinitialspace=True)
        n = 0
        for row in csv_input:
            n += 1
        return n

def Read_One_Column_File(file_name):
    """
    Reads one column file.
    Input parameter is file name.
    """
    with open(file_name, 'r', encoding = 'utf8') as f_input:
        csv_input = csv.reader(decomment(f_input), delimiter=' ', skipinitialspace=True)
        y = []
        for cols in csv_input:
            y.append(float(cols[0]))
    return y

def Read_constant(file_name):
    """
    Reads file containing only constant.
    Input parameter is file name.
    """
    with open(file_name, 'r', encoding = 'utf8') as f_input:
        for line in f_input:
            line = line.split('#',1)[0].strip()
            if not line:
                continue
            y = line
    return float(y)

def Read_One_Column_String_File(file_name):
    """
    Reads file containing one column.
    Input parameter is file name.
    """
    with open(file_name, 'r', encoding = 'utf8') as f_input:
        for line in f_input:
            line = line.split('#',1)[0].strip()
            if not line:
                continue
            y = line
    return y

def Read_Two_Column_File(file_name):
    """
    Reads two column files.
    Input parameter is file name.
    """
    with open(file_name, 'r', encoding = 'utf8') as f_input:
        csv_input = csv.reader(decomment(f_input), delimiter=' ', skipinitialspace=True)
        x = []
        y = []
        i = 0
        for cols in csv_input:
            x.append(float(cols[0]))
            y.append(float(cols[1]))
    return x, y

def flatten(input):
    """
    Reduces 2D list to 1D list.

    """
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)

    return new_list

def flatten_float(input):
    """
    Reduces 2D list to 1D list and converts elements to float.
    """
    new_list = []
    for i in input:
        for j in i:
            new_list.append(float(j))
    return new_list

def read_speclist(file_path):
    """
    Function reads list of species from 'speclist.cfg' file
    and extracts species names and species properties filename.
    """
    lin = []
    x = []

    file_name = file_path + '/speclist.cfg'
    with open(file_name, 'r') as f_input:
        for line in f_input:
            line = line.split('#',1)[0].strip()
            if not line:
                continue
            if "file:" in line:
                lin.append(line)

    i = 0
    while i < len(lin):
        lin[i] = lin[i].replace("file:","")
        x.append(lin[i].split())
        i += 1

    y = []
    i = 0
    while i < len(x):
        y.append(x[i][1].split('.'))
        i += 1

    species_name = []
    species_properties_file_name = []
    species_name_tc = []

    i = 0
    while i < len(x):
        species_name.append(x[i][0])
        species_properties_file_name.append(x[i][1])
        species_name_tc.append(y[i][0])
        i += 1

    p_num = len(species_name)
    return p_num, species_name, species_properties_file_name, species_name_tc

def reaction_matrices(path, species):
    """
    Reads the reaction scheme from "reacscheme.cfg" file
    and creates power, loss and gain matrices.
    """

    file_name = path + '/reacscheme.cfg'

    lin = []

    with open(file_name, 'r') as f_input:
        for line in f_input:
            line = line.split('#',1)[0].strip()
            if not line:
                continue
            lin.append(line)

    reaction = []
    rest = []
    loss = []
    gain = []

    i = 0
    while i < len(lin):
        reaction.append(lin[i].partition(' Type:')[0])
        rest.append(lin[i].partition(' Type:')[2])
        loss.append(reaction[i].partition(' -> ')[0])
        gain.append(reaction[i].partition(' -> ')[2])
        loss[i] = loss[i].rstrip()
        gain[i] = gain[i].rstrip()
        i += 1

    power_matrix = np.zeros([len(reaction), len(species)], dtype = int)
    loss_matrix = np.zeros([len(reaction), len(species)], dtype = int)
    gain_matrix = np.zeros([len(reaction), len(species)], dtype = int)
    l_matrix = np.zeros([len(reaction), len(species)], dtype = int)
    g_matrix = np.zeros([len(reaction), len(species)], dtype = int)
    lg_matrix = np.zeros([len(reaction), len(species)], dtype = int)

    i = 0
    while i < len(reaction):
        j = 0
        while j < len(species):
            power_matrix[i, j] = power_matrix[i, j] + loss[i].count(species[j])
            l_matrix[i, j] = l_matrix[i, j] + int(loss[i].count(species[j]))
            g_matrix[i, j] = g_matrix[i, j] + int(gain[i].count(species[j]))
            j += 1
        i += 1

    lg_matrix = l_matrix - g_matrix

    i = 0
    while i < len(reaction):
        j = 0
        while j < len(species):
            loss_matrix[i, j] = max(lg_matrix[i, j], 0)
            gain_matrix[i, j] = -min(lg_matrix[i, j], 0)
            j += 1
        i += 1
    return power_matrix, loss_matrix, gain_matrix

def rate_coefficient_file_names(path):
    """
    Reads names of reaction coefficient files from "reacscheme.cfg" file.
    Input parameter is the path to the folder.
    """
    lin = []

    file_name = path + '/reacscheme.cfg'
    rate_coefficient_folder_path = path + '/rate_coefficients/'
    with open(file_name, 'r') as f_input:
        for line in f_input:
            line = line.split('#',1)[0].strip()
            if not line:
                continue
            lin.append(line)

    rcfn = []
    regex = re.compile(r'kfile: ([A-Za-z0-9_]+.[A-Za-z0-9_]+)')

    i = 0
    while i < len(lin):
        rcfn.append(regex.findall(lin[i]))
        i += 1
    rcfn = flatten(rcfn)
    i = 0
    while i < len(rcfn):
        rcfn[i] = rate_coefficient_folder_path + rcfn[i]
        i += 1
    return rcfn

def energy_loss(path):
    """
    Reads energy loss values from "reacscheme.cfg" file.
    Input argument is the path to the folder.
    """
    lin = []

    file_name = path +  '/reacscheme.cfg'

    with open(file_name, 'r', encoding = 'utf8') as f_input:
        for line in f_input:
            line = line.split('#',1)[0].strip()
            if not line:
                continue
            lin.append(line)

    energy_loss_value = []
    regex = re.compile('Uin:\s?([+-]?\d+.\d+[eE]?[-+]?\d+|0|1.0)')

    i = 0
    while i < len(lin):
        energy_loss_value.append(regex.findall(lin[i]))
        i += 1
    energy_loss_value = flatten(energy_loss_value)
    i = 0
    while i < len(energy_loss_value):
        energy_loss_value[i] = float(energy_loss_value[i])
        i += 1
    if(MPI.rank(MPI.comm_world)==0):
        print(energy_loss_value)
    return energy_loss_value

def dependence(file_name):
    """
    Reads dependence of rate coefficients from the corresponding file.
    """
    nr = len(file_name)
    x =[0]*nr
    i = 0
    while i <  nr:
        lin = []
        with open(file_name[i], 'r', encoding = 'utf8') as f_input:
            for line in f_input:
                if not line:
                    continue
                if "Dependence:" in line:
                    lin.append(line.split()[2])
        x[i] = lin[0]
        i += 1
    return x

def read_rate_coefficients(rate_coefficient_file_names, k_dependence):
    """
    Reading rate coefficients from files. Input
    are file names and dependences.
    """
    number_of_reactions = len(rate_coefficient_file_names)
    k_x = [0]*number_of_reactions
    k_y = [0]*number_of_reactions
    i = 0
    while i < number_of_reactions:
        if(MPI.rank(MPI.comm_world)==0):
            print(rate_coefficient_file_names[i])
        if k_dependence[i] == 'const':
            k_x[i] = 0.0
            k_y[i] = Read_constant(rate_coefficient_file_names[i])
        elif k_dependence[i] == 'Umean':
            k_x[i], k_y[i] = Read_Two_Column_File(rate_coefficient_file_names[i])
        elif k_dependence[i] == 'E/N':
            k_x[i], k_y[i] = Read_Two_Column_File(rate_coefficient_file_names[i])
        elif k_dependence[i] == 'fun:Te,Tgas':
            k_x[i] = 0.0
            k_y[i] = Read_One_Column_String_File(rate_coefficient_file_names[i])
            k_y[i] = str(k_y[i])
        elif k_dependence[i] == 'fun:Tgas':
            k_x[i] = 0.0
            k_y[i] = Read_One_Column_String_File(rate_coefficient_file_names[i])
            k_y[i] = str(k_y[i])
            # k_y[i] = eval(k_y[i])
        elif k_dependence[i] == 'ElecDist':
            k_x[i], k_y[i] = Read_Two_Column_File(rate_coefficient_file_names[i])
        i += 1
    return k_x, k_y

def reading_transport_coefficients(particle_names, type, model):
    """
    Reading transport coefricients from files. Input are
    particle names, type of transport coefficient (diffusion or mobility)
    and model.
    """
    path = 'file_input/' + model +'/transport_coefficients/'
    number_of_particles = len(particle_names)
    k_x = [0]*number_of_particles
    k_y = [0]*number_of_particles
    k_dependence = [0]*number_of_particles
    i = 0
    while i < number_of_particles:
        if type == 'Diffusion':
            file_name = path + particle_names[i] + '_ND.dat'
            if(MPI.rank(MPI.comm_world)==0):
                print(file_name)
            lin = []
            with open(file_name, 'r') as f_input:
                for line in f_input:
                    if not line:
                        continue
                    if "Dependence:" in line:
                        lin.append(line.split()[2])
            k_dependence[i] = lin[0]
            if k_dependence[i] == 'const':
                k_x[i] = 0.0
                k_y[i] = Read_constant(file_name)
            elif k_dependence[i] == 'Umean':
                k_x[i], k_y[i] = Read_Two_Column_File(file_name)
            elif k_dependence[i] == 'E/N':
                k_x[i], k_y[i] = Read_Two_Column_File(file_name)
            elif k_dependence[i] == 'fun:Te,Tgas':
                k_x[i] = 0.0
                k_y[i] = Read_One_Column_String_File(file_name)
                k_y[i] = eval(k_y[i])
            elif k_dependence[i] == 'Tgas':
                k_x[i], k_y[i] = Read_Two_Column_File(file_name)
            elif k_dependence[i] == 'Te':
                k_x[i], k_y[i] = Read_Two_Column_File(file_name)
            elif k_dependence[i] == 'ESR':
                k_x[i] = 0.0
                k_y[i] = 0.0
            elif k_dependence[i] == 'fun:E':
                k_x[i] = 0.0
                k_y[i] = Read_One_Column_String_File(file_name)

        elif type == 'mobility':
            file_name = path + particle_names[i] + '_Nb.dat'
            if(MPI.rank(MPI.comm_world)==0):
                print(file_name)
            if os.path.isfile(file_name):
                k_dependence[i] = 0
                lin = []
                with open(file_name, 'r') as f_input:
                    for line in f_input:
                        if not line:
                            continue
                        if "Dependence:" in line:
                            lin.append(line.split()[2])
                k_dependence[i] = lin[0]
                if k_dependence == 'const':
                    k_x[i] = 0.0
                    k_y[i] = Read_constant(file_name)
                elif k_dependence[i] == 'Umean':
                    k_x[i], k_y[i] = Read_Two_Column_File(file_name)
                elif k_dependence[i] == 'E/N':
                    k_x[i], k_y[i] = Read_Two_Column_File(file_name)
                elif k_dependence[i] == 'fun:Te,Tgas':
                    k_x[i] = 0.0
                    k_y[i] = Read_One_Column_String_File(file_name)
                    k_y[i] = eval(k_y[i])
                elif k_dependence[i] == 'Tgas':
                    k_x[i], k_y[i] = Read_Two_Column_File(file_name)
                elif k_dependence[i] == 'Te':
                    k_x[i], k_y[i] = Read_Two_Column_File(file_name)
                elif k_dependence[i] == 'fun:E':
                    k_x[i] = 0.0
                    k_y[i] = Read_One_Column_String_File(file_name)
        i += 1
    return k_x, k_y, k_dependence

def read_particle_properties(fn, model):
    """
    Reads particle properties from input file.
    Input are file name and model.
    """
    path = 'file_input/' + model +'/species/'
    if (MPI.rank(MPI.comm_world) == 0):
        print(fn)
    number_of_particles = len(fn)
    M = []
    charge = []
    i = 0
    while i < number_of_particles:
        file_name = path + fn[i]
        if (MPI.rank(MPI.comm_world) == 0):
            print(file_name)

        regex_mass = re.compile('Mass\s?=\s?([+-]?\d+.\d+[eE]?[-+]?\d+|0|1.0)')
        regex_charge = re.compile('Z\s+?=\s+?([+-]?\d+)')
        lin = []
        with open(file_name, 'r') as f_input:
            for line in f_input:
                line = line.split('#',1)[0].strip()
                if not line:
                    continue
                lin.append(line)
        M_temp = []
        charge_temp = []
        j = 0
        while j < len(lin):
            if (MPI.rank(MPI.comm_world) == 0):
                print(lin[j])
            M_temp.append(regex_mass.findall(lin[j]))
            charge_temp.append(regex_charge.findall(lin[j]))
            j += 1
        M.append(float(M_temp[1][0]))
        charge.append(float(charge_temp[0][0]))
        i += 1
    return M, charge

def print_time_step(dt):
    """
    Prints time step.
    """
    if(MPI.rank(MPI.comm_world)==0):
        print("Time step is dt =", dt)

def print_time(t):
    """
    Prints time.
    """
    if(MPI.rank(MPI.comm_world)==0):
        print("t =", t)

def file_output(t, t_old, t_out, step, t_out_list, step_list, file_type, output_file_list, particle_name, u_old, u_old1, unit = 's'):
    """
    Writing value of desired variable to file. Value is calculated by linear interpolation.
    Input arguments are current time step, previous time step, current time step length,
    previous time step length, list of output times, list of steps (which needs have same size),
    list of output files, list of variable values in current and previous time steps, and
    (optional) units of time.
    """
    if unit == 'ns':
        scale = 1e9
    elif unit == 'us':
        scale = 1e6
    elif unit == 'ms':
        scale = 1e3
    elif unit == 's':
        scale = 1.0

    if t > max(t_out_list):
        index = len(t_out_list) - 1
    else:
        index = next(x for x, val in enumerate(t_out_list)
                                          if val > t)
    while t >= t_out:
        no_files = len(output_file_list)
        i = 0
        while i < no_files:
            temp = Function(u_old1[i].function_space())
            temp.vector()[:] = u_old1[i].vector()[:] + (t_out - t_old)*(u_old[i].vector()[:] - u_old1[i].vector()[:])/(t - t_old)
            if file_type[i] == 'pvd':
                temp.rename(particle_name[i], str(i))
                output_file_list[i] << (temp, t_out*scale)
            elif file_type[i] == 'xdmf':
                output_file_list[i].write_checkpoint(temp, particle_name[i], t_out*scale, XDMFFile.Encoding.HDF5, True)  #appending to file
            i += 1
        # print(str(out_index))
        if t_out >= 0.999*t_out_list[index-1] and t_out < 0.999*t_out_list[index]:
            step = step_list[index-1]
        elif t_out >= 0.999*t_out_list[index]:
            step = step_list[index]
        t_out += step
    return t_out, step

def log(type, log_file, *arg):
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
    if(MPI.rank(MPI.comm_world)==0):
        if type == 'properties':
            gas  = arg[0]
            model = arg[1]
            particle_species_file_names = arg[2]
            M = arg[3]
            charge = arg[4]
            log_file.write('Gas:\t' + gas + '\n\n' + 'model:\t' + model + '\n' + '\nParticle names: \n' + str(particle_species_file_names) + '\n' + '\nMass:  \n' + str(M) + '\n' + '\nCharge: \n' + str(charge) + '\n' + '\n')
            log_file.flush()

        elif type == 'conditions':
            dt_var = arg[0]
            U_w = arg[1]
            p0 = arg[2]
            gap_length = arg[3]
            N0 = arg[4]
            Tgas = arg[5]
            log_file.write('Simulation conditions:\n' + 'dt = ' + str(dt_var) + ' s' + '\t U_w = ' + str(U_w) + ' V'  + '\t p_0 = ' + str(p0) + ' Torr' + '\t d = ' +  str(gap_length) + ' m' + '\t N_0 = ' +  str(N0) + ' m^-3' + '\t T_gas = ' +  str(Tgas) + ' K' + '\n\n')
            log_file.flush()
        elif type == 'matrices':
            gain = arg[0]
            loss = arg[1]
            power = arg[2]
            n_g_rows = gain.shape[1]
            n_g_columns = gain.shape[0]
            n_l_rows = loss.shape[1]
            n_l_columns = loss.shape[0]
            n_p_rows = power.shape[1]
            n_p_columns = power.shape[0]
            log_file.write('Gain matrix:' + '\n')
            i = 0
            while i < n_g_columns:
                j=0
                while j < n_g_rows:
                    log_file.write(str(gain[i][j]))
                    j += 1
                log_file.write('\n')
                i += 1
            log_file.write('\n')
            log_file.write('Loss matrix:' + '\n')
            i = 0
            while i < n_l_columns:
                j=0
                while j < n_l_rows:
                    log_file.write(str(loss[i][j]))
                    j += 1
                log_file.write('\n')
                i += 1
            log_file.write('\n')
            log_file.write('Power matrix:' + '\n')
            i = 0
            while i < n_p_columns:
                j=0
                while j < n_p_rows:
                    log_file.write(str(power[i][j]))
                    j += 1
                log_file.write('\n')
                i += 1
            log_file.write('\n')
            log_file.flush()

        elif type == 'initial time':
            t = arg[0]
            log_file.write('Time:\n' + str(t) + '\n')
            log_file.flush()

        elif type == 'time':
            t = arg[0]
            log_file.write(str(t) + '\n')
            log_file.flush()
    if type == 'mesh':
        mesh = arg[0]
        n_element = MPI.sum(MPI.comm_world, mesh.num_cells())
        #measures the greatest distance between any two vertices of a cell
        hmax = MPI.max(MPI.comm_world, mesh.hmax())
        #measures the smallest distance between any two vertices of a cell
        hmin = MPI.min(MPI.comm_world, mesh.hmin())
        if(MPI.rank(MPI.comm_world)==0):
            log_file.write("Number of elements in the mesh is: ")
            print("%.*g" % (5, n_element), file = log_file)
            log_file.write("Maximum element edge length is: ")
            print("%.*g" % (5, hmax), file=log_file)
            log_file.write("Minimum element edge length is: ")
            print("%.*g" % (5, hmin), file=log_file)
            log_file.write('\n')
