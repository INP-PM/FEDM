from dolfin import *
from mshr import *
import numpy as np
from timeit import default_timer as timer
import time

elementary_charge = 1.6021766208e-19 #[C]
me = 9.10938356e-31 #[kg]
epsilon_0 = 8.854187817e-12 #[F/m]
kB = 1.38064852e-23 #[J/K]
kB_eV = 8.6173303e-5 #[J/eV]
speed_of_light = 2.99792458e8 #[m/s]
h_planck = 6.62607015e-34 #[J/s]
mag_perm = 1.25663706212e-6 #[N/A^2]
N_avogadro = 6.02214076e23 #[1/mol]
Ry_const = 	10973731.568160 #[1/m]
M_atomic = 1.66053906660e-27 #[kg]
