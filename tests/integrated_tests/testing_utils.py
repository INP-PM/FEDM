from pathlib import Path
import numpy as np
from vtk import vtkXMLUnstructuredGridReader

def l_inf_norm(x: np.ndarray) -> np.ndarray:
    return np.max(np.abs(x))

def l1_norm(x: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(x))

def l2_norm(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(x**2))

def read_vtu(path: Path, field_name: str) -> np.ndarray:
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    return np.asarray(reader.GetOutput().GetPointData().GetArray(field_name))

