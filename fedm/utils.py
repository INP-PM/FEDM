from textwrap import dedent
from typing import List
import dolfin as df


def print_rank_0(*args, **kwargs) -> None:
    """
    Utility function, print to terminal if MPI rank 0, otherwise do nothing.
    """
    if df.MPI.rank(df.MPI.comm_world) == 0:
        print(*args, **kwargs)


def comma_separated(strings: List[str]) -> str:
    """
    Utility function, takes a list of strings and returns a single string containing
    each of those strings in single quotes and separated by  ', '
    """
    return ", ".join([f"'{string}'" for string in strings])


def mesh_info(mesh: df.Mesh) -> str:
    n_element = df.MPI.sum(df.MPI.comm_world, mesh.num_cells())
    # measures the greatest distance between any two vertices of a cell
    hmax = df.MPI.max(df.MPI.comm_world, mesh.hmax())
    # measures the smallest distance between any two vertices of a cell
    hmin = df.MPI.min(df.MPI.comm_world, mesh.hmin())
    info_str = dedent(
        f"""\
        Number of elements is: {int(n_element)}
        Maximum element edge length is: {hmax:.5g}
        Minimum element edge length is: {hmin:.5g}
        """
    )
    return info_str
