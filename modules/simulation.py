import numpy as np
import numpy.random as nr
from numba import njit

@njit
def compute_DeltaE(spin_field:np.ndarray, flip_idx:int, N_lat:int, J:float, H:float)-> float:
    """
    ## input variables
    `spin_field`: Entire spin field of the system with PBC. \\
    `flip_idx`: Index of the point where you want to flip the spin. \\
    `N_lat`: Linear size of the lattice. \\
    `J`: Coupling constant. \\
    `H`: External field.
    ---
    ## return
    `DeltaE`: Energy difference between before flip and after flip.
    """
    x:int = flip_idx % N_lat
    y:int = flip_idx // N_lat

    sum_spin_neighbor = (
        spin_field[(x+1)%N_lat][y]
        + spin_field[x][(y+1)%N_lat]
        + spin_field[x-1][y]
        + spin_field[x][y-1]
    )

    return  2 * (J * sum_spin_neighbor + H) * spin_field[x][y]

@njit
def should_accept(spin_field:np.ndarray, flip_idx:int, T:float, N_lat:int, J:float, H:float)-> bool:
    DeltaE = compute_DeltaE(spin_field, flip_idx, N_lat, J, H)
    if DeltaE < 0: return True
    else:
        P = np.exp(-DeltaE/T)
        dice = nr.uniform()
        if dice < P: return True
        else: return False
