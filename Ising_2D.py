import numpy as np
import numpy.random as nr

from modules.initialize import cold_start, hot_start
from modules.simulation import should_accept

###################################################################
#region Params
###################################################################

N_lat = 16
N_points = N_lat*N_lat
J = -1
H = 0
T = 1

iterations = int(1e5)

#endregion
###################################################################
#region Set memories
###################################################################

idx_array = np.arange(N_points)
x_array = idx_array % N_lat
y_array = idx_array // N_lat

m_cold = np.zeros(iterations)
m_hot = np.zeros(iterations)

#endregion
###################################################################
#region Simulation
###################################################################

# cold start
spin_field = cold_start((N_lat, N_lat))

for i in range(iterations):
    flip_idx = nr.randint(0, N_points)
    if should_accept(spin_field, flip_idx, T, N_lat, J, H):
        spin_field[x_array[flip_idx]][y_array[flip_idx]] *= -1

    m_cold[i] = spin_field.mean()

# hot start
spin_field = hot_start((N_lat, N_lat))

for i in range(iterations):
    flip_idx = nr.randint(0, N_points)
    if should_accept(spin_field, flip_idx, T, N_lat, J, H):
        spin_field[x_array[flip_idx]][y_array[flip_idx]] *= -1

    m_hot[i] = spin_field.mean()

#endregion
###################################################################
#region Plot
###################################################################

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

def field_plot():
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    ax.set_aspect('equal')

    cmap = ListedColormap(['b', 'r'])
    ax.imshow(spin_field.T, origin='lower', extent=(0, N_lat, 0, N_lat), cmap=cmap, vmin=-1, vmax=1)

    red_patch = mpatches.Patch(color='r', label='Up-spin')
    blue_patch = mpatches.Patch(color='b', label='Down-spin')
    ax.legend(handles=[red_patch, blue_patch])

    ax.set_xticks(range(17)); ax.set_yticks(range(1, 17))
    ax.tick_params(axis='both', color='none')

    fig.tight_layout()
    fig.savefig("Ising2D.png", dpi=300)

def magnetization_plot():
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()

    ax.plot(range(1, iterations+1), m_cold, c='b', label='Cold start', alpha=0.5)
    ax.plot(range(1, iterations+1), m_hot, c='r', label='Hot start', alpha=0.5)

    ax.tick_params(axis='both', direction='in')

    ax.set_xlim(0, iterations)
    ax.set_ylim(-0.5,0.5)

    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(f"Ising2D-magnetization_T={T}.png", dpi=300)

magnetization_plot()
# field_plot()

#endregion