import numpy as np
import numpy.random as nr

###################################################################
#region Params
###################################################################

N_lat = 16
N_points = N_lat*N_lat
J = 1
H = 0

iterations = int(1e5)
repetition = 100

start_sampling_idx = int(2e4)

#endregion
###################################################################
#region Define functions
###################################################################

def compute_DeltaE(spin_field:np.ndarray, flip_idx:int)-> float:
    """
    ## input variables
    `spin_field`: Entire spin field of the system with PBC. \\
    `flip_idx`: Index of the point where you want to flip the spin.
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

def should_accept(spin_field:np.ndarray, flip_idx:int, T:float)-> bool:
    DeltaE = compute_DeltaE(spin_field, flip_idx)
    if DeltaE < 0: return True
    else:
        P = np.exp(-DeltaE/T)
        dice = nr.uniform()
        if dice < P: return True
        else: return False

def cold_start(shape)-> np.ndarray:
    return np.ones(shape=shape)

def hot_start(shape)-> np.ndarray:
    return nr.randint(2, size=shape)*2 - 1

#endregion
###################################################################
# Set memories
###################################################################

idx_array = np.arange(N_points)
x_array = idx_array % N_lat
y_array = idx_array // N_lat

T_array = np.linspace(0.1, 5, 70)

m_temperature = np.zeros(len(T_array))
m_square_temperature = np.zeros(len(T_array))

#region
###################################################################
# Simulation
###################################################################
# Execute simulation with different temperatures
for i, T in enumerate(T_array):
    print(T)

    m_tmp = np.zeros((repetition, iterations))

    for j in range(repetition):
        if j%2 == 0:
            # cold start
            spin_field = cold_start((N_lat, N_lat))
        else: 
            # hot start
            spin_field = hot_start((N_lat, N_lat))

        for k in range(iterations):
            flip_idx = nr.randint(0, N_points)
            if should_accept(spin_field, flip_idx, T):
                spin_field[x_array[flip_idx]][y_array[flip_idx]] *= -1
            
            m_tmp[j][k] = spin_field.mean()

    m_temperature[i] = np.abs(m_tmp[:,start_sampling_idx:]).mean()
    m_square_temperature[i] = (m_tmp[:,start_sampling_idx:]**2).mean()

chi_T = (N_points/T_array)*(m_square_temperature-m_temperature**2)

mask = T_array > 2
valid_T = T_array[mask]
valid_chi = chi_T[mask]
valid_m = m_temperature[mask]

Tc_idx = np.argmax(valid_chi)
Tc = valid_T[Tc_idx]
m_Tc = valid_m[Tc_idx]
chi_Tc = valid_chi.max()

print("Critical temperature: ", Tc)

#endregion
###################################################################
#region Plot
###################################################################

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()

ax.plot(T_array, m_temperature, c='b', marker='o', markersize=5)
ax.axvline(Tc, ls='--', c='gray')

ax.tick_params(axis='both', direction='in')
ax.set_xlim(0, T_array.max())

ax.set_xlabel("$T$")
ax.set_ylabel("$\\langle |m| \\rangle$")

ax.grid(True)

fig.tight_layout()
fig.savefig(f"magnetization.png", dpi=300)

###################################################################

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()

ax.plot(T_array, chi_T, c='b', marker='o', markersize=5)
ax.axvline(Tc, ls='--', c='gray')

ax.tick_params(axis='both', direction='in')
ax.set_xlim(0, T_array.max())

ax.set_xlabel("$T$")
ax.set_ylabel("$\\chi_T$")

ax.grid(True)

fig.tight_layout()
fig.savefig(f"chi_T.png", dpi=300)

#endregion