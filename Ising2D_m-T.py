import numpy as np
import numpy.random as nr

from modules.initialize import cold_start, hot_start
from modules.simulation import should_accept

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
            if should_accept(spin_field, flip_idx, T, N_lat, J, H):
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