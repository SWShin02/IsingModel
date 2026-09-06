import datetime

import numpy as np
import numpy.random as nr
import h5py

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
repetition = 1000

start_sampling_idx = int(2e4)

#endregion
###################################################################
#region Set memories
###################################################################

idx_array = np.arange(N_points)
x_array = idx_array % N_lat
y_array = idx_array // N_lat

T_array = np.linspace(0.1, 5, 70)

m_temperature = np.zeros(len(T_array))
m_square_temperature = np.zeros(len(T_array))

#endregion
###################################################################
#region Simulation
###################################################################
# Execute simulation with different temperatures
magnetization = np.zeros((len(T_array), repetition, iterations//10))

for i, T_target in enumerate(T_array):
    print(f"temperature: {T_target:.3f}")
    T_anealing = np.linspace(np.max(T_array), T_target, iterations//2)
    T=T_anealing[0]

    for j in range(repetition):
        if j%2 == 0:
            # cold start
            spin_field = cold_start((N_lat, N_lat))
        else: 
            # hot start
            spin_field = hot_start((N_lat, N_lat))

        for k in range(iterations):
            if k < iterations//2: T=T_anealing[k]
            
            flip_idx = nr.randint(0, N_points)
            if should_accept(spin_field, flip_idx, T, N_lat, J, H):
                spin_field[x_array[flip_idx]][y_array[flip_idx]] *= -1

            if k%10 == 0: magnetization[i,j,k//10] = spin_field.mean()

#endregion
###################################################################
#region Save results
###################################################################

group_name = datetime.datetime.now().strftime("%Y%m%d%H%M")

with h5py.File("Ising2D.h5", "a") as f:
    group = f.create_group(group_name)
    group.create_dataset("magnetization", data=magnetization, compression='gzip')
    group.create_dataset("temperature", data=T_array)

#endregion