#%%
import numpy as np
import matplotlib.pyplot as plt
import particle as p
import particleStats as pstats

sd_array = np.linspace(1,10,10)

num_balls = 200

# for sd in sd_array:
histogram = False
ballarray = p.BallsArray(container_r=20)
ballarray.random_vel(num_balls, 0, 5, 5, 0.25) 
simulation = p.Simulation(ballarray)
simulation.run(1, animate=True, histogram=True)

if histogram == False:
    pressure_t, pressure = simulation.get_pressure()
    t, distballs, distcenter = simulation.get_distances(histogram=histogram)
    t, v, vx, vy = simulation.get_velocities(histogram=histogram)
    t, Etotal, Esingle = simulation.get_energies(histogram=True)

    center, hist, hist_err, hist_norm, hist_norm_err\
         = pstats.get_histogram(pressure)
    plt.bar(center, hist_norm, width=center[1]-center[0],\
         color='yellow', edgecolor='b')
    plt.errorbar(center, hist_norm, yerr=hist_norm_err, capsize=2, fmt='none')
    plt.xlabel('Pressure')
    plt.ylabel('Normalised frequency')
    plt.show()

    center, hist, hist_err, hist_norm, hist_norm_err\
         = pstats.get_histogram(distballs)
    plt.bar(center, hist_norm, width=center[1]-center[0],\
         color='yellow', edgecolor='b')
    plt.errorbar(center, hist_norm, yerr=hist_norm_err, capsize=2, fmt='none')
    plt.xlabel('Distance between balls')
    plt.ylabel('Normalised frequency')
    plt.show()

    center, hist, hist_err, hist_norm, hist_norm_err\
         = pstats.get_histogram(distcenter)
    plt.bar(center, hist_norm, width=center[1]-center[0],\
         color='yellow', edgecolor='b')
    plt.errorbar(center, hist_norm, yerr=hist_norm_err, capsize=2, fmt='none')
    plt.xlabel('Distance of balls to center')
    plt.ylabel('Normalised frequency')
    plt.show()

    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(1,3,1)
    center, hist, hist_err, hist_norm, hist_norm_err\
         = pstats.get_histogram(v)
    plt.bar(center, hist_norm, width=center[1]-center[0],\
         color='yellow', edgecolor='b')
    plt.errorbar(center, hist_norm, yerr=hist_norm_err, capsize=2, fmt='none')
    plt.xlabel('v')
    plt.ylabel('Normalised frequency')
    plt.subplot(1,3,2)
    center, hist, hist_err, hist_norm, hist_norm_err\
         = pstats.get_histogram(vx)
    plt.bar(center, hist_norm, width=center[1]-center[0],\
         color='yellow', edgecolor='b')
    plt.errorbar(center, hist_norm, yerr=hist_norm_err, capsize=2, fmt='none')
    plt.xlabel('vx')
    plt.subplot(1,3,3)
    center, hist, hist_err, hist_norm, hist_norm_err\
         = pstats.get_histogram(vy)
    plt.bar(center, hist_norm, width=center[1]-center[0],\
         color='yellow', edgecolor='b')
    plt.errorbar(center, hist_norm, yerr=hist_norm_err, capsize=2, fmt='none')
    plt.xlabel('vy')
    plt.show()

plt.plot(t, Etotal, '.')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.show()
# %%

# %%
