#%%
import numpy as np
import matplotlib.pyplot as plt
import particle as p
import particleStats as pstats

#%%

num_balls = 50

histogram = False
ballarray = p.BallsArray(container_r=20)
ballarray.random_vel(num_balls, 0, 5, 1e-26, 0.25) 
simulation = p.Simulation(ballarray)
simulation.run(600, animate=True, histogram=True, timeInterval=1)


pressure_t, pressure = simulation.get_pressure()
t, distballs, distcenter = simulation.get_distances(histogram=histogram)
t, v, vx, vy = simulation.get_velocities(histogram=histogram)
t, Etotal, Esingle = simulation.get_energies(histogram=True)

center, hist, hist_err, hist_norm, hist_norm_err\
     = pstats.get_histogram(pressure, bins=6)
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
t, temp = simulation.get_temp()

# %%
sd_array = [5,10,15,20,25,30,35,40,45,50]
num_balls = 50
histogram = False

avg_pressure = []
sd_pressure = []
avg_temp = []
sd_temp = []

v_centers = []
v_counts = []
v_counts_err = []
iteration = 1
for sd in sd_array:
     ballarray = p.BallsArray(container_r=20)
     ballarray.random_vel(num_balls, 0, sd, 1e-26, 0.25) 
     simulation = p.Simulation(ballarray)
     simulation.run(600, animate=True, histogram=True, timeInterval=1)

     pressure_t, pressure = simulation.get_pressure()
     avg_pressure.append(np.mean(pressure))
     sd_pressure.append(np.std(pressure))

     t, temp = simulation.get_temp()
     avg_temp.append(np.mean(temp))
     sd_temp.append(np.std(temp))

     t, v, vx, vy = simulation.get_velocities(histogram=histogram)
     center, hist, hist_err, hist_norm, hist_norm_err = pstats.get_histogram(v)
     v_centers.append(center)
     v_counts.append(hist_norm)
     v_counts_err.append(hist_norm_err)

     print('Iteration', iteration)
     iteration += 1

     ballarray.reset()

# %%
plt.errorbar(avg_temp, avg_pressure, xerr=sd_temp, yerr=sd_pressure, fmt='.', capsize=2)
fit, cov = np.polyfit(avg_temp, avg_pressure, deg=1, w=sd_temp, cov=True)
line = np.poly1d(fit)
plt.plot(np.linspace(0,70,1000), line(np.linspace(0,70,1000)))
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (arbitrary units)')
plt.show()

# for i in range(0,len(v_centers)):
#      plt.errorbar(v_centers[i], v_counts[i], yerr=v_counts_err[i], fmt='.', capsize=2, label=avg_temp[i])
# plt.legend()
# plt.show

def maxwell(v,m,T):
    kb = 1.38e-23
    maxwell = np.exp((-1*m*v*v)/(2*kb*T))
    maxwell = maxwell*(m*v)/(kb*T)
    return maxwell

plt.figure(figsize=(10, 10), dpi=80)
for i in range(0,len(v_centers)):
     plt.subplot(3,4,i+1)
     plt.bar(v_centers[i], v_counts[i], width=v_centers[i][2]-v_centers[i][1])
     plt.errorbar(v_centers[i], v_counts[i], yerr=v_counts_err[i], fmt='none', capsize=2, c='green')
     plt.title(avg_temp[i])
     plt.xlim(0,130)
     plt.ylim(0,1)
plt.legend()
plt.show
# %%
