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
""" 
Changing the temperature by initialising velocity distribution of balls
with different standard deviations.
Use the results to make a T vs P plot.
"""
sd_array = [5,15,25,35,45,55,65]
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
fit, cov = np.polyfit(avg_temp, avg_pressure, deg=1, w=1/np.array(sd_temp), cov=True)
line = np.poly1d(fit)
plt.plot(np.linspace(0,110,1000), line(np.linspace(0,110,1000)))
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (arbitrary units)')
plt.grid()
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
     plt.title(f'{avg_temp[i]:.3}')
     plt.xlim(0,130)
     plt.ylim(0,1)
plt.show
# %%
""" 
Doing the same as above (making P vs T plot) but instead of small balls,
we increase the radii of balls.
"""
r_array = [0.25, 0.5, 1, 1.5]
sd_array = [5,20,40,60,70]
num_balls = 50
histogram = False

avg_pressure_r = []
sd_pressure_r = []
avg_temp_r = []
sd_temp_r = []

iteration = 1
for radius in r_array:
     avg_pressure = []
     sd_pressure = []
     avg_temp = []
     sd_temp = []
     
     for sd in sd_array:
          ballarray = p.BallsArray(container_r=20)
          ballarray.random_vel(num_balls, 0, sd, 1e-26, radius) 
          simulation = p.Simulation(ballarray)
          simulation.run(500, animate=True, histogram=True, timeInterval=0.2)

          # pressure_t, pressure = simulation.get_pressure()
          # avg_pressure.append(np.mean(pressure))
          # sd_pressure.append(np.std(pressure))
          avg_pressure.append(simulation.whole_average_pressure())

          t, temp = simulation.get_temp()
          avg_temp.append(np.mean(temp))
          sd_temp.append(np.std(temp))

          print('Iteration', iteration)
          iteration += 1

          ballarray.reset()

     avg_pressure_r.append(avg_pressure)
     sd_pressure_r.append(sd_pressure_r)
     avg_temp_r.append(avg_temp)
     sd_temp_r.append(sd_temp_r)
# %%
color = ['red', 'green','blue','black','purple']
for  i in range(0,len(avg_pressure_r)):
     plt.plot(avg_temp_r[i], avg_pressure_r[i], 'o', label=r_array[i], c=color[i])
     fit, cov = np.polyfit(avg_temp_r[i], avg_pressure_r[i], deg=1, cov=True)
     line = np.poly1d(fit)
     x = np.linspace(0,150,1000)
     plt.plot(x, line(x), c=color[i])
     plt.errorbar(avg_temp_r[i], avg_pressure_r[i], yerr=0.05*np.array(avg_pressure_r[i]), fmt='none', c=color[i], capsize=2)
legend = plt.legend()
legend.set_title('Radii')
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (abritrary units)')
plt.grid()
plt.show()


# %%
