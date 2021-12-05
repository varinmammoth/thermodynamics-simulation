#%%
import numpy as np
import matplotlib.pyplot as plt
import particle as p
import particleStats as pstats

#%%
""" 
Plot for the pressure graphs/histograms, inter-ball distance and center-ball disntace.
"""
num_balls = 300

histogram = False
ballarray = p.BallsArray(container_r=20)
ballarray.random_vel(num_balls, 0, 5, 1e-26, 0.25) 
simulation = p.Simulation(ballarray)
simulation.run(2000, animate=False, histogram=True, timeInterval=1)


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
t, temp = simulation.get_temp(N=num_balls)

plt.plot(pressure_t, pressure, '.')
plt.axhline(np.mean(pressure), color='g', linestyle='-')
plt.axhline(np.mean(pressure)-np.std(pressure), color='r', linestyle='--')
plt.axhline(np.mean(pressure)+np.std(pressure), color='r', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (N m^-1)')
plt.grid()

# %%
""" 
Changing the temperature by initialising velocity distribution of balls
with different standard deviations.
Use the results to make a T vs P plot.

Also plot Maxwell Boltzmann distribution for each temperature.
"""
sd_array = [5,20,40,60,80,100,120]
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
     ballarray.random_vel(num_balls, 0, sd, 1e-26, 0.01) 
     simulation = p.Simulation(ballarray)
     simulation.run(500, animate=False, histogram=True, timeInterval=1)

     pressure_t, pressure = simulation.get_pressure()
     avg_pressure.append(np.mean(pressure))
     sd_pressure.append(np.std(pressure))

     t, temp = simulation.get_temp(N=num_balls)
     avg_temp.append(np.mean(temp))
     sd_temp.append(np.std(temp))

     t, v, vx, vy = simulation.get_velocities(histogram=histogram)
     center, hist, hist_err, hist_norm, hist_norm_err = pstats.get_histogram(v, bins=10)
     v_centers.append(center)
     v_counts.append(hist_norm)
     v_counts_err.append(hist_norm_err)

     print('Iteration', iteration)
     iteration += 1

     ballarray.reset()
# %%
plt.errorbar(avg_temp, avg_pressure, xerr=sd_temp, yerr=sd_pressure, fmt='.', capsize=2)
fit, cov = np.polyfit(avg_temp, avg_pressure, deg=1, cov=True)
line = np.poly1d(fit)
plt.plot(np.linspace(0,8,1000), line(np.linspace(0,8,1000)))
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (N m^-1)')
plt.grid()
plt.show()

def maxwell(v,m,T):
    kb = 1.38e-23
    maxwell = (v**2)*(np.exp((-1*m*v*v)/(2*kb*T)))
    maxwell =4*np.pi*maxwell*((m/(2*np.pi*kb*T))**1.5)
    return maxwell

xlimarray = [50,50,50,50,200,200,200,200,300,300,300,300]
ylimarray = [0.15, 0.15, 0.15, 0.15, 0.03, 0.03, 0.03, 0.03, 0.0175, 0.0175, 0.0175, 0.0175]
plt.figure(figsize=(16, 12), dpi=80)
plt.xlabel('Speed (m/s)')
plt.ylabel('Normalised frequency')
x = np.linspace(0,1000,1000)
for i in range(0,len(v_centers)):
     plt.subplot(3,4,i+1)
     plt.bar(v_centers[i], v_counts[i], width=v_centers[i][2]-v_centers[i][1])
     plt.errorbar(v_centers[i], v_counts[i], yerr=v_counts_err[i], fmt='none', capsize=2, c='green')
     plt.title(f'{avg_temp[i]:.3}')
     plt.xlim(0,xlimarray[i])
     plt.ylim(0,ylimarray[i])
     plt.plot(x, maxwell(x, 1e-26, avg_temp[i]), c='red')
plt.show()
# %%
""" 
Doing the same as above (making P vs T plot) but instead of small balls,
we increase the radii of balls.

Then use the gradient to determine b (Van der Waals constant).

Note: this takes a while to run. The output are stored as variables in the code below.
"""
r_array = [0.25,0.5,1.0,1.5,1.75]
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
          simulation.run(500, animate=False, histogram=True, timeInterval=0.2)

          # pressure_t, pressure = simulation.get_pressure()
          # avg_pressure.append(np.mean(pressure))
          # sd_pressure.append(np.std(pressure))
          avg_pressure.append(simulation.whole_average_pressure())

          t, temp = simulation.get_temp(N=num_balls)
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
def get_b(A, N, slope):
     kb = 1.38e-23
     b = (A/N) - (kb/slope)
     return b
b = []
b_err = []
color = ['red', 'green','blue','black','purple','orange','grey','yellow']
avg_pressure_r=[[1.32323183224433e-26,
  2.1583703917517314e-25,
  7.962631948590212e-25,
  1.9323851225567055e-24,
  2.7534947763328717e-24],
 [1.480933659671273e-26,
  2.0422089689556104e-25,
  8.955860452479798e-25,
  1.827380992024887e-24,
  2.9973066490897853e-24],
 [1.3867204586281166e-26,
  2.132839131199682e-25,
  9.964701639248048e-25,
  2.24015129542594e-24,
  2.386973676081832e-24],
 [1.716078348509841e-26,
  3.522944675519486e-25,
  1.858442500738293e-24,
  2.7984293692113493e-24,
  4.471954499160728e-24],
  [3.862317615218784e-26,
  6.200270774846808e-25,
  3.083442845779768e-24,
  4.462095162095348e-24,
  5.547094548457887e-24]]
avg_temp_r = [[0.019473897286875024,
  0.3224325758612069,
  1.1259751118099923,
  2.5554646905705973,
  3.9017847443420663],
 [0.017854044871837952,
  0.2783918009301402,
  0.9341390738472986,
  2.279362292575304,
  3.903967128193322],
 [0.017591119035217324,
  0.27090168466336645,
  1.1836941772999194,
  2.8971893867991327,
  3.1540429321104013],
 [0.01165008470708901,
  0.2650024727427809,
  1.340693618994919,
  1.9938094129330384,
  3.7001971612799047],
  [0.020938543703838946,
  0.3364245311375173,
  1.6027393985848852,
  2.434259195193266,
  3.118683967136061]]
for  i in range(0,len(avg_pressure_r)):
     plt.plot(avg_temp_r[i], avg_pressure_r[i], 'o', label=r_array[i], c=color[i])
     fit, cov = np.polyfit(avg_temp_r[i], avg_pressure_r[i], deg=1, cov=True)
     line = np.poly1d(fit)
     x = np.linspace(0,4,1000)
     plt.plot(x, line(x), c=color[i])
     plt.errorbar(avg_temp_r[i], avg_pressure_r[i], yerr=0.05*np.array(avg_pressure_r[i]), fmt='none', c=color[i], capsize=2)
     bvalue = get_b(np.pi*(20**2), 50, fit[0])
     b.append(bvalue)
     b_err.append((np.sqrt(cov[0,0])/fit[0])*bvalue)
     print(fit, cov)
legend = plt.legend()
legend.set_title('Ball radii (m)')
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (N m^-1)')
plt.grid()
plt.show()

plt.errorbar(r_array, b, yerr=b_err, fmt='o', capsize=2)
fit, cov = np.polyfit(r_array, b, deg=1, cov=True)
line = np.poly1d(fit)
x = np.linspace(0.2,2,1000)
plt.plot(x, line(x))
plt.xlabel('Radius (m)')
plt.ylabel('b (m^2)')
plt.grid()
plt.show()
print(fit, cov)

# %%
""" 
Changing number of balls N and keeping temperature T constant to investigate Van der Waals equation.

Note: this takes a while to run. The output are stored as variables in the code below.
"""
avg_pressure_N = []
sd_pressure_N = []
avg_temp_N = []
sd_temp_N = []

num_balls_array = [125, 175, 225, 250]

iteration=1
for num_balls in num_balls_array:
     ballarray = p.BallsArray(container_r=50)
     ballarray.uniform(num_balls, [1,0], 1, 2) 
     simulation = p.Simulation(ballarray)
     simulation.run(1000, animate=False, histogram=False, timeInterval=0.2)

     avg_pressure_N.append(simulation.whole_average_pressure())

     t, temp = simulation.get_temp(N=num_balls, histogram=False)
     avg_temp_N.append(temp)

     print('Iteration', iteration)
     iteration += 1
     print(avg_pressure_N)
     print(avg_temp_N)

     ballarray.reset()

# %%
num_balls_array = [25,50,75,100,150,200,125]
avg_pressure_N = [0.0024105882109667722,0.005017169094481611,0.009638686764183219,0.013188381972289822,0.024354199919639528,0.04070146064253679, 0.018546374953925766]
plt.plot(num_balls_array, avg_pressure_N, '.')
# %%
#%%
""" 
Finding b from Van der Waals equation using Scipy.
"""

num_balls_array = [25,50,75,100,150,200,125,175,225]
avg_pressure_N = [0.0024105882109667722,0.005017169094481611,0.009638686764183219,0.013188381972289822,0.024354199919639528,0.04070146064253679, 0.018546374953925766, 0.02919310791475456, 0.05202315815995248]
plt.plot(num_balls_array, avg_pressure_N, '.')

import scipy.optimize as sci

#vanderwaals equation fit
def vanderwaals(N,b):
    kb = 1.38e-23
    A=np.pi*(50**2)
    T=3.623189410258068e+22
    P = ((N*kb*T)/(A-N*b))
    return P

p0=[20]
values, cov = sci.curve_fit(vanderwaals, num_balls_array, avg_pressure_N, p0=p0)

#fit of first two points when ball radius negligible
def line(x):
    m=(avg_pressure_N[1]-avg_pressure_N[0])/(25)
    return m*x

x = np.linspace(0,240,1000)
plt.errorbar(num_balls_array, avg_pressure_N, yerr=0.1*np.array(avg_pressure_N), fmt='.', capsize=2, c='black')
plt.plot(x, line(x), label="Ideal gas")
plt.plot(x, vanderwaals(x, *values), label="Van der Waals gas", c='red')
plt.legend()
plt.grid()
plt.xlabel('Number of balls')
plt.ylabel('Pressure (N m^-1)')