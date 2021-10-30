#%%
import particle as p
import matplotlib.pyplot as plt

plt.plot(1,1) #used to initialise animation (IDE specific thing)
plt.show()

simulation = p.Simulation(p.ballarray)

frames = 200000 #number of iterations

simulation.run(frames, animate=True)
# plt.plot(simulation._timeArray,simulation._pressureArray)
# plt.show()

# # %%
# import numpy as np
# import poisson_disc as poi
# import matplotlib.pyplot as plt

# points = poi.Bridson_sampling(dims=np.array([10,10]), radius=1, k=5)
# plt.plot(points[:,0],points[:,1], '.')
# plt.show()

# #to get arbritary number of points, just set k to be large, and then get
# #length of points array, then delete elements until we get the
# #length that we wanted
# %%
