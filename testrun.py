#%%
import particle as p
import matplotlib.pyplot as plt

plt.plot(1,1) #used to initialise animation (IDE specific thing)
plt.show()

ballarray = p.BallsArray(container_r=20)
ballarray.uniform(1, [1,0], 1, 0.5)
simulation = p.Simulation(ballarray)

frames = 2000 #number of iterations

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
import numpy as np
def get_t(t_array):
    t_array_real = []
    for i in t_array:
        if isinstance(i, complex) == False and (i > 0):
            t_array_real.append(i)

    if len(t_array_real) != 0:
        return np.min(t_array_real)
    else:
        return None
# %%
