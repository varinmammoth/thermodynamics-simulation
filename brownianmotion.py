#%%
import particle as p
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
#%%
class brownian_ball(p.Ball):
    """ 
    A class to create the Brownain ball.
    """
    def __init__(self,m,r,pos,v,type='ball',color='green'):
        """Initialises brownian_ball object with all the nessecary attributes.
        Args:
            m (float): Mass of the object.
            r (float): Radius of the object.
            p (list): Initial x, y coordinates of object in a list of length 2.
            v (list): Initial x, y components of object's velocity in a list of length 2.
            type (str, optional): Type of the object, either "ball" or "container". Defaults to "ball".
        """
        p.Ball.__init__(self,m,r,pos,v,type,color)
        self._time = []
        self._x = []
        self._y = []

    def get_time_pos(self):
        """Returns the time, x position, and y postion array of the Brownian ball.

        Returns:
            list: Time array.
            list: x position array.
            list: y position array.
        """
        return self._time, self._x, self._y

class brownian_simulation(p.Simulation):
    """ 
    A class to simulate Brownian motion.
    """
    def run(self, num_frames, animate=False, histogram=True, timeInterval=0.25):
        """Function to run the whole simulation for a set number of frames. The abilities to display a
        visual animation and store various properties during each iteration to later produce an animated
        historgram is available.

        Args:
            num_frames (int): Number of frames to run the simulation for.
            animate (bool, optional): Set to True to display a visual animation. Defaults to False.
            histogram (bool, optional): Set to True to save various properties of the system during 
            each iteration to be later used in an animated histogram.. Defaults to True.
            timeInterval (float): This time interval will be used in average pressure calculations.
        """
        if animate:
            f = pl.figure()
            ax = pl.axes(xlim=(-self._ballarray.get_array()[-1]._r, self._ballarray.get_array()[-1]._r), ylim=(-self._ballarray.get_array()[-1]._r, self._ballarray.get_array()[-1]._r))
            ax.add_artist(self._ballarray.get_array()[-1].get_patch())
            for ball in self._ballarray.get_array()[0:-1]:
                ax.add_patch(ball.get_patch())
        for frame in range(num_frames):
            self._ballarray.get_array()[0]._x.append(self._ballarray.get_array()[0].pos()[0])
            self._ballarray.get_array()[0]._y.append(self._ballarray.get_array()[0].pos()[1])
            self._ballarray.get_array()[0]._time.append(self._t)
            self.next_collision(histogram, timeInterval)
            if animate:
                ax.set_title(frame)
                print(frame)
                pl.pause(0.01)
        if animate:
            pl.show()

"""
Testing that Brownian motion works.
"""
# brownian_molecule = brownian_ball(0.01, 5, [0,0], [0,0])
# atoms = p.BallsArray(container_r=20)
# atoms.manual_add_ball(brownian_molecule)
# atoms.brownian(50,0,5,0.0001,0.25,6)

# brownian_motion = brownian_simulation(atoms)
# brownian_motion.run(1000, animate=True, histogram=False)

# f = pl.figure()
# ax = pl.axes(xlim=(-20, 20), ylim=(-20, 20))
# ax.add_artist(pl.Circle([0,0], 20, fill=False))
# t, posx, posy = brownian_molecule.get_time_pos()
# ax.plot(posx,posy)
# pl.show()

# plt.subplot(1,2,1)
# plt.plot(t, posx)
# plt.xlabel('Time (s)')
# plt.ylabel('x')
# plt.subplot(1,2,2)
# plt.plot(t, posy)
# plt.xlabel('Time (s)')
# plt.ylabel('y')
# plt.show()
# %%
""" 
Investigating effects of temperature on Brownian motion.
"""
time_temp = []
x_temp = []
y_temp = []

sd_array = [5,10,20,30,40,50]

iteration = 1
for sd in sd_array:
    brownian_molecule = brownian_ball(0.01, 5, [0,0], [0,0])
    atoms = p.BallsArray(container_r=20)
    atoms.manual_add_ball(brownian_molecule)
    atoms.brownian(50,0,sd,0.0001,0.25,6)

    brownian_motion = brownian_simulation(atoms)
    brownian_motion.run(1000, animate=False, histogram=False)

    t, x, y = brownian_molecule.get_time_pos()
    time_temp.append(t)
    x_temp.append(x)
    y_temp.append(y)

    print(iteration)
    iteration+=1
# %%
colors = ['red', 'orange', 'black', 'green', 'blue', 'indigo']
f = pl.figure()
ax = pl.axes(xlim=(-20, 20), ylim=(-20, 20))
ax.add_artist(pl.Circle([0,0], 20, fill=False))
t, posx, posy = brownian_molecule.get_time_pos()
for i in range(0,len(x_temp)):
    ax.plot(x_temp[i], y_temp[i], label=sd_array[i])
plt.show()

plt.figure(figsize=(8, 4), dpi=80)
for i in range(0,len(x_temp)):
    plt.subplot(1,2,1)
    plt.plot(time_temp[i], np.sqrt(np.array(x_temp[i])**2), label=sd_array[i], c=colors[i])
    plt.xlabel('Time (s)')
    plt.ylabel('x')
    plt.xlim((0,4.8))
    plt.subplot(1,2,2)
    plt.plot(time_temp[i], np.sqrt(np.array(y_temp[i])**2), label=sd_array[i], c=colors[i])
    plt.xlabel('Time (s)')
    plt.ylabel('y')
    plt.xlim((0,4.8))
    legend = plt.legend()
    legend.set_title("Velocity sd")  
plt.show()

plt.figure(figsize=(8, 4), dpi=80)
for i in range(0,len(x_temp)):
    plt.plot(time_temp[i], np.sqrt(np.array(x_temp[i])**2 + np.array(y_temp[i])**2), label=sd_array[i], c=colors[i]) 
    plt.xlabel('Time (s)')
    plt.ylabel('Distance from center (m)')
    plt.xlim((0,4.5))
    legend = plt.legend()
    legend.set_title("Velocity sd") 
    plt.grid()
plt.show()
# %%

