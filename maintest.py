#%%
import particle as p
import matplotlib.pyplot as plt

plt.plot(1,1)
plt.show()
""" 
# One ball collision with container in x axis
# """
# ballarray = p.BallsArray()
# ballarray.manual_add_ball(p.Ball(1,1,[0,0],[1,0]))
# ballarray.manual_add_container()
# simulation = p.Simulation(ballarray)

# simulation.run(100,animate=True)

""" 
One ball collision with container in y axis
"""
# ballarray = p.BallsArray()
# ballarray.manual_add_ball(p.Ball(1,1,[0,0],[0,1]))
# ballarray.manual_add_container()
# simulation = p.Simulation(ballarray)

# simulation.run(100,animate=True)

""" 
One ball collision with container diagnally
"""
# ballarray = p.BallsArray()
# ballarray.manual_add_ball(p.Ball(1,1,[0.1,0.1],[1,1]))
# ballarray.manual_add_container()
# simulation = p.Simulation(ballarray)
# simulation.run(10, animate=True)

""" 
One ball collision with container in an arbirtary direction
doesn't work
"""
# ballarray = p.BallsArray()
# ballarray.manual_add_ball(p.Ball(1,1,[0.1,0.1],[-3.4,-7.2]))
# ballarray.manual_add_container()
# simulation = p.Simulation(ballarray)
# simulation.run(10, animate=True)

"""
Ball to ball collision
"""
ballarray = p.BallsArray()
ballarray.manual_add_ball(p.Ball(1,1,[5,0],[-5,0]))
ballarray.manual_add_ball(p.Ball(1,1,[-5,0],[5,0]))
ballarray.manual_add_container()
simulation = p.Simulation(ballarray)
simulation.run(10, animate=True)

# %%
import numpy as np
import cmath
import particle as p

self = p.Ball(1,1,[0.1,0.1],[-3.4,-7.2])
other = p.Ball(1e38, 10, [0,0], [0,0], type='container')

"""Return the time to the next collision of self with another object of class Ball.

Args:
    other (Ball): The other object self is colliding with.

Returns:
    float: Time for self to collide with other. Returns None if objects do not collide.
"""
r = np.subtract(self.pos(), other.pos())
v = np.subtract(self.vel(), other.vel())

#Check what self is colliding with.
#If colliding with another ball we use the R = r1 + r2 case
#else, if not colliding with ball then must be a container, so 
#use the R = r1 - r2 case
if (self._type == "ball" and other._type == "ball"):
    R = self._r + other._r
else:
    R = self._r - other._r

def get_t(R):
    #A list of length 2 with each of the solutions to the dt quadratic.
    t_array = [((-np.dot(r,v) + cmath.sqrt(np.dot(r,v)**2 - (np.dot(v,v))*(np.dot(r,r) - R**2))))/np.dot(v,v),\
            ((-np.dot(r,v) - cmath.sqrt(np.dot(r,v)**2 - (np.dot(v,v))*(np.dot(r,r) - R**2))))/np.dot(v,v)]
    return t_array

def get_pos_real(t_array):
    t_array_real = []
    for i in t_array:
        if (i.imag == 0) and (i.real > 0):
            t_array_real.append(i.real)

    if len(t_array_real) != 0:
        return np.min(t_array_real)
    else:
        return None

def get_neg_real(t_array):
    t_array_real = []
    for i in t_array:
        if (i.imag == 0) and (i.real < 0):
            t_array_real.append(i.real)

    if len(t_array_real) != 0:
        return np.max(t_array_real)
    else:
        return None

t_array = get_t(R)

overlap = abs(np.dot(r,r) - R**2)
epsilon = 1e-20

self._errorCorrectionMode = False
if self._type == 'ball' and other._type == 'ball':
    if overlap >= epsilon:
        time_to_collision = get_pos_real(t_array)
    else:
        time_to_collision = get_neg_real(t_array)
        self._errorCorrectionMode = True
else:
    if overlap <= epsilon:
        time_to_collision = get_pos_real(t_array)
    else:
        time_to_collision = get_neg_real(t_array)
        self._errorCorrectionMode = True


# %%
