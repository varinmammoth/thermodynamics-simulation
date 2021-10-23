"""Module defining the particle object and its actions.
"""
#%%
import numpy as np
import types

from numpy.core.numeric import Inf
from numpy.lib.user_array import container

class Ball():
    def __init__(self, m, r, p, v, type="ball"):
        self._type = "ball"
        self._m = m
        self._r = r
        self._p = np.array(p)[:].astype(np.float32)
        self._v = np.array(v)[:].astype(np.float32)
        self._type = type
        self._patch = ""

    def pos(self):
        return self._p

    def vel(self):
        return self._v

    def move(self, dt):
        self._p = np.add(self._p, dt*self._v)

    def time_to_collision(self, other):
        r = np.subtract(self._p, other._p)
        v = np.subtract(self._v, other._v)

        def get_t(R):
            #A list of length 2 with each of the solutions to the dt quadratic.
            t_array = [(-np.dot(r,v) + np.sqrt(np.dot(r,v)**2 - (np.dot(v,v))*(np.dot(r,r) - R**2))),\
                 (-np.dot(r,v) - np.sqrt(np.dot(r,v)**2 - (np.dot(v,v))*(np.dot(r,r) - R**2)))]
            
            #checks elements of t_array and only returns the positive && real case
            for i in t_array:
                if isinstance(i, complex) == False:
                    if i >=0:
                        return i

        #Check what self is colliding with.
        #If colliding with another ball we use the R = r1 + r2 case
        #else, if not colliding with ball then must be a container, so 
        #use the R = r1 - r2 case
        if other._type == "ball":
            R = self._r + other._r
        else:
            R = self._r - other._r
        
        #use the get_t function to return the positive && real time to next collision
        return get_t(R)

    def collide(self, other):
        #the modulus of the component of v1 and v2 parallel to r,
        #the vector joining the center of two balls
        r = np.subtract(self._p, other._p)
        rhat = r/np.linalg.norm(r)
        v1 = self._v
        v2 = other._v
        m1 = self._m
        m2 = other._m

        #calculating the modulus of the component of v1 and v2
        #parallelt to r after the collision
        mod_v1_par = np.dot(v1, r)/np.linalg.norm(r)
        mod_v2_par = np.dot(v2, r)/np.linalg.norm(r)
        mod_v1_par_new = ((m1-m2)/(m1+m2))*mod_v1_par + ((2*m2)/(m1+m1))*mod_v2_par
        mod_v2_par_new = ((2*m1)/(m1+m2))*mod_v1_par + ((m2-m1)/(m1+m2))*mod_v2_par
        
        #turning the modulus into a vector by multiplying by rhat
        vec_v1_par = mod_v1_par*rhat
        vec_v2_par = mod_v2_par*rhat
        vec_v1_par_new = mod_v1_par_new*rhat
        vec_v2_par_new = mod_v2_par_new*rhat

        #getting the component of v1 and v2 perpendicular to r
        vec_v1_perp = np.subtract(v1, vec_v1_par)
        vec_v2_perp = np.subtract(v2, vec_v2_par)

        #Adding parallel and perpendicular components to get 
        #new v vectors and update the _v attribute of particle 1 and 2
        self._v = np.add(vec_v1_par_new, vec_v1_perp)
        other._v = np.add(vec_v2_par_new, vec_v2_perp)

# %%
container = Ball(m=9999999, r=10, p=[0,0], v=[0,0], type="container")
ball = Ball(m=1, r=1, p=[-5,0], v=[1,0])
class Simulation():
    t = 0
    def __init__(self, container, ball):
        self._container = container
        self._ball = ball

    def next_collision(self):
        dt = self._ball.time_to_collision(self._container)
        Simulation.t += dt
        self._ball.move(dt)
        self._ball.collide(self._container)

simulation1 = Simulation(container, ball)

simulation1.next_collision()
# %%
