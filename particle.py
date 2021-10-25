"""Module defining the particle object and its actions.
"""
#%%
import numpy as np
import pylab as pl

from numpy.core.numeric import Inf
from numpy.lib.user_array import container

class Ball():
    def __init__(self, m, r, p, v, type="ball"):
        """Initialises Ball object with all the nessecary attributes.

        Args:
            m (float): Mass of the object.
            r (float): Radius of the object.
            p (list): Initial x, y coordinates of object in a list of length 2.
            v (list): Initial x, y components of object's velocity in a list of length 2.
            type (str, optional): Type of the object, either "ball" or "container". Defaults to "ball".
        """
        self._type = "ball"
        self._m = m
        self._r = r
        self._p = np.array(p)[:].astype(np.float32)
        self._v = np.array(v)[:].astype(np.float32)
        self._type = type
        if self._type == "ball":
            self._patch = pl.Circle(self._p, self._r, fc='r', fill=True)
        else:
            self._patch = pl.Circle(self._p, self._r, fc='b', fill=False)

    def pos(self):
        """Return current position of object.

        Returns:
            np.ndarray: Current position of object, in the form [x,y].
        """
        return self._p

    def vel(self):
        """Return current position of object.

        Returns:
            np.ndarray: Current velocity of object, in the form [vx, vy].
        """
        return self._v

    def move(self, dt):
        """Updates the position of object to it's position dt seconds later.   

        Args:
            dt (float): Object's position is updated to the position dt seconds later.
        """
        self._p = np.add(self._p, dt*self._v)
        #updates patch
        self._patch.center = self._p

    def time_to_collision(self, other):
        """Return the time to the next collision of self with another object of class Ball.

        Args:
            other (Ball): The other object self is colliding with.

        Returns:
            float: Time for self to collide with other. Returns None if objects do not collide.
        """
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
        """Updates the velocities of self and other after they collide.

        Args:
            other (Ball): The other object self is colliding with.
        """
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
        mod_v1_par = np.dot(v1, rhat)
        mod_v2_par = np.dot(v2, rhat)
        mod_v1_par_new = ((m1-m2)/(m1+m2))*mod_v1_par + ((2*m2)/(m1+m2))*mod_v2_par
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

    def get_patch(self):
        """Returns patch of ball for animation.

        Returns:
            Pylab patch: Pylab patch of ball for animation.
        """
        return self._patch

    def kinetic(self):
        return 0.5*self._m*(np.linalg.norm(self.vel())**2)

    def momentum(self):
        return self._m*(np.linalg.norm(self.vel()))
# %%
container = Ball(m=1e38, r=10, p=[0,0], v=[0,0], type="container")
ball = Ball(m=1, r=1, p=[-5,0], v=[1,0])
ball2 = Ball(m=1, r=1, p=[-6.708,-6.708], v=[1/np.sqrt(2),1/np.sqrt(2)])
timeInterval = 50
class Simulation():
    pressureCalculation = False
    def __init__(self, container, ball):
        self._container = container
        self._ball = ball
        self._totalKE = self._ball.kinetic()
        self._t = 0
        
        self._delta_p = 0 #change in momentume to calculate force
        self._delta_t = 0   #change in time to calculate force
        self._pressureArray = []
        self._timeArray = []

    def next_collision(self):
        dt = self._ball.time_to_collision(self._container)
        self._t += dt
        self._ball.move(dt)
        self._ball.collide(self._container)
        self._delta_p += 2*self._ball.momentum()
        self._delta_t += dt

        if self._delta_t > timeInterval:
            self._pressureArray.append((self._delta_p/self._delta_t)/(2*np.pi*(self._container._r**2))) 
            self._timeArray.append(self._t)
            self._delta_t = 0
            self._delta_p = 0

    def run(self, num_frames, animate=False):
        if animate:
            f = pl.figure()
            ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10))
            ax.add_artist(self._container.get_patch())
            ax.add_patch(self._ball.get_patch())
        for frame in range(num_frames):
            self.next_collision()
            if animate:
                pl.pause(0.001)
                print(self._totalKE)
        if animate:
            pl.show()
        
    

simulation1 = Simulation(container, ball)
simulation2 = Simulation(container, ball2)

simulation1.next_collision()

# %%
