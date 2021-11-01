"""Module defining the particle object and its actions.
"""
#%%
import numpy as np
import pylab as pl
import generate_points as points

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
        self._errorCorrectionMode = False

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
        r = np.subtract(self.pos(), other.pos())
        v = np.subtract(self.vel(), other.vel())

        #Check what self is colliding with.
        #If colliding with another ball we use the R = r1 + r2 case
        #else, if not colliding with ball then must be a container, so 
        #use the R = r1 - r2 case
        if other._type == "ball":
            R = self._r + other._r
        else:
            R = self._r - other._r

        c = np.sqrt(np.dot(r,r) - np.dot(R,R)) #error (see labbook)
        epsilon = 1e-15 #error correction factor
        
        def get_t(R):
            #A list of length 2 with each of the solutions to the dt quadratic.
            t_array = [((-np.dot(r,v) + np.sqrt(np.dot(r,v)**2 - (np.dot(v,v))*(np.dot(r,r) - R**2))))/np.dot(v,v),\
                 ((-np.dot(r,v) - np.sqrt(np.dot(r,v)**2 - (np.dot(v,v))*(np.dot(r,r) - R**2))))/np.dot(v,v)]
            
            #checks elements of t_array and only returns the positive && real case
            
            if c < epsilon:
                self._errorCorrectionMode = False
                t_array_real = []
                for i in t_array:
                    if isinstance(i, complex) == False and (i > 0):
                        t_array_real.append(i)

                if len(t_array_real) != 0:
                    return np.min(t_array_real)
                else:
                    return None
            else:
                self._errorCorrectionMode = True
                t_array_real = []
                for i in t_array:
                    if isinstance(i, complex) == False and (i < 0):
                        t_array_real.append(i)
                
                if len(t_array_real) != 0:
                    return np.max(t_array_real)
                else:
                    return None
        
        #use the get_t function to return the positive && real time to next collision
        return get_t(R)

    def collide(self, other):
        """Updates the velocities of self and other after they collide.

        Args:
            other (Ball): The other object self is colliding with.
        
        Returns:
            (bool): Returns true if the collision is a ball-container collision
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

        if self._type == 'ball' and other._type == 'ball':
            return False
        else:
            return True

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

    def errorCorrectionMode():
        return self._errorCorrectionMode
# %%
class BallsArray():
    def __init__(self, container_r=10):
        self._ballarray = []
        self._container_r = container_r
        self._container = Ball(m=1e38, r=container_r, p=[0,0], v=[0,0], type="container")
        
    def get_array(self):
        """Return list with ball all ball objects. Last element is the container.

        Returns:
            list: List with all ball objects. Last element is the container.
        """
        return self._ballarray

    def get_all_pairs(self):
        """Returns all pairings of ball-ball and ball-container.

        Returns:
            list: List of length (n+1)c2 containing all pairings (ie. includes container).
        """
        #from stackoverflow
        return [(self._ballarray[i],self._ballarray[j]) for i in range(len(self._ballarray)) for j in range(i+1, len(self._ballarray))]
    
    def reset(self):
        self._ballarray = []

    def move_balls(self, dt):
        for ball in self._ballarray:
            ball.move(dt)

    def uniform(self, n, v, m, r):
        """Creates a list with ball objects with uniform mass and velocity.
            To return the list, use self.get_array()

        Args:
            n (int): number of balls
            v (list): [x,y] initial velocities of balls
            m (float): mass of balls
            r (float): radius of ball

        Returns:
            list: list of length n containing ball objects of unifrom mass and velocity
            randomly distributed
        """
        self._ballarray = []
        
        p_array = points.generate_points(n, r, self._container_r)

        #Create the ballarray
        for i in range (0,n):
            self._ballarray.append(Ball(m, r, [p_array[i][0], p_array[i][1]], v, type="ball"))

        self._ballarray.append(self._container)

    def manual_add_ball(self, newBall):
        self._ballarray.append(newBall)

    def manual_add_container(self):
        self._ballarray.append(self._container)
#%%

timeInterval = 50
class Simulation():
    def __init__(self, ballarray):
        self._ballarray = ballarray
        self._t = 0
        
        self._delta_p = 0 #change in momentume to calculate force
        self._delta_t = 0   #change in time to calculate force
        self._pressureArray = []
        self._timeArray = []

    def updateKE(self):
        KE = []
        for ball in self._ballarray:
            KE.append(ball.kinetic())
        self._KE = KE

    def next_collision(self):
        """Performs the next collision. Also updates self._timeArray and
            self._pressureArray.
        """
        times_to_collision = []
        for pair in self._ballarray.get_all_pairs():
            times_to_collision.append(pair[0].time_to_collision(pair[1]))
        
        #changes None type to some very large number, essentially infinity
        for i in range(0,len(times_to_collision)):
            if times_to_collision[i] == None:
                times_to_collision[i] = 1e10

        pair_index = np.argmin(times_to_collision)
        dt = np.min(times_to_collision)
        self._t += dt
        self._ballarray.move_balls(dt)
        isContainer = self._ballarray.get_all_pairs()[pair_index][0].collide(self._ballarray.get_all_pairs()[pair_index][1])
        if isContainer:
            #if ball collide with container, add 2*momentumBall to self._delta_p
            #but first, need to select which of the two in the pair is the ball
            if self._ballarray.get_all_pairs()[pair_index][0]._type == 'ball':
                ball = self._ballarray.get_all_pairs()[pair_index][0]
            else:
                ball = self._ballarray.get_all_pairs()[pair_index][1]
            self._delta_p += 2*ball.momentum()
        self._delta_t += dt

        # Whenever self._delta_t is greater than some value
        # timeInterval, self._delta_p and self._delta_t is used to calculate
        # the average pressure on the container, the pressure gets appended
        # to self._pressureArray, the time gets appended to self._timeArray,
        # then self._delta_p and self._delta_t is reset to 0.
        if self._delta_t > timeInterval:
            self._pressureArray.append((self._delta_p/self._delta_t)/(2*np.pi*(self._ballarray.get_array()[-1]._r**2))) 
            self._timeArray.append(self._t)
            self._delta_t = 0
            self._delta_p = 0

    def run(self, num_frames, animate=False):
        if animate:
            f = pl.figure()
            ax = pl.axes(xlim=(-self._ballarray.get_array()[-1]._r, self._ballarray.get_array()[-1]._r), ylim=(-self._ballarray.get_array()[-1]._r, self._ballarray.get_array()[-1]._r))
            ax.add_artist(self._ballarray.get_array()[-1].get_patch())
            for ball in self._ballarray.get_array()[0:-1]:
                ax.add_patch(ball.get_patch())
        for frame in range(num_frames):
            self.next_collision()
            if animate:
                self._ballarray.get_array()[0]
                pl.pause(0.001)
        if animate:
            pl.show()
        


# %%
