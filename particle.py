"""Module defining the particle object and its actions.
"""
#%%
import numpy as np
import pylab as pl
import generate_points as points

debug = False

class Ball():
    def __init__(self, m, r, p, v, type="ball", color='r'):
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
            self._patch = pl.Circle(self._p, self._r, fc=color, fill=True)
        else:
            self._patch = pl.Circle(self._p, self._r, fc='b', fill=False)

        self._v_past = self._v #velocity in the previous iteration

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

    def vel_past(self):
        return self._v_past

    def move(self, dt):
        """Updates the position of object to it's position dt seconds later.   

        Args:
            dt (float): Object's position is updated to the position dt seconds later.
        """
        self._p = np.add(self._p, dt*self._v)
        #updates patch
        self._patch.center = self._p
    
    def move_correct(self, dt):
        self._p = np.add(self._p, dt*self._v_past)

    def correct_error(self, other, epsilon=1e-3):
        if (self._type == "ball" and other._type == "ball"):
            R = self._r + other._r
        else:
            R = self._r - other._r

        r = np.subtract(self.pos(), other.pos())
        
        error = np.dot(r,r) - R**2

        if self._type == 'ball' and other._type == 'ball':
            if error < 0:
                self._p = np.subtract(self._p, epsilon*self.vel_past())
                #updates patch
                self._patch.center = self._p
        else:
            if error > 0:
                self._p = np.subtract(self._p, epsilon*self.vel_past())

    def time_to_collision(self, other):
        """Return the time to the next collision of self with another object of class Ball.

        Args:
            other (Ball): The other object self is colliding with.

        Returns:
            float: Time for self to collide with other. Returns None if objects do not collide.
        """
        #Check what self is colliding with.
        #If colliding with another ball we use the R = r1 + r2 case
        #else, if not colliding with ball then must be a container, so 
        #use the R = r1 - r2 case
        if (self._type == "ball" and other._type == "ball"):
            R = self._r + other._r
        else:
            R = self._r - other._r

        r = np.subtract(self.pos(), other.pos())
        v = np.subtract(self.vel(), other.vel())
        
        def get_t(R):
            #A list of length 2 with each of the solutions to the dt quadratic.
            t_array = [((-np.dot(r,v) + np.sqrt(np.dot(r,v)**2 - (np.dot(v,v))*(np.dot(r,r) - R**2))))/np.dot(v,v),\
                 ((-np.dot(r,v) - np.sqrt(np.dot(r,v)**2 - (np.dot(v,v))*(np.dot(r,r) - R**2))))/np.dot(v,v)]
            return t_array
        
        def get_pos_real(t_array):
            #returns the smallest positive real solution
            t_array_real = []
            for i in t_array:
                if (np.isnan(i) == False) and (i > 0):
                    t_array_real.append(i.real)

            if len(t_array_real) != 0:
                return np.min(t_array_real)
            else:
                return None

        t_array = get_t(R)

        error = np.dot(r,r) - R**2
        if self._type == 'ball' and other._type == 'ball':
            if error < 0:
                v = np.subtract(self.vel_past(), other.vel_past())
            else:
                v = np.subtract(self.vel(), other.vel())
        else:
            if error > 0:
                v = np.subtract(self.vel_past(), other.vel_past())
            else:
                v = np.subtract(self.vel(), other.vel())

        def get_t(R):
            #A list of length 2 with each of the solutions to the dt quadratic.
            t_array = [((-np.dot(r,v) + np.sqrt(np.dot(r,v)**2 - (np.dot(v,v))*(np.dot(r,r) - R**2))))/np.dot(v,v),\
                 ((-np.dot(r,v) - np.sqrt(np.dot(r,v)**2 - (np.dot(v,v))*(np.dot(r,r) - R**2))))/np.dot(v,v)]
            return t_array
        
        def get_pos_real(t_array):
            #returns the smallest positive real solution
            t_array_real = []
            for i in t_array:
                if (np.isnan(i) == False) and (i > 0):
                    t_array_real.append(i.real)
            if len(t_array_real) != 0:
                return np.min(t_array_real)
            else:
                return None

        def get_neg_real(t_array):
            #returns the largest negative real solution
            t_array_real = []
            for i in t_array:
                if (np.isnan(i) == False) and (i < 0):
                    t_array_real.append(i.real)

            if len(t_array_real) != 0:
                return np.max(t_array_real)
            else:
                return None

        t_array = get_t(R)

        if self._type == 'ball' and other._type == 'ball':
            if error < 0:
                time_to_collision = get_neg_real(t_array)
            else:
                time_to_collision = get_pos_real(t_array)
        else:
            if error > 0:
                time_to_collision = get_neg_real(t_array)
            else:
                time_to_collision = get_pos_real(t_array)

        return time_to_collision
            
    def collide(self, other):
        """Updates the velocities of self and other after they collide.

        Args:
            other (Ball): The other object self is colliding with.
        
        Returns:
            (bool): Returns true if the collision is a ball-container collision
        """
        #before updating self._v, update self._v_past. Same for other.
        self._v_past = self.vel()
        other._v_past = other.vel()
        
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
            if ball._type == 'ball':
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

    def random_vel(self, n, v_avg, sd, m, r):
        """Creates a list with ball objects with uniform mass and random velocities
            with average velocity v_avg and standard deviation sd.
            To return the list, use self.get_array()

        Args:
            n (int): number of balls
            v (list): [x,y] average initial velocities of balls
            m (float): mass of balls
            r (float): radius of ball

        Returns:
            list: list of length n containing ball objects of unifrom mass and velocity
            randomly distributed
        """
        self._ballarray = []
        
        p_array = points.generate_points(n, r, self._container_r)

        v = []
        for i in range(0,n):
            v.append([np.random.normal(v_avg, sd), np.random.normal(v_avg, sd)])

        #Create the ballarray
        for i in range (0,n):
            self._ballarray.append(Ball(m, r, [p_array[i][0], p_array[i][1]], v[i], type="ball"))

        self._ballarray.append(self._container)

    def manual_add_ball(self, newBall):
        self._ballarray.append(newBall)

    def manual_add_container(self):
        self._ballarray.append(self._container)

    def dist_between_balls(self):
        distance_array = []
        for pair in self.get_all_pairs()[:-1]:
            distance_array.append(np.linalg.norm(pair[1].pos() - pair[0].pos()))
        return distance_array

    def dist_to_center(self):
        distance_array = []
        center = self.get_array()[-1].pos()
        for ball in self.get_array()[:-1]:
            distance_array.append(np.linalg.norm(ball.pos() - center))
        return distance_array

    def vel_all_balls(self):
        vel_array = []
        velx_array = []
        vely_array = []
        for ball in self.get_array()[:-1]:
            vel_array.append(np.linalg.norm(ball.vel()))
            velx_array.append(ball.vel()[0])
            vely_array.append(ball.vel()[1])
        return (velx_array, vely_array, vel_array)
#%%

timeInterval = 0.5
class Simulation():
    def __init__(self, ballarray):
        self._ballarray = ballarray
        self._t = 0

        self._delta_p = 0 #change in momentume to calculate force
        self._delta_t = 0   #change in time to calculate force
        self._pressureArray = []
        self._pressureTimeArray = []

        self._distanceTimeArray = []
        self._distanceToCenter = [] #distances of balls to center at each time
        self._distanceToBalls = []  #distances of balls to balls at each time
        self._velArray = [] #velocities magnitude of balls at each time
        self._velxArray = [] 
        self._velyArray = []

        self._errorCorrectionMode = False

    def updateKE(self):
        KE = []
        for ball in self._ballarray:
            KE.append(ball.kinetic())
        self._KE = KE

    def next_collision(self):
        """Performs the next collision. Also updates self._pressureTimeArray and
            self._pressureArray.
        """
        #check for any overlaps and corrects them
        for pair in self._ballarray.get_all_pairs():
            pair[0].correct_error(pair[1])

        times_to_collision = []
        for pair in self._ballarray.get_all_pairs():
            times_to_collision.append(pair[0].time_to_collision(pair[1]))
        
        #changes None type to some very large number, essentially infinity
        for i in range(0,len(times_to_collision)):
            if (times_to_collision[i] == None) or (np.isnan(i)):
                times_to_collision[i] = 1e15

        neg_times = []
        pos_times = []
        for i in times_to_collision:
            if i < 0:
                neg_times.append(i)
            else:
                pos_times.append(i)
            
        if len(neg_times) == 0:
            dt = np.min(times_to_collision)
            pair_indices = np.where(times_to_collision == dt)[0]
        else:
            dt = np.max(neg_times)
            pair_indices = np.where(times_to_collision == dt)[0]
            self._errorCorrectionMode = True
        
        if self._errorCorrectionMode == False:
            self._t += dt
            self._ballarray.move_balls(dt)

            self._distanceTimeArray.append(self._t)
            self._distanceToBalls.append(self._ballarray.dist_between_balls())
            self._distanceToCenter.append(self._ballarray.dist_to_center())
            self._velArray.append(self._ballarray.vel_all_balls()[0])
            self._velxArray.append(self._ballarray.vel_all_balls()[1])
            self._velyArray.append(self._ballarray.vel_all_balls()[2])
            
            for pair_index in pair_indices:
                isContainer = self._ballarray.get_all_pairs()[pair_index][0].collide(self._ballarray.get_all_pairs()[pair_index][1])
                if isContainer:
                    #if ball collide with container, add 2*momentumBall to self._delta_p
                    #but first, need to select which of the two in the pair is the ball
                    if self._ballarray.get_all_pairs()[pair_index][0]._type == 'ball':
                        ball = self._ballarray.get_all_pairs()[pair_index][0]
                    else:
                        ball = self._ballarray.get_all_pairs()[pair_index][1]
                    self._delta_p += 2*ball.momentum()
            if isContainer:
                self._delta_t += dt
        else:
            for pair_index in pair_indices:
                self._ballarray.get_all_pairs()[pair_index][0].move_correct(dt)
                self._ballarray.get_all_pairs()[pair_index][1].move_correct(dt)
            self._errorCorrectionMode = False

        # Whenever self._delta_t is greater than some value
        # timeInterval, self._delta_p and self._delta_t is used to calculate
        # the average pressure on the container, the pressure gets appended
        # to self._pressureArray, the time gets appended to self._pressureTimeArray,
        # then self._delta_p and self._delta_t is reset to 0.
        if self._delta_t > timeInterval:
            self._pressureArray.append((self._delta_p/self._delta_t)/(2*np.pi*(self._ballarray.get_array()[-1]._r**2))) 
            self._pressureTimeArray.append(self._t)
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
            if debug:
                print(' ')
                print('frame: ' + str(frame))
            self.next_collision()
            if animate:
                ax.set_title(frame)
                pl.pause(0.01)
        if animate:
            pl.show()
        
    def get_pressure(self):
        """Returns the time array and pressure array.

        Returns:
            list: Time.
            list: Pressure at corresponding time.
        """
        return self._pressureTimeArray, self._pressureArray

    def get_distances(self):
        """Returns time array, distance between balls array, and distances to center array.
        For example, distanceToBalls[i] is a list of distances between all balls at time 
        distanceTimeArray[i].

        Returns:
            list: Time
            list: Distance between all balls at corresponding time.
            list: Distance between balls and center at corresponding time.
        """
        return self._distanceTimeArray, self._distanceToBalls, self._distanceToCenter

    def get_velocities(self):
        """Returns time array, x component of velocity array, y component of velocity array,
        and velocity magnitude array.
        For example, vx[i] corresponds to tbhe x component of velocity at time distanceTimeArray[i].

        Returns:
            list: Time
            list: x component of velocity at corresponding time
            list: y component of velocity at corresponding time
            list: velocity magnitude at corresponding time
        """
        return self._distanceTimeArray, self._velArray, self._velxArray, self._velyArray


# %%
