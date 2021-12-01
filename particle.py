"""Module defining the particle object and its actions.
"""
#%%
import numpy as np
import pylab as pl
import generate_points as points

np.random.seed(30)
debug = False
kb = 1.38e-23

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
        """Return velocity of object in the past iteration.
        Returns:
            np.ndarray: Velocity of object in previous iteration, in the form [vx, vy].
        """
        return self._v_past

    def mass(self):
        """Return mass of ball object.

        Returns:
            mass (float): Mass of ball object
        """
        return self._m

    def move(self, dt):
        """Updates the position of object to it's position dt seconds later.   
        Args:
            dt (float): Object's position is updated to the position dt seconds later.
        """
        self._p = np.add(self._p, dt*self._v)
        #updates patch
        self._patch.center = self._p
    
    def move_correct(self, dt):
        """Moves the ball in the opposite direction of it's previous movement by taking in a negative dt.
        Args:
            dt (float): Object's position is corrected tothe position dt seconds ago. dt is a negative
        """
        self._p = np.add(self._p, dt*self._v_past)

    def correct_error(self, other, epsilon=1e-3):
        """A brute force approach to fixing overlaps. Ball is moved in the opposite direction of its
        previous movement by an amount proportional to epsilon.

        Args:
            other (Ball Object): The other ball self has collided with.
            epsilon (float, optional): The balls are moved backwards proportional to
            this amount. Defaults to 1e-3.
        """
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
        
        def get_t(R):
            """Returns the two solutions of the quadratic for finding time to next collision.

            Args:
                R (np.ndarray): A linear combination of the two ball objects' positions.

            Returns:
                list: List of length 2 with; each element is one of the solutions.
            """
            r_dot_v = np.dot(r,v)
            v_dot_v = np.dot(v,v)
            disc = np.sqrt(r_dot_v**2 - (v_dot_v)*(np.dot(r,r) - R**2))
            t_array = [((-1*r_dot_v + disc))/v_dot_v,\
                ((-1*r_dot_v - disc))/v_dot_v]
            return t_array
        
        def get_pos_real(t_array):
            """Returns smallest positive and real solution. For use in general non-overlap cases.

            Args:
                t_array (list): Array of the two solutions to the dt quadratic.

            Returns:
                float: time to next collision
            """
            t_array_real = []
            for i in t_array:
                if (np.isnan(i) == False) and (i > 0):
                    t_array_real.append(i.real)
            if len(t_array_real) != 0:
                return np.min(t_array_real)
            else:
                return None

        def get_neg_real(t_array):
            """Returns laragest negative and real solution. For use to correct overlap cases.

            Args:
                t_array (list): Array of the two solutions to the dt quadratic.

            Returns:
                float: (negative) time that makes the balls not overlap.
            """
            t_array_real = []
            for i in t_array:
                if (np.isnan(i) == False) and (i < 0):
                    t_array_real.append(i.real)

            if len(t_array_real) != 0:
                return np.max(t_array_real)
            else:
                return None

        error = np.dot(r,r) - R**2

        if self._type == 'ball' and other._type == 'ball':
            if error < 0:
                v = np.subtract(self.vel_past(), other.vel_past())
                t_array = get_t(R)
                time_to_collision = get_neg_real(t_array)
            else:
                v = np.subtract(self.vel(), other.vel())
                t_array = get_t(R)
                time_to_collision = get_pos_real(t_array)
        else:
            if error > 0:
                v = np.subtract(self.vel_past(), other.vel_past())
                t_array = get_t(R)
                time_to_collision = get_neg_real(t_array)
            else:
                v = np.subtract(self.vel(), other.vel())
                t_array = get_t(R)
                time_to_collision = get_pos_real(t_array)
        
        return time_to_collision
            
    def collide(self, other):
        """Updates the velocities of self and other after they collide.
        Args:
            other (Ball): The other object self is colliding with.
        
        Returns:
            (bool): Returns true if the collision is a ball-container collision
        """
        #before updating self._v, update self._v_past. Same for other. Used for error correction.
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
        """Returns kinetic energy of the ball.

        Returns:
            float: Kinetic energy of the ball.
        """
        return 0.5*self._m*(np.linalg.norm(self.vel())**2)

    def momentum(self):
        """Returns momentum of the ball.

        Returns:
            float: Momentum of the ball.
        """
        return self._m*(np.linalg.norm(self.vel()))


# %%
class BallsArray():
    def __init__(self, container_r=10):
        """Initialises BallsArray with all the nessecary attributes. After initialising the class,
        can now add balls either manually, or using one of the pre-written distributions.

        Args:
            container_r (float, optional): Radius of the container. Defaults to 10.
        """
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
        return [(self._ballarray[i],self._ballarray[j]) for i in range(len(self._ballarray)) for j in range(i+1, len(self._ballarray))]
    
    def reset(self):
        """Resets the BallsArray object to start from scratch. It will be as if this object were
        just created.
        """
        self._ballarray = []

    def move_balls(self, dt):
        """Move all balls to a time dt in the future.

        Args:
            dt (float): Time increment to move the balls by.
        """
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

        for i in range (0,n):
            self._ballarray.append(Ball(m, r, [p_array[i][0], p_array[i][1]], v, type="ball"))

        self._ballarray.append(self._container)

    def random_vel(self, n, v_avg, sd, m, r):
        """Creates a list with ball objects with uniform mass and random velocities
            with average velocity v_avg and standard deviation sd.
            To return the list, use self.get_array()
        Args:
            n (int): number of balls
            v_avg (float): average initial velocity of the balls
            sd (float): standard deviation of initial velocity
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

        for i in range (0,n):
            self._ballarray.append(Ball(m, r, [p_array[i][0], p_array[i][1]], v[i], type="ball"))

        self._ballarray.append(self._container)

    def brownian(self, n, v_avg, sd, m, r, brownian_r):
        """Creates a list with ball objects with uniform mass and random velocities
            with average velocity v_avg and standard deviation sd. The points will not
            occupy the center of the container up to a radius r_brownian away to make
            space for the Brownian particle.
            To return the list, use self.get_array()
        Args:
            n (int): number of balls
            v_avg (float): average initial velocity of the balls
            sd (float): standard deviation of initial velocity
            m (float): mass of balls
            r (float): radius of ball
            r_brownian (float): radius of the Brownian particle centered at origin.
        Returns:
            list: list of length n containing ball objects of unifrom mass and velocity
            randomly distributed
        """
        
        p_array = points.generate_points_brownian(n, r, self._container_r, brownian_r)

        v = []
        for i in range(0,n):
            v.append([np.random.normal(v_avg, sd), np.random.normal(v_avg, sd)])

        for i in range (0,n):
            self._ballarray.append(Ball(m, r, [p_array[i][0], p_array[i][1]], v[i], type="ball"))

        self._ballarray.append(self._container)

    def manual_add_ball(self, newBall):
        """Manually add a single ball to the BallsArray object.
        Care should be taken to first reset the BallsArray object to avoid confusion.

        Args:
            newBall (Ball Object): The ball object generated using the Ball class to be added to BallsArray.
        """
        self._ballarray.append(newBall)

    def manual_add_container(self):
        """Manually add the container to the BallsArray object.
        Care should be taken to only add the container once all the balls have been manually added.
        """
        self._ballarray.append(self._container)

    def dist_between_balls(self):
        """Returns the distance between all pairs of balls in a list.

        Returns:
            list: List of all the distances between all pairs of balls.
        """
        distance_array = []
        for pair in self.get_all_pairs()[:-1]:
            distance_array.append(np.linalg.norm(pair[1].pos() - pair[0].pos()))
        return distance_array

    def dist_to_center(self):
        """Returns the distance of the balls to the center of the container in a list.

        Returns:
            list: List of all the distances of the balls to the center of the container.
        """
        distance_array = []
        center = self.get_array()[-1].pos()
        for ball in self.get_array()[:-1]:
            distance_array.append(np.linalg.norm(ball.pos() - center))
        return distance_array

    def vel_all_balls(self):
        """Returns each component of velocity and magnitude of velocity at the instant in time this
        function is called.

        Returns:
            list: x-component of velocities of all the balls at this instance in time.
            list: y-component of velocities of all the balls at this instance in time.
            list: Magnitude of velocities of all the balls at this instance in time.
        """
        vel_array = []
        velx_array = []
        vely_array = []
        for ball in self.get_array()[:-1]:
            vel_array.append(np.linalg.norm(ball.vel()))
            velx_array.append(ball.vel()[0])
            vely_array.append(ball.vel()[1])
        return velx_array, vely_array, vel_array

    def energy(self):
        """Returns total energy and energy of individual balls at the instant in time this function
        is called.

        Returns:
            float: Total kinetic energy of the system at this instance in time.
            list: A list of the kinetic energy of each ball at this instance in time.
        """
        energy_individual = []
        for ball in self.get_array()[:-1]:
            energy_individual.append(ball.kinetic())
        energy_total = sum(energy_individual)
        return energy_total, energy_individual

    def momentum(self):
        """Returns total momentum and momentum of individual balls at the instant in time this
        function is called. 

        Returns:
            float: Total momentum of the system at this instace in time.
            list. A list of the momentum of each ball at this instance in time.
        """
        momentum_individual = []
        for ball in self.get_array()[:-1]:
            momentum_individual.append(ball.momentum())
        momentum_total = sum(momentum_individual)
        return momentum_total, momentum_individual
#%%

class Simulation():
    def __init__(self, ballarray):
        self._ballarray = ballarray
        self._t = 0

        #below are attributes relating to histogram animation
        self._delta_p = 0 #change in momentume to calculate force
        self._delta_t = 0   #change in time to calculate force
        self._pressureArray = []
        self._pressureTimeArray = []
        self._KETime = [True]
        self._KE = []
        self._total_delta_p = 0

        self._generalTimeArray = [] #time
        self._distanceToCenter = [] #distances of balls to center at each time
        self._distanceToBalls = []  #distances of balls to balls at each time
        self._velArray = [] #velocities magnitude of balls at each time
        self._velxArray = [] #velocities x-component of balls at each time
        self._velyArray = [] #velocities y-component of balls at each time
        self._energy_total = [] #total energy at each time
        self._energy_individual = [] #energy of individual balls at each time

        self._errorCorrectionMode = False

    def updateKE(self):
        kinetic = []
        for ball in self._ballarray.get_array():
            kinetic.append(ball.kinetic())
        self._KE.append(sum(kinetic))
        self._KETime.append(self._t)

    def next_collision(self, histogram=True, timeInterval=0.25):
        """Performs the next collision. Also updates self._pressureTimeArray and
            self._pressureArray.

            If histogram is set to True, various properties of the system will at each time will
            be calculated, along with a list of the times the values correspond to. 
            These can be returned using the corresponding functions. 
            Turning this function off will save mememory.
        Args:
            histogram (bool, optional): If set to True, velocity, inter-ball distances, ball to center 
            distance, etc will be available to be returned using the corresponding functions.
            Set to False to save memory. Defaults to True.
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
            self.updateKE()

            if histogram:
                self._generalTimeArray.append(self._t)
                self._distanceToBalls.append(self._ballarray.dist_between_balls())
                self._distanceToCenter.append(self._ballarray.dist_to_center())
                self._velArray.append(self._ballarray.vel_all_balls()[0])
                self._velxArray.append(self._ballarray.vel_all_balls()[1])
                self._velyArray.append(self._ballarray.vel_all_balls()[2])
                self._energy_total.append(self._ballarray.energy()[0])
                self._energy_individual.append(self._ballarray.energy()[1])
            
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
                    self._total_delta_p += 2*ball.momentum()
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
            self._pressureArray.append((self._delta_p/self._delta_t)/(2*np.pi*(self._ballarray.get_array()[-1]._r))) 
            self._pressureTimeArray.append(self._t)
            self._delta_t = 0
            self._delta_p = 0

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
            self.next_collision(histogram, timeInterval)
            if animate:
                ax.set_title(frame)
                print(frame)
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

    def get_distances(self, histogram=True):
        """Returns time array, distance between balls array, and distances to center array.
        For example, distanceToBalls[i] is a list of distances between all balls at time 
        distanceTimeArray[i].
        
        Args:
            histogram (bool, optional): Set histogram=False if histogram has not been generated during simulation run.
            If False, returns arrays for last frame of simulation. Defaults to True.
        Returns:
            list: Time
            list: Distance between all balls at corresponding time.
            list: Distance between balls and center at corresponding time.
        """
        if histogram == False:
            self._distanceToBalls.append(self._ballarray.dist_between_balls())
            self._distanceToCenter.append(self._ballarray.dist_to_center())
            return [self._t], self._distanceToBalls, self._distanceToCenter
        else:
            return self._generalTimeArray, self._distanceToBalls, self._distanceToCenter

    def get_velocities(self, histogram=True):
        """Returns time array, x component of velocity array, y component of velocity array,
        and velocity magnitude array.
        For example, vx[i] corresponds to tbhe x component of velocity at time distanceTimeArray[i].
        Args:
            histogram (bool, optional): Set histogram=False if histogram has not been generated during simulation run.
            If False, returns arrays for last frame of simulation. Defaults to True.
        Returns:
            list: Time
            list: x component of velocity at corresponding time
            list: y component of velocity at corresponding time
            list: velocity magnitude at corresponding time
        """
        if histogram == False:
            self._velxArray, self._velyArray, self._velArray = self._ballarray.vel_all_balls()
            return [self._t], self._velArray, self._velxArray, self._velyArray
        else:
            return self._generalTimeArray, self._velArray, self._velxArray, self._velyArray

    def get_energies(self, histogram=True):
        """Returns time array, total energy of the system array, and array of energy of individual balls.
        For example, energy_individual[i] is a list of length n_balls containing kinetic energies of each ball.

        Args:
            histogram (bool, optional): Set histogram=False if histogram has not been generated during simulation run. 
            If False, returns arrays for last frame of simulation. Defaults to True.

        Returns:
            list: Time
            list: Array of total energy of system at corresponding time
            list: Array of individual energies of balls at corresponding time
        """
        if histogram == False:
            self._energy_total, self._energy_individual = self._ballarray.energy()
            return [self._t], self._energy_total, self._energy_individual
        else:
            return self._generalTimeArray, self._energy_total, self._energy_individual

    def get_temp(self, N=50, histogram=True):
        """Returns temperature of the system.

        Args:
            N(int): Number of balls in the simulation.
            histogram (bool, optional): Set histogram=False if histogram has not been generated during simulation run. 
            If False, returns arrays for last frame of simulation. Defaults to True.

        Returns:
            list: Time
            list: Array of temperature at corresponding time.
        """
        if histogram == False:
            totalenergy = self.get_energies(histogram=False)[1]
            temp = totalenergy/(kb*N)
            return [self._t], temp
        else:
            temp = np.array(self._energy_total)/(kb*N)
            return self._generalTimeArray, temp

    def whole_average_pressure(self):
        """Returns average pressure calculated over the whole duration of the simulation.

        Returns:
            float: Average pressure calculated over the whole duration of the simulation.
        """
        pressure = (self._total_delta_p/self._t)/(2*np.pi*(self._ballarray.get_array()[-1]._r))
        return pressure
# %%