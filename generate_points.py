#%%
import random
import numpy as np

#random seed so we get reproducible results every single run
#choice of 30 was arbitrary
random.seed(30)

def generate_points(N, ball_r, container_r):
    """A function to randomly generate N non overlapping ball positions given a ball anc container radius.

    Args:
        N (int): Number of balls.
        ball_r (float): Radius of ball.
        container_r (float): Radius of container.

    Returns:
        list: A list of N [x,y] coordinates that can be used to generate N non overlapping balls inside
        the container.
    """
    points = []
    n = 0
    epsilon = 1e-1
    while n < N:
        goodball=True
        theta = random.uniform(0, 2*np.pi)
        rho = container_r*np.sqrt(random.uniform(0, 1))
        x = rho*np.cos(theta)
        y = rho*np.sin(theta)
        point = np.array([x,y])
        if np.sqrt(x**2 + y**2) > (container_r-ball_r-epsilon):
            goodball=False
        if goodball==True:
            for i in points:
                distance = np.linalg.norm(i - point)
                if distance < 2*ball_r:
                    goodball = False
                    break
        if goodball:
            points.append(point)
            n+=1
    print('Random initialization of balls done.')
    return np.array(points)

def generate_points_brownian(N, ball_r, container_r, brownian_r):
    """A function to randomly generate N non overlapping ball positions given a ball anc container radius,
    and also does not overlap with a central Brownian particle centered at the origin.

    Args:
        N (int): Number of balls.
        ball_r (float): Radius of ball.
        container_r (float): Radius of container.
        brownian_r (float): Radius of the Brownian particle.

    Returns:
        list: A list of N [x,y] coordinates that can be used to generate N non overlapping balls inside
        the container.
    
    """
    points = []
    n = 0
    bounds = container_r - ball_r
    while n < N:
        goodball=True
        theta = random.uniform(0, 2*np.pi)
        rho = container_r*np.sqrt(random.uniform(0, 1))
        x = rho*np.cos(theta)
        y = rho*np.sin(theta)
        point = np.array([x,y])
        if (np.sqrt(x**2 + y**2) > 0.9*container_r) or (np.sqrt(x**2 + y**2) < 2*brownian_r):
            goodball=False
        if goodball==True:
            for i in points:
                distance = np.linalg.norm(i - point)
                if distance < 2*ball_r:
                    goodball = False
                    break
        if goodball:
            points.append(point)
            n+=1
    print('Random initialization of balls done.')
    return np.array(points)
# %%
