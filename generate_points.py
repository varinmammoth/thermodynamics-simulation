#%%
import random
import numpy as np
import matplotlib.pyplot as plt

#random seed so we get reproducible results every single run
#choice of 30 was arbitrary
random.seed(30)

def generate_points(N, ball_r, container_r):
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
        if np.sqrt(x**2 + y**2) > 0.8*container_r:
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
        if (np.sqrt(x**2 + y**2) > 0.8*container_r) or (np.sqrt(x**2 + y**2) < 2*brownian_r):
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
