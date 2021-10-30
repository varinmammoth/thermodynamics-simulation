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
    bounds = container_r-1
    while n < N:
        goodball=True
        x = random.uniform(-bounds, bounds)
        y = random.uniform(-bounds, bounds)
        point = np.array([x,y])
        for i in points:
            distance = np.linalg.norm(i - point)
            if distance < ball_r:
                goodball = False
                break
        if goodball:
            points.append(point)
            n+=1
    return np.array(points)


# %%
