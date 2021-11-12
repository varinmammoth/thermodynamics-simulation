#%%
import particle as p
import matplotlib.pyplot as plt

import particleStats as ps

plt.plot(1,1)
plt.show()
""" 
# One ball collision with container in x axis
# """
# ballarray = p.BallsArray()
# ballarray.manual_add_ball(p.Ball(1,1,[0,0],[1,0]))
# ballarray.manual_add_container()
# simulation = p.Simulation(ballarray)

# simulation.run(10,animate=True)

""" 
One ball collision with container in y axis
"""
# ballarray = p.BallsArray()
# ballarray.manual_add_ball(p.Ball(1,1,[0,0],[0,1]))
# ballarray.manual_add_container()
# simulation = p.Simulation(ballarray)

# simulation.run(10,animate=True)

""" 
One ball collision with container diagnally
"""
# ballarray = p.BallsArray()
# ballarray.manual_add_ball(p.Ball(1,1,[0.1,0.1],[1,1]))
# ballarray.manual_add_container()
# simulation = p.Simulation(ballarray)
# simulation.run(10, animate=True)

""" 
One ball collision with container in an arbirtary initial position and direction
"""
# ballarray = p.BallsArray()
# ballarray.manual_add_ball(p.Ball(1,1,[6,-5],[-3,4], color='b'))
# ballarray.manual_add_container()
# simulation = p.Simulation(ballarray)
# simulation.run(10, animate=True)

"""
Ball to ball collision in on x-axis
"""
# ballarray = p.BallsArray()
# ballarray.manual_add_ball(p.Ball(1,1,[-3,0],[1,0]))
# ballarray.manual_add_ball(p.Ball(1,1,[3,0],[-1,0], color='b'))
# ballarray.manual_add_container()
# simulation = p.Simulation(ballarray)
# for pair in ballarray.get_all_pairs():
#     print(pair[0].pos(), pair[1].pos())
# simulation.run(10, animate=True)

"""
Ball to ball collision in on y-axis
"""
# ballarray = p.BallsArray()
# ballarray.manual_add_ball(p.Ball(1,1,[0,3],[0,-1]))
# ballarray.manual_add_ball(p.Ball(1,1,[0,-3],[0,1], color='b'))
# ballarray.manual_add_container()
# simulation = p.Simulation(ballarray)
# simulation.run(10, animate=True)

""" 
Ball to ball collision in an arbritrary direction
"""
# ballarray = p.BallsArray()
# ballarray.manual_add_ball(p.Ball(1,1,[-3,-3],[1,1]))
# ballarray.manual_add_ball(p.Ball(1,1,[0,0],[0,0], color='b'))
# ballarray.manual_add_container()
# simulation = p.Simulation(ballarray)
# simulation.run(10, animate=True)

"""
Ball to ball collision, but balls have different masses.
"""
# ballarray = p.BallsArray()
# ballarray.manual_add_ball(p.Ball(2,1,[-3,-3],[1,1]))
# ballarray.manual_add_ball(p.Ball(1,1,[0,0],[0,0], color='b'))
# ballarray.manual_add_container()
# simulation = p.Simulation(ballarray)
# simulation.run(10, animate=True)

"""
Try an arbritrary number of balls with uniform mass and initial velocity.
"""
# ballarray = p.BallsArray(container_r=20)
# ballarray.uniform(20, [25,0], 1, 1)
# simulation = p.Simulation(ballarray)
# simulation.run(2000, animate=False)

"""
Testing the simulation.get_pressure() function.
Experiment with various values of timeInterval in particle.py until get
sensible pressure plot.

"""
# ballarray = p.BallsArray(container_r=20)
# ballarray.uniform(15, [25,0], 1, 1)
# simulation = p.Simulation(ballarray)
# simulation.run(200, animate=True)
# t, P = simulation.get_pressure()
# plt.plot(t, P, 'x')
# plt.xlabel('Time (s)')
# plt.ylabel('Pressure (Pa)')
# plt.show()

# center, hist, hist_err, hist_norm, hist_norm_err = ps.get_histogram(P)
# plt.bar(center, hist)
# plt.xlabel('Pressure (Pa)')
# plt.ylabel('Frequency')
# plt.show()

"""
Testing the uniformly positioned but random velocity with fixed mean and s.d.
"""
ballarray = p.BallsArray(container_r=20)
ballarray.random_vel(100, 0, 1, 1, 1)
simulation = p.Simulation(ballarray)
simulation.run(500, animate=False)
timeDistance, ballsDistance, centerDistance = simulation.get_distances()

for i in range(0,len(timeDistance)):
    ballsDistanceHist = ps.get_histogram(ballsDistance[i], bins=20)
    centerDistanceHist = ps.get_histogram(centerDistance[i], bins=20)
    plt.subplot(1,2,1)
    plt.bar(ballsDistanceHist[0], ballsDistanceHist[1])
    plt.xlim((0,40))
    plt.ylim((0,2000))
    plt.xlabel('Distance between balls')
    plt.ylabel('Count')
    plt.subplot(1,2,2)
    plt.bar(centerDistanceHist[0], centerDistanceHist[1])
    plt.xlim((0,20))
    plt.ylim((0,50))
    plt.xlabel('Distance to center')
    plt.pause(0.00000001)
    plt.clf()

plt.show()
# %%
""" 
note to fix tmr:
when ball overlap with container, we get and error > 0.
Then we go and get t_array, but the t_array has no negative solutions.
This is because t_array was calculated using the new velocity after the collision,
however we need to calculate t_array using the velocity from the previous iteration.

note to fix:
implement while loop to check and correct error:
while there is an error: keep correcting
"""

# %%
