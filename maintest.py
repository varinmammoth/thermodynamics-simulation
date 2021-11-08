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
# ballarray.manual_add_ball(p.Ball(1,1,[6,-5],[-3,4]))
# ballarray.manual_add_container()
# simulation = p.Simulation(ballarray)
# simulation.run(10, animate=True)

"""
Ball to ball collision in on x-axis
"""
# ballarray = p.BallsArray()
# ballarray.manual_add_ball(p.Ball(1,1,[-3,0],[1,0]))
# ballarray.manual_add_ball(p.Ball(1,1,[3,0],[-1,0]))
# ballarray.manual_add_container()
# simulation = p.Simulation(ballarray)
# for pair in ballarray.get_all_pairs():
#     print(pair[0].pos(), pair[1].pos())
# simulation.run(10, animate=True)

"""
Ball to ball collision in on y-axis
"""
ballarray = p.BallsArray()
ballarray.manual_add_ball(p.Ball(1,1,[0,3],[0,-1]))
ballarray.manual_add_ball(p.Ball(1,1,[0,-3],[0,1]))
ballarray.manual_add_container()
simulation = p.Simulation(ballarray)
simulation.run(10, animate=True)
# %%
""" 
note to fix tmr:
when ball overlap with container, we get and error > 0.
Then we go and get t_array, but the t_array has no negative solutions.
This is because t_array was calculated using the new velocity after the collision,
however we need to calculate t_array using the velocity from the previous iteration.

"""

# %%
