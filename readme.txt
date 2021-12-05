Files in this folder:
readme.txt

Modules:
particle.py------------| Module to create Balls, BallsArray, and run the simulation.
particleStats.py-------| Module to generate histograms.
generate_points.py-----| Module used by particle.py to initialize N balls.
brownianmotion.py------| Module to investigate Brownian motion.

Runs:
maintest.py------------| Main testing file.
Plots.py---------------| Code used to generate plots for the report.

To get started:

Import nessecary modules. Modules to import:
particle.py
Optional:
particleStats.py (For easy histogram manipulation)
brownianmotion.py (For Brownian motion)

Steps for a basic simulation:
1) Create a BallsArray.
    ###
    ballarray = particle.BallsArray()

2) Add balls to BallsArray. There are two ways. 
    i) Using pre-written methods (see documentation for all possible initialization distributions
    and details on what to pass in), e.g.
    ###
    ballarray.uniform(20, [25,0], 1, 1)
    
    ii)Manually add balls. Can add as many balls. After done adding balls, add a container. Note: container
    must be added after all the balls have been added, e.g.
    ###
    ballarray.manual_add_ball(p.Ball(2,1,[-3,-3],[1,1]))
    ballarray.manual_add_ball(p.Ball(1,1,[0,0],[0,0], color='b'))
    ballarray.manual_add_container()

    To start over, rewrite the variable, or use
    ###
    ballarray.reset()

3) Create a simulation and pass in BallsArray.
    ###
    simulation = partilce.Simulation(ballarray)

4) Run the simulation. Set histogram=False to save memory if not making animated histograms.
    ###
    frames = 2000
    simulation.run(frames, animate=True, histogram=False)