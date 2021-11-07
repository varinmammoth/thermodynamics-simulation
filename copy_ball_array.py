""" 
We require the ability to deepcopy the ball_array,
which is and array of the ball instances. The normal
copy.deepcopy() does not work because there is an issue
with copying matplotlib objects, returning the error:
    TransformNode instances can not be copied. Consider using frozen() instead.
In this case, one of the attributes of the ball instances is the 
ball patch; this causes copy.deepcopy() to run into the problem.

We do not need the whole ball instance, only the velocities
to be copied. This script implements this function.
"""

def copy_vel_pairs(ball_array_pairs):
    copy = []
    for i in ball_array_pairs:
        a = [i[0].vel(), i[1].vel()]
        copy.append(a)
    return copy

def copy_vel(ball_array):
    copy = []
    for i in ball_array:
        copy.append(i.vel())
    return copy