"""Module defining the particle object and its actions.
"""
#%%
import numpy as np
import types

class Ball():
    def __init__(self, m, r, p, v):
        self._m = m
        self._r = r
        self._p = np.array(p)[:].astype(np.float32)
        self._v = np.array(v)[:].astype(np.float32)
        self._patch = ""

    def pos(self):
        return self._p

    def vel(self):
        return self._v

    def move(self, dt):
        self._r = np.add(self._r, dt*self._v)

    def time_to_collision(self, other):
        r = np.subtract(self._p, other._p)
        v = np.subtract(self._v, other._v)
        R_pos = self._r + other._r
        R_neg = self._r - other._r

        def get_t(R):
            t = [-np.dot(r,v) + np.sqrt(np.dot(r,v)**2 - (r**2 - R**2)),\
                 -np.dot(r,v) - np.sqrt(np.dot(r,v)**2 - (r**2 - R**2))]
            return t

        t_array = get_t(R_pos) + get_t(R_neg)

        for i in t_array:
            if isinstance(i, types.ComplexType) == False:
                if i >=0:
                    return i

    def collide(self, other):
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
        mod_v1_par = np.dot(v1, r)/np.linalg.norm(r)
        mod_v2_par = np.dot(v2, r)/np.linalg.norm(r)
        mod_v1_par_new = ((m1-m2)/(m1+m2))*mod_v1_par + ((2*m2)/(m1+m1))*mod_v2_par
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

# %%
