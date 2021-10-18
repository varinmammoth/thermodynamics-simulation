"""Ray tracing python module.
"""
#%%
import numpy as np
import types

from numpy.core.numeric import Inf

class ray():
    def __init__(self, p, k):
        """Initialises the ray with initial position p and initial direction k.

        Args:
            p (list): A list of length 3 with the intial x, y, z coordinates of the
            ray.
            k (list): A list of length 3 with the intial x, y, z direction of the
            ray.
        """        
        self._p = np.array([p])
        self._k = np.array([k])

    def p(self):
        """Returns the current x, y, z coordiantes of the ray.

        Returns:
            list: A list with the current x, y, z coordiantes of the ray.
        """
        return self._p[-1]

    def k(self):
        """Returns the current x, y, z direction of the ray.

        Returns:
            list: The current x, y, z direction of the ray.
        """
        return self._k[-1]

    def append(self, p, k):
        """Adds the new position and direction to the ray instance.

        Args:
            p (list): A list of length 3 with the intial x, y, z coordinates of the
            ray.
            k (list): A list of length 3 with the intial x, y, z direction of the
            ray.
        """
        self._p = np.concatenate((self._p, [p]))
        self._k = np.concatenate((self._k, [k]))
# %%
class OpticalElement:
    def propagate_ray(self, ray):
        "propagate a ray through the optical element"
        raise NotImplementedError()
# %%
class SphericalRefraction(OpticalElement):
    def __init__(self, z0, curv, aperRad, n1, n2):
        """Initialises spherical refraction element.

        Args:
            z0 (float): Intercept of element with z-axis.
            curv (float): Curvature. A positive curvature
                is when center is at location z>z0. A negative curvature
                is when center is at locationz<z0.
            aperRad (float): Apeture radius
            n1 (float): Refractive index outside elemtn.
            n2 ([type]): Refractive index of element.
        """
        self._z0 = z0
        self._curv = curv
        self._aperRad = aperRad
        self._n1 = n1
        self._n2 = n2

    def intercept(self, ray):
        if self._curv != Inf: #ie. the surface is not flat
            r = np.array([0,0,self._z0]) - ray.p()
            kHat = ray.k() / (ray.k()**2).sum()**0.5
            l1 = -np.dot(r, kHat) + np.sqrt(np.dot(r, kHat)**2 - (np.linalg.norm(r)**2 - (1/self._curv)**2))
            l2 = -np.dot(r, kHat) - np.sqrt(np.dot(r, kHat)**2 - (np.linalg.norm(r)**2 - (1/self._curv)**2))
            
            if isinstance(l1, types.ComplexType):
                if isinstance(l2, types.ComplexType):
                    return None
                else:
                    l = l2
            elif isinstance(l2, types.ComplexType):
                if isinstance(l1, types.ComplexType):
                    return None
                else:
                    l = l1
            elif abs(l1) <= abs(l2):
                l = l1
            else:
                l = l2
            
            return ray._p + l*kHat
        else:
            "some other function to deal with flat surfaces"
        

        
            
            
            
            
            
            
        