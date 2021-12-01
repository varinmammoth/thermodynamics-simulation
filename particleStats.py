import numpy as np

def get_histogram(x, bins = 10):
    """Function to obtain histogram bin centers and values
    for plotting.

    Args:
        x (list): List of data

    Returns:
        (tuple): bin center, histogram value, histogram value uncertainty, normalised histogram value, normalise histogram value uncertainty
    """
    hist, bins = np.histogram(x, bins=bins)
    center = (bins[:-1] + bins[1:]) / 2
    hist_err = np.sqrt(hist)

    totalarea = np.sum((center[1]-center[0])*np.array(hist))

    hist_norm = np.array(hist)/totalarea
    hist_norm_err = (np.array(hist_err)/np.array(hist))*np.array(hist_norm)

    return center, hist, hist_err, hist_norm, hist_norm_err

def maxwell(v,m,T):
    kb = 1.38e-23
    maxwell = np.exp((-1*m*v*v)/(2*kb*T))
    maxwell = maxwell*(m*v)/(kb*T)
    return maxwell