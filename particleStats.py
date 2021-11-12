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

    hist_norm = hist/np.linalg.norm(hist)
    hist_norm_err = hist_err/np.linalg.norm(hist)

    return center, hist, hist_err, hist_norm, hist_norm_err