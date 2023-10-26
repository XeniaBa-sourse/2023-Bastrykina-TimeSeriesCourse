import numpy as np


def ED_distance(ts1, ts2):
    """
    Calculate the Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    ed_dist : float
        Euclidean distance between ts1 and ts2.
    """
    
    ed_dist = 0

    if len(ts1) != len(ts2):
      return -1
    
    for i,j in zip(ts1, ts2):
      ed_dist += (i-j)**2
    return ed_dist**(0.5)


def standard_deviation(ts, mu):
  sum = 0
  for t in ts:
    sum += t**2 - mu**2
  return (sum/len(ts))**0.5

def norm_ED_distance(ts1, ts2):
    """
    Calculate the normalized Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    norm_ed_dist : float
        The normalized Euclidean distance between ts1 and ts2.
    """

    n = len(ts1)
    mu1 = 1/n * sum(ts1)
    mu2 = 1/n * sum(ts2)
    sigma1 = standard_deviation(ts1, mu1)
    sigma2 = standard_deviation(ts2, mu2)
    norm_ed_dist = (abs(2*n*(1-(np.dot(ts1,ts2)-n*mu1*mu2)/(n*sigma1*sigma2))))**0.5 

    return norm_ed_dist


def DTW_distance(ts1, ts2, r=None):
    """
    Calculate DTW distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    r : float
        Warping window size.
    
    Returns
    -------
    dtw_dist : float
        DTW distance between ts1 and ts2.
    """

    m = len(ts1)
    DTW = np.zeros((m + 1, m + 1))
    DTW[:, :] = float('Inf')
    DTW[0, 0] = 0

    if r == None:
        for i in range(1, m + 1):
            for j in range(1, m + 1):
                cost = (ts1[i-1] - ts2[j-1])**2
                DTW[i, j] = cost + min(DTW[i-1, j], DTW[i-1, j-1], DTW[i, j-1])
    else:
        for i in range(1, m + 1):
            for j in range(max(1, i - int(np.floor(m * r))), min(m, i + int(np.floor(m * r))) + 1):
                cost = (ts1[i-1] - ts2[j-1])**2
                DTW[i, j] = cost + min(DTW[i-1, j], DTW[i-1, j-1], DTW[i, j-1])

    return DTW[m, m]