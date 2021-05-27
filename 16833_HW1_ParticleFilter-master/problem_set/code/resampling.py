'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
    Modified by Kerou Zhang (kerouz@andrew.cmu.edu), Tianxiang Lin (tianxian@andrew.cmu.edu), March 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """

    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """

        X_bar_resampled = np.zeros_like(X_bar)
        M = X_bar.shape[0]
        c = X_bar[:, 3]
        r = np.random.multinomial(M, c)

        ind = 0
        for m in range(M):
            while r[ind] == 0:
                ind += 1
            X_bar_resampled[m, :] = X_bar[ind, :]
            r[ind] -= 1

        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """

        X_bar_resampled = np.zeros_like(X_bar)
        M = X_bar.shape[0]
        r = np.random.uniform(0, 1.0 / M) % (1 / M)

        # Normalize
        X_bar[:, 3] = X_bar[:, 3] / np.sum(X_bar[:, 3])

        c = X_bar[0, 3]
        i = 0

        for m in range(1, M + 1):
            u = r + (m - 1) * 1.0 / M
            while u > c:
                i += 1
                c += X_bar[i, 3]
            X_bar_resampled[m - 1, :] = X_bar[i, :]

        return X_bar_resampled