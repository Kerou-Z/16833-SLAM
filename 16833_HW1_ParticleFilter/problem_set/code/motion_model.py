'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
    Modified by Tianxiang Lin (tianxian@andrew.cmu.edu), March 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.01
        self._alpha2 = 0.01
        self._alpha3 = 0.01
        self._alpha4 = 0.01

    def update_vector(self, u_t0_arr, u_t1_arr, x_t0_arr):
        """
        param[in] u_t0_arr : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1_arr : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0_arr : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1_arr : particle state belief [x, y, theta] at time t [world_frame]
        """
        # u_t1_y - u_t0_y
        delta_y = u_t1_arr[:,1] - u_t0_arr[:,1]
        # u_t1_x - u_t0_x
        delta_x = u_t1_arr[:,0] - u_t0_arr[:,0]
        # u_t1_theta - u_t0_theta
        delta_theta = u_t1_arr[:,2] - u_t0_arr[:,2]


        delta_r1 = np.arctan2(delta_y, delta_x) - u_t0_arr[:, 2]
        delta_t = np.sqrt(delta_x * delta_x + delta_y * delta_y)
        delta_r2 = delta_theta - delta_r1

        delta_r1_bar = delta_r1 - self.sample(self._alpha1 * delta_r1 + self._alpha2 * delta_t)
        delta_t_bar = delta_t - self.sample(self._alpha3 * delta_t+ self._alpha4 * (delta_r1 + delta_r2))
        delta_r2_bar = delta_r2 - self.sample(self._alpha1 * delta_r2 + self._alpha2 * delta_t)

        x_prime = x_t0_arr[:, 0] + delta_t_bar * np.cos(x_t0_arr[:, 2] + delta_r1_bar)
        y_prime = x_t0_arr[:, 1] + delta_t_bar * np.sin(x_t0_arr[:, 2] + delta_r1_bar)
        theta_prime = x_t0_arr[:, 2] + delta_r1_bar + delta_r2_bar

        x_t1_arr = np.stack((x_prime, y_prime, theta_prime), axis=1)
        return x_t1_arr


    def sample(self, b):
        return np.random.normal(0,scale = abs(b))