'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
    Modified by Tianxiang Lin (tianxian@andrew.cmu.edu), Kerou Zhang (kerouz@andrew.cmu.edu), Jiayin Xia (jiayinx@andrew.cmu.edu), March 2021
'''
from numba import cuda
from numba.cuda import cudamath
import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader
import argparse

class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """

        ##################################################################
        #Parameters
        self._z_hit = 6
        self._z_short = 0.2
        self._z_max = 1.01
        self._z_rand = 1000

        self._sigma_hit = 100
        self._lambda_short = 0.1

        self._max_range = 1000
        self._min_probability = 0.35
        self._subsampling = 5
        self._stride = 5

        # Sensor Offset from the center of the robot(cm) and Map Resolution(cm)
        self.sensor_offset = 25
        self.res = 10
        self.map = occupancy_map
        self.map = np.ascontiguousarray(self.map)
        ##################################################################



    def p_hit_vector(self, z_k_t, z_k_t_star):
        z_k_t = np.tile(z_k_t, (z_k_t_star.shape[0], 1))
        p_hit = np.zeros_like(z_k_t)

        # z_k_t >= 0
        p_hit_bool_0 = np.zeros_like(z_k_t, dtype=bool)
        p_hit_bool_0[z_k_t >= 0] = True
        # z_k_t <= self._max_range
        p_hit_bool_1 = np.zeros_like(z_k_t, dtype=bool)
        p_hit_bool_1[z_k_t - self._max_range <= 0] = True

        p_hit_bool = np.logical_and(p_hit_bool_0, p_hit_bool_1)

        eta = 1
        p_hit[p_hit_bool == True] = eta * norm.pdf(z_k_t[p_hit_bool == True], \
                                                   loc=z_k_t_star[p_hit_bool == True], scale=self._sigma_hit)
        # p_hit[p_hit_bool == False] = 0

        return p_hit

    def p_short_vector(self, z_k_t, z_k_t_star):
        z_k_t = np.tile(z_k_t, (z_k_t_star.shape[0], 1))
        p_short = np.zeros_like(z_k_t)

        # z_k_t >= 0
        p_short_bool_0 = np.zeros_like(z_k_t, dtype=bool)
        p_short_bool_0[z_k_t >= 0] = True
        # z_k_t <= z_k_t_star
        p_short_bool_1 = np.zeros_like(z_k_t, dtype=bool)
        p_short_bool_1[z_k_t - z_k_t_star <= 0] = True

        p_short_bool = np.logical_and(p_short_bool_0, p_short_bool_1)

        eta = np.ones_like(z_k_t)
        eta[p_short_bool == True] = 1.0 / (1 - np.exp(-self._lambda_short * z_k_t_star[p_short_bool == True]))

        p_short[p_short_bool == True] = eta[p_short_bool == True] * \
                                        self._lambda_short * \
                                        np.exp(-self._lambda_short * z_k_t[p_short_bool == True])
        # p_short[p_short_bool == False] = 0
        return p_short

    def p_max_vector(self, z_k_t, num_paticles):
        z_k_t = np.tile(z_k_t, (num_paticles, 1))
        p_max = np.zeros_like(z_k_t)

        p_max[z_k_t >= self._max_range] = 1
        p_max[z_k_t < self._max_range] = 0
        return p_max

    def p_rand_vector(self, z_k_t, num_paticles):
        z_k_t = np.tile(z_k_t, (num_paticles, 1))
        p_rand = np.zeros_like(z_k_t)

        # z_k_t >= 0
        p_rand_bool_0 = np.zeros_like(z_k_t, dtype=bool)
        p_rand_bool_0[z_k_t >= 0] = True
        # z_k_t <= z_k_t_star
        p_rand_bool_1 = np.zeros_like(z_k_t, dtype=bool)
        p_rand_bool_1[z_k_t < self._max_range] = True

        p_rand_bool = np.logical_and(p_rand_bool_0, p_rand_bool_1)

        p_rand[p_rand_bool == True] = 1.0 / self._max_range
        # p_rand[not(z_k_t >=0 or z_k_t < self._max_range)] = 0
        return p_rand

    def beam_range_finder_model(self, z_t1_arr, x_t1_arr):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1_arr : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        # Pose of sensor
        theta = x_t1_arr[:,2]
        x = x_t1_arr[:,0] + self.sensor_offset * np.cos(theta)
        y = x_t1_arr[:,1] + self.sensor_offset * np.sin(theta)
        theta = np.ascontiguousarray(theta)
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)

        num_particles = x_t1_arr.shape[0]
        max_angle = z_t1_arr.shape[0]
        arr_size = int(max_angle / self._subsampling)

        # Dim: 2 * arr_size * num_particles
        record = np.zeros((num_particles, arr_size, 2))
        record_dist = np.zeros((num_particles, arr_size))
        record = np.ascontiguousarray(record)
        record_dist = np.ascontiguousarray(record_dist)

        ######################################################################################
        # Ray Casting with CUDA

        col = 1024
        row = 1024
        block_dim = (16,16)
        grid_dim  = (int((col + block_dim[0] - 1)/ block_dim[0]) , int((row + block_dim[1]  - 1) / block_dim[1]))

        time_rc_start = time.time()

        map_device = cuda.to_device(self.map)
        record_device = cuda.device_array_like(record)
        record_dist_device = cuda.device_array_like(record_dist)
        theta_device = cuda.to_device(theta)
        x_device = cuda.to_device(x)
        y_device = cuda.to_device(y)

        ray_casting_kernel2[grid_dim, block_dim]( self._subsampling, map_device, self.res, \
                                                 self._min_probability, self._stride, x_device, y_device, \
                                                 theta_device, record_device, record_dist_device)
        cuda.synchronize()
        # print("gpu vector add time " + str(time() - start))
        record = record_device.copy_to_host()
        record_dist = record_dist_device.copy_to_host()

        time_rc_end = time.time()
        time_m = time_rc_end - time_rc_start
        # print('time ray casting cost', time_m, 's')
        #######################################################################################

        time_p_start = time.time()
        z_t1_star_arr = record_dist

        # Vectorization
        _p_hit = self.p_hit_vector(z_t1_arr[::self._subsampling], z_t1_star_arr)
        _p_short = self.p_short_vector(z_t1_arr[::self._subsampling], z_t1_star_arr)
        _p_max = self.p_max_vector(z_t1_arr[::self._subsampling], z_t1_star_arr.shape[0])
        _p_rand = self.p_rand_vector(z_t1_arr[::self._subsampling], z_t1_star_arr.shape[0])
        p = self._z_hit * _p_hit + self._z_short * _p_short + self._z_max * _p_max + self._z_rand * _p_rand
        # prob_zt1 = p[p >= 0]
        prob_zt1 = np.sum(np.log(p), axis=1)

        prob_zt1 = np.exp(prob_zt1)
        time_p_end = time.time()
        time_m = time_p_end - time_p_start
        # print('time prob calc cost', time_m, 's')

        return prob_zt1, record, record_dist

    ##########################################################
    # One-Step Parallel
    ##########################################################
    def beam_range_finder_model_one_step(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        # Debug
        # print("Sensor")
        prob_zt1 = 0

        # Pose of sensor
        theta = x_t1[2]
        x = x_t1[0] + self.sensor_offset * np.cos(theta)
        y = x_t1[1] + self.sensor_offset * np.sin(theta)

        arr_size = int(180 / self._subsampling)
        record = np.zeros((arr_size, 2))
        record_dist = -np.ones((arr_size,))
        record = np.ascontiguousarray(record)
        record_dist = np.ascontiguousarray(record_dist)

        ######################################################################################
        # Ray Casting with CUDA
        n = 180
        threads_per_block = 45
        blocks_per_grid = math.ceil(n / threads_per_block)

        time_rc_start = time.time()

        map_device = cuda.to_device(self.map)
        record_device = cuda.device_array_like(record)
        record_dist_device = cuda.device_array_like(record_dist)

        ray_casting_kernel[blocks_per_grid, threads_per_block]( self._subsampling, map_device, self.res, \
                                                             self._min_probability, self._stride, x, y, \
                                                             theta, record_device, record_dist_device)
        cuda.synchronize()
        # print("gpu vector add time " + str(time() - start))
        record = record_device.copy_to_host()
        record_dist = record_dist_device.copy_to_host()

        time_rc_end = time.time()
        time_m = time_rc_end - time_rc_start
        # print('time ray casting cost', time_m, 's')
        #######################################################################################

        time_p_start = time.time()
        # Sequential
        z_t1_star_arr = record_dist
        p_arr = -np.ones_like(record_dist)
        for k in range(0, arr_size):
            z_t1_k = z_t1_arr[k*self._subsampling]
            z_t1_star = record_dist[k]

            _p_hit = self.p_hit(z_t1_k, z_t1_star)
            _p_short = self.p_short(z_t1_k, z_t1_star)
            _p_max = self.p_max(z_t1_k)
            _p_rand = self.p_rand(z_t1_k)

            p = self._z_hit * _p_hit + self._z_short * _p_short + self._z_max * _p_max + self._z_rand * _p_rand
            prob_zt1 += np.log(p)
            p_arr[k] = p
        prob_zt1 = math.exp(prob_zt1)

        time_p_end = time.time()
        time_m = time_p_end - time_p_start


        ##for test raycasting
        return prob_zt1,record,record_dist

    def ray_casting(self, ray_theta, x, y, theta):
        x_end = x
        y_end = y

        x_idx = int(np.around(x_end / self.res))
        y_idx = int(np.around(y_end / self.res))

        # Debug
        # print(x_idx, y_idx)

        direction = theta + ray_theta

        prob_freespace = self.map[y_idx, x_idx]

        while prob_freespace < self._min_probability and \
                prob_freespace >= 0 and \
                x_idx >= 0 and x_idx < 800 and \
                y_idx >= 0 and y_idx < 800:
            prob_freespace = self.map[y_idx, x_idx]
            x_end += self._stride * np.cos(direction)
            y_end += self._stride * np.sin(direction)
            x_idx = int(np.around(x_end / self.res))
            y_idx = int(np.around(y_end / self.res))

        dist = math.sqrt((x - x_end) ** 2 + (y - y_end) ** 2)

        return dist, x_idx, y_idx

    def check_prob(self):
        plt.figure()
        # for x in (self._max_range,)
        for z_t1_k in range(0,self._max_range+100,10):
            z_t1_star=10 #z_k_t_star
            #x: real value by raycasting z_k_t

            _p_hit = self.p_hit(z_t1_k, z_t1_star)
            _p_short = self.p_short(z_t1_k, z_t1_star)
            _p_max = self.p_max(z_t1_k)
            _p_rand = self.p_rand(z_t1_k)
            # plt.scatter(z_t1_k, _p_hit, c="r")



            p = self._z_hit * _p_hit + self._z_short * _p_short + self._z_max * _p_max + self._z_rand * _p_rand
            # plt.scatter(z_t1_k,_p_short,c = "r")
            plt.scatter(z_t1_k, p, c="b")
        plt.title("z* = 10")

        plt.figure()
        # for x in (self._max_range,)
        for z_t1_k in range(0, self._max_range + 100, 10):
            z_t1_star = 500  # z_k_t_star
            # x: real value by raycasting z_k_t

            _p_hit = self.p_hit(z_t1_k, z_t1_star)
            _p_short = self.p_short(z_t1_k, z_t1_star)
            _p_max = self.p_max(z_t1_k)
            _p_rand = self.p_rand(z_t1_k)
            # plt.scatter(z_t1_k, _p_hit, c="r")

            p = self._z_hit * _p_hit + self._z_short * _p_short + self._z_max * _p_max + self._z_rand * _p_rand
            # plt.scatter(z_t1_k, _p_short, c="r")
            plt.scatter(z_t1_k, p, c="b")
        plt.title("z* = 500")
        plt.show()

######################################################################
# One-step Parallel
@cuda.jit
def ray_casting_kernel(_subsampling, map, res, _min_probability, _stride, x_begin, y_begin, \
                       theta, record, record_dist):
    # Angle
    n = 180
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if (idx >= n /_subsampling):
        return

    x_end = x_begin
    y_end = y_begin

    x_idx = int(round(x_end / res))
    y_idx = int(round(y_end / res))

    theta_radians = (idx * _subsampling - 90) * math.pi / 180
    direction = theta + theta_radians

    prob_freespace = map[y_idx, x_idx]

    while prob_freespace < _min_probability and \
            prob_freespace >= 0 and \
            x_idx >= 0 and x_idx < 800 and \
            y_idx >= 0 and y_idx < 800:
        prob_freespace = map[y_idx, x_idx]
        x_end += _stride * math.cos(direction)
        y_end += _stride * math.sin(direction)
        x_idx = int(round(x_end / res))
        y_idx = int(round(y_end / res))

    record[idx, 0] = x_idx
    record[idx, 1] = y_idx
    record_dist[idx] = math.sqrt((x_begin - x_end) ** 2 + (y_begin - y_end) ** 2)

######################################################################
# Fully Parallel
@cuda.jit
def ray_casting_kernel2(_subsampling, map, res, _min_probability, _stride, x_arr, y_arr, \
                       theta_arr, record, record_dist):
    # Angle
    max_angle = record.shape[1] * _subsampling
    particle_idx = cuda.gridDim.x * cuda.blockIdx.x + cuda.blockIdx.y
    angle_idx    = cuda.blockDim.x * cuda.threadIdx.x + cuda.threadIdx.y
    if (particle_idx > x_arr.shape[0] or angle_idx >= record.shape[1]):
        return

    x_end = x_arr[particle_idx]
    y_end = y_arr[particle_idx]

    x_idx = int(round(x_end / res))
    y_idx = int(round(y_end / res))

    theta_radians = (angle_idx * _subsampling - 90) * math.pi / 180
    direction = theta_arr[particle_idx] + theta_radians

    prob_freespace = map[y_idx, x_idx]

    while prob_freespace < _min_probability and \
            prob_freespace >= 0 and \
            x_idx >= 0 and x_idx < 800 and \
            y_idx >= 0 and y_idx < 800:
        prob_freespace = map[y_idx, x_idx]
        x_end += _stride * math.cos(direction)
        y_end += _stride * math.sin(direction)
        x_idx = int(round(x_end / res))
        y_idx = int(round(y_end / res))

    record[particle_idx, angle_idx, 0] = x_idx
    record[particle_idx, angle_idx, 1] = y_idx
    record_dist[particle_idx, angle_idx] = math.sqrt((x_arr[particle_idx] - x_end) ** 2 + \
                                                     (y_arr[particle_idx] - y_end) ** 2)


#for showing shape of the sum of all p
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')

    args = parser.parse_args()

    src_path_map = args.path_to_map

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()

    sensor_model = SensorModel(occupancy_map)

    sensor_model.check_prob()