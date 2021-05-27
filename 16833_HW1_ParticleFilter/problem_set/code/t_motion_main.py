'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
    Modified by Kerou Zhang (kerouz@andrew.cmu.edu), Tianxiang Lin (tianxian@andrew.cmu.edu), Jiayin Xia (jiayinx@andrew.cmu.edu), March 2021
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o', s=5)
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)
    scat.remove()


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    TODO : Add your code here
    This version converges faster than init_particles_random
    """
    # Length == num_particles, element = [x,y]-->occupancy map pixels whose prob is greater than 0.35
    # min_prob = 0.35
    # bool_map = occupancy_map[occupancy_map > min_prob]
    # cord = np.argwhere(bool_map==True)

    # occupancy map == 0 ----> free space
    cord = np.argwhere(occupancy_map == 0)
    cord[:,[0,1]] = cord[:,[1,0]]

    #pixel_array
    pixel_idx = np.random.randint(0, cord.shape[0], (num_particles))
    xy_vals = cord[pixel_idx,:]*10 + np.random.uniform(0, 10, (num_particles, 2))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    # X_bar_init = np.zeros((num_particles, 4))
    X_bar_init = np.hstack((xy_vals,theta0_vals, w0_vals))

    return X_bar_init


# if __name__ == '__main__':
#     """
#     Description of variables used
#     u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
#     u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
#     x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
#     x_t1 : particle state belief [x, y, theta] at time t [world_frame]
#     X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
#     z_t : array of 180 range measurements for each laser scan
#     """
#     """
#     Initialize Parameters
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path_to_map', default='../data/map/wean.dat')
#     parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
#     parser.add_argument('--output', default='results')
#     parser.add_argument('--num_particles', default=1, type=int)
#     parser.add_argument('--visualize', action='store_true')
#     args = parser.parse_args()
#
#     src_path_map = args.path_to_map
#     src_path_log = args.path_to_log
#     os.makedirs(args.output, exist_ok=True)
#
#     map_obj = MapReader(src_path_map)
#     occupancy_map = map_obj.get_map()
#     logfile = open(src_path_log, 'r')
#
#     motion_model = MotionModel()
#     sensor_model = SensorModel(occupancy_map)
#     resampler = Resampling()
#
#     num_particles = args.num_particles
#     # X_bar = init_particles_random(num_particles, occupancy_map)
#     X_bar = init_particles_freespace(num_particles, occupancy_map)
#     """
#     Monte Carlo Localization Algorithm : Main Loop
#     """
#     if args.visualize:
#         visualize_map(occupancy_map)
#
#     first_time_idx = True
#
#     X_bar = X_bar[:, 0:3]  # only take x,y, theta, no weights
#
#
#     for time_idx, line in enumerate(logfile):
#
#         # Read a single 'line' from the log file (can be either odometry or laser measurement)
#         # L : laser scan measurement, O : odometry measurement
#         meas_type = line[0]
#
#         # convert measurement values from string to double
#         meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')
#
#         # odometry reading [x, y, theta] in odometry frame
#         odometry_robot = meas_vals[0:3]
#         time_stamp = meas_vals[-1]
#
#         # ignore pure odometry measurements for (faster debugging)
#         # O x y theta ts
#         # if ((time_stamp <= 0.0) | (meas_type == "O")):
#         #     continue
#
#         # L x y theta xl yl thetal r1 ... r180 ts
#         if (meas_type == "L"):
#             # [x, y, theta] coordinates of laser in odometry frame
#             odometry_laser = meas_vals[3:6]
#             # 180 range measurement values from single laser scan
#             ranges = meas_vals[6:-1]
#
#         print("Processing time step {} at time {}s".format(
#             time_idx, time_stamp))
#
#         if first_time_idx: #initialize ut0 (robot coordinate) for time_idx=0
#             u_t0 = odometry_robot
#             record_particle = X_bar[0]
#             record_robot = u_t0
#             first_time_idx = False
#             continue
#
#         X_bar_new = np.zeros((num_particles, 3), dtype=np.float64)
#         u_t1 = odometry_robot
#
#         """
#         MOTION MODEL
#         """
#         x_t0 = X_bar[0]
#         x_t1 = motion_model.update(u_t0, u_t1, x_t0)
#         X_bar_new[0,:] = x_t1
#
#         X_bar = X_bar_new
#         u_t0 = u_t1
#         record_robot=np.vstack((record_robot,u_t1))
#         record_particle=np.vstack((record_particle,x_t1))
#
#     plt.figure(1)
#     plt.scatter(record_robot[:,0],record_robot[:,1],label="robot", c='b', marker='o', s=5)
#     plt.legend()
#     plt.figure(2)
#     plt.scatter(record_particle[:, 0], record_particle[:, 1],label="particle", c='r', marker='o', s=5)
#     plt.legend()
#     plt.pause(0)

if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=10, type=int)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    # X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)

    first_time_idx = True

    X_bar = X_bar[:, 0:3]  # only take x,y, theta, no weights
    print(X_bar)

    # Index of random sampled particles
    # particle_idx = 2


    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        # O x y theta ts
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        # L x y theta xl yl thetal r1 ... r180 ts
        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx: #initialize ut0 (robot coordinate) for time_idx=0
            u_t0 = odometry_robot
            u_t0_arr = np.tile(u_t0, (num_particles, 1))
            print(X_bar)
            record_particle = X_bar
            record_robot = u_t0
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 3), dtype=np.float64)
        u_t1 = odometry_robot
        u_t1_arr = np.tile(u_t1, (num_particles, 1))

        """
        MOTION MODEL
        """
        x_t0 = X_bar
        x_t1 = motion_model.update_vector(u_t0_arr, u_t1_arr, x_t0)
        X_bar_new = x_t1

        X_bar = X_bar_new
        u_t0 = u_t1
        u_t0_arr = np.tile(u_t0, (num_particles, 1))
        record_robot=np.vstack((record_robot,u_t1))
        record_particle=np.vstack((record_particle,x_t1))

    plt.figure(1)
    plt.scatter(record_robot[:,0],record_robot[:,1],label="robot", c='b', marker='o', s=5)
    plt.legend()
    plt.figure(2)
    plt.scatter(record_particle[:, 0], record_particle[:, 1],label="particle", c='r', marker='o', s=5)
    plt.legend()
    plt.pause(0)