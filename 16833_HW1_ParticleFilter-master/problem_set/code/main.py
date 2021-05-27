'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
    Modified by Tianxiang Lin (tianxian@andrew.cmu.edu), Kerou Zhang (kerouz@andrew.cmu.edu), Jiayin Xia (jiayinx@andrew.cmu.edu), March 2021
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


    scat = plt.scatter(x_locs, y_locs, c='r', marker='o', s=3)

    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)

    scat.remove()




def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(4000, 6000, (num_particles, 1))
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


def init_particles_testing(num_particles, occupancy_map):
    #correct to initial position
    X_1=np.array([[4400,3920,-3.14,1/num_particles],\
                 [4560,3880,-2.7,1/num_particles],\
                  [4150,3990,3,1/num_particles],\
                  [4610,3920,2.9,1/num_particles]])
    # incorrect to initial position
    X_2 = np.array([[3570, 3290, 0, 1 / num_particles], \
                    [4040, 4900, 1.7, 1 / num_particles], \
                    [3940, 2860, -1.7, 1 / num_particles], \
                    [4610, 910, 1.6, 1 / num_particles]])
    take_x1 = X_1[:int(num_particles/2),:]
    take_x2 = X_2[:int(num_particles / 2), :]
    X_bar_init = np.vstack((take_x1,take_x2))

    return X_bar_init

def accelerated_mcl():
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
    time_start_all=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=500, type=int)
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
    # X_bar = init_particles_testing(num_particles, occupancy_map)
    # X_bar[0,:]=np.array([4150, 3990, 3, 1 / num_particles])
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)

    first_time_idx = True
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
        #    continue

        # L x y theta xl yl thetal r1 ... r180 ts
        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:  # initialize ut0 (robot coordinate) for time_idx=0
            u_t0 = odometry_robot

            # Duplicate vector u to an n*3 Matrix
            u_t0_arr = np.tile(u_t0, (num_particles, 1))

            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # Duplicate vector u to an n*3 Matrix
        u_t1_arr = np.tile(u_t1, (num_particles, 1))


        # Vecterized Version
        """
        MOTION MODEL
        """
        time_motion_start = time.time()
        x_t0_arr = X_bar[:, 0:3]
        x_t1_arr = motion_model.update_vector(u_t0_arr, u_t1_arr, x_t0_arr)
        time_motion_end = time.time()

        """
        SENSOR MODEL
        """
        time_sensor_start = time.time()

        ##################################
        # Full Acceleration
        ##################################
        if (meas_type == "L"):
            z_t = ranges  # measurement of robot, size: (180,)
            w_t, record_endpoint, record_dist = sensor_model.beam_range_finder_model(z_t, x_t1_arr)
            X_bar_new = np.hstack((x_t1_arr, np.reshape(w_t, (-1,1))))
        else:
            X_bar_new = np.hstack((x_t1_arr, np.reshape(X_bar[:, 3], (-1,1))))
        ##################################

        ##################################
        # One-Step Acceleration
        ##################################
        # for m in range(0, num_particles):
        #
        #     x_t1 = x_t1_arr[m, 0:3]
        #
        #     if (meas_type == "L"):
        #         z_t = ranges  # measurement of robot, size: (180,)
        #         w_t, record_endpoint,record_dist = sensor_model.beam_range_finder_model(z_t, x_t1)
        #
        #         X_bar_new[m, :] = np.hstack((x_t1, w_t))
        #     else:
        #         X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))
        ##################################
        time_sensor_end=time.time()

        X_bar = X_bar_new
        u_t0 = u_t1
        u_t0_arr = u_t1_arr

        """
        RESAMPLING
        """
        time_resamp_start=time.time()
        X_bar = resampler.low_variance_sampler(X_bar)
        time_resamp_end=time.time()

        # time_m = time_motion_end - time_motion_start  # Execution Time Cost
        # print('time motion cost', time_m, 's')
        # time_s = time_sensor_end - time_sensor_start  # Execution Time Cost
        # print('time sensor cost', time_s, 's')
        # time_r= time_resamp_end - time_resamp_start  # Execution Time Cost
        # print('time resampling cost', time_r, 's')

        timeused = time.time() - time_start_all
        # print('time used', timeused, 's')
        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output)




def main():

    accelerated_mcl()

if __name__ == '__main__':
    main()
