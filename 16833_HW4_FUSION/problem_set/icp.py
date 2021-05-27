'''
    Initially written by Ming Hsiao in MATLAB
    Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
'''

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import argparse
import transforms
import o3d_utility


def find_projective_correspondence(source_points,
                                   source_normals,
                                   target_vertex_map,
                                   target_normal_map,
                                   intrinsic,
                                   T_init,
                                   dist_diff=0.07):
    '''
    \param source_points Source point cloud locations, (N, 3)
    \param source_normals Source point cloud normals, (N, 3)
    \param target_vertex_map Target vertex map, (H, W, 3)
    \param target_normal_map Target normal map, (H, W, 3)
    \param intrinsic Intrinsic matrix, (3, 3)
    \param T_init Initial transformation from source to target, (4, 4)
    \param dist_diff Distance difference threshold to filter correspondences
    \return source_indices: indices of points in the source point cloud with a valid projective correspondence in the target map, (M, 1)
    \return target_us: associated u coordinate of points in the target map, (M, 1)
    \return target_vs: associated v coordinate of points in the target map, (M, 1)
    '''
    h, w, _ = target_vertex_map.shape

    R = T_init[:3, :3]
    t = T_init[:3, 3:]

    # Transform source points from the source coordinate system to the target coordinate system
    T_source_points = (R @ source_points.T + t).T

    # Set up initial correspondences from source to target
    source_indices = np.arange(len(source_points)).astype(int)
    target_us, target_vs, target_ds = transforms.project(
        T_source_points, intrinsic)
    target_us = np.round(target_us).astype(int)
    target_vs = np.round(target_vs).astype(int)

    # TODO: first filter: valid projection
    mask = np.ones_like(target_us).astype(bool)
    # print("before filter 1 length", len(mask))
    mask[target_us>=w]=False
    mask[target_us < 0] = False
    mask[target_vs < 0] = False
    mask[target_vs >= h] = False
    mask[target_ds<0]= False
    # End of TODO


    source_indices = source_indices[mask]
    # print("after filter 1 length", len(source_indices))
    target_us = target_us[mask]
    target_vs = target_vs[mask]
    T_source_points = T_source_points[mask]

    # TODO: second filter: apply distance threshold
    mask = np.zeros_like(target_us).astype(bool)
    q = target_vertex_map[target_vs,target_us,:]
    subtract = q-T_source_points
    distance_square = subtract*subtract
    distance = np.sqrt(distance_square.sum(axis = 1))
    # print("before filter 2 length",len(mask))
    mask[distance<=dist_diff]=True
    # End of TODO

    source_indices = source_indices[mask]
    # print("after filter 2 length", len(source_indices))
    target_us = target_us[mask]
    target_vs = target_vs[mask]

    return source_indices, target_us, target_vs


def build_linear_system(source_points, target_points, target_normals, T):
    M = len(source_points)
    assert len(target_points) == M and len(target_normals) == M

    R = T[:3, :3]
    t = T[:3, 3:]

    p_prime = (R @ source_points.T + t).T
    q = target_points
    n_q = target_normals

    A = np.zeros((M, 6))
    b = np.zeros((M, ))


    # TODO: build the linear system
    n = n_q.T[np.newaxis, :, :]
    P_dash = np.zeros((3,6,M))
    P_dash[0,1,:] = p_prime[:,2]
    P_dash[0, 2, :] = -p_prime[:, 1]
    P_dash[1,0, :] = -p_prime[:, 2]
    P_dash[1, 2, :] = p_prime[:, 0]
    P_dash[2, 0, :] = p_prime[:, 1]
    P_dash[2, 1, :] = -p_prime[:, 0]
    P_dash[0,3,:]=np.ones((M,))
    P_dash[1, 4, :] = np.ones((M,))
    P_dash[2, 5, :] = np.ones((M,))

    A = np.einsum('ijk,jlk->ilk', n, P_dash)
    A = A[0,:,:].T


    dif = q-p_prime
    b_ = dif.T[:,np.newaxis, :]
    b = np.einsum('ijk,jlk->ilk', n, b_)
    b = b[0, :, :].T

    # for i in range(M):
    #     px = p_prime[i,0]
    #     py = p_prime[i, 1]
    #     pz = p_prime[i, 2]
    #     P = np.array([[0, pz, -py,1,0,0], [-pz, 0, px,0,1,0], [py, -px, 0,0,0,1]])
    #     A[i,:] = n_q[i,:] @ P
    #     b[i] = n_q[i,:]@np.transpose((q[i,:]-p_prime[i,:]))

    # End of TODO

    return A, b


def pose2transformation(delta):
    '''
    \param delta Vector (6, ) in the tangent space with the small angle assumption.
    \return T Matrix (4, 4) transformation matrix recovered from delta
    Reference: https://en.wikipedia.org/wiki/Euler_angles in the ZYX order
    '''
    w = delta[:3]
    u = np.expand_dims(delta[3:], axis=1)

    T = np.eye(4)

    # yapf: disable
    R = np.array([[
        np.cos(w[2]) * np.cos(w[1]),
        -np.sin(w[2]) * np.cos(w[0]) + np.cos(w[2]) * np.sin(w[1]) * np.sin(w[0]),
        np.sin(w[2]) * np.sin(w[0]) + np.cos(w[2]) * np.sin(w[1]) * np.cos(w[1])
    ],
    [
        np.sin(w[2]) * np.cos(w[1]),
        np.cos(w[2]) * np.cos(w[0]) + np.sin(w[2]) * np.sin(w[1]) * np.sin(w[0]),
        -np.cos(w[2]) * np.sin(w[0]) + np.sin(w[2]) * np.sin(w[1]) * np.cos(w[0])
    ],
    [
        -np.sin(w[1]),
        np.cos(w[1]) * np.sin(w[0]),
        np.cos(w[1]) * np.cos(w[0])
    ]])
    # yapf: enable

    T[:3, :3] = R
    T[:3, 3:] = u

    return T


def solve(A, b):
    '''
    \param A (6, 6) matrix in the LU formulation, or (N, 6) in the QR formulation
    \param b (6, 1) vector in the LU formulation, or (N, 1) in the QR formulation
    \return delta (6, ) vector by solving the linear system. You may directly use dense solvers from numpy.
    '''
    # TODO: write your relevant solver
    AA =  A.T @ A
    bb=A.T @ b
    x =  np.linalg.solve(AA,bb)
    # return np.zeros((6, ))
    return x.flatten()


def icp(source_points,
        source_normals,
        target_vertex_map,
        target_normal_map,
        intrinsic,
        T_init=np.eye(4),
        debug_association=False):
    '''
    \param source_points Source point cloud locations, (N, 3)
    \param source_normals Source point cloud normals, (N, 3)
    \param target_vertex_map Target vertex map, (H, W, 3)
    \param target_normal_map Target normal map, (H, W, 3)
    \param intrinsic Intrinsic matrix, (3, 3)
    \param T_init Initial transformation from source to target, (4, 4)
    \param debug_assocation Visualize association between sources and targets for debug
    \return T (4, 4) transformation from source to target
    '''

    T = T_init

    for i in range(10):
        # TODO: fill in find_projective_correspondences
        source_indices, target_us, target_vs = find_projective_correspondence(
            source_points, source_normals, target_vertex_map,
            target_normal_map, intrinsic, T)

        # Select associated source and target points
        corres_source_points = source_points[source_indices]
        corres_target_points = target_vertex_map[target_vs, target_us]
        corres_target_normals = target_normal_map[target_vs, target_us]

        # Debug, if necessary
        if debug_association:
            o3d_utility.visualize_correspondences(corres_source_points,
                                                  corres_target_points, T)

        # TODO: fill in build_linear_system and solve
        A, b = build_linear_system(corres_source_points, corres_target_points,
                                   corres_target_normals, T)
        delta = solve(A, b)

        # Update and output
        T = pose2transformation(delta) @ T
        loss = np.mean(b**2)
        print('iter {}: avg loss = {:.4e}, inlier count = {}'.format(
            i, loss, len(corres_source_points)))

    return T


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', help='path to the dataset folder containing rgb/ and depth/')
    parser.add_argument('--source_idx',
                        type=int,
                        help='index to the source depth/normal maps',
                        default=10)
    parser.add_argument('--target_idx',
                        type=int,
                        help='index to the source depth/normal maps',
                        default=50)
    args = parser.parse_args()

    intrinsic_struct = o3d.io.read_pinhole_camera_intrinsic('intrinsics.json')
    intrinsic = np.array(intrinsic_struct.intrinsic_matrix)

    depth_path = os.path.join(args.path, 'depth')
    normal_path = os.path.join(args.path, 'normal')

    # TUM convention -- uint16 value to float meters
    depth_scale = 5000.0

    # Source: load depth and rescale to meters
    source_depth = o3d.io.read_image('{}/{}.png'.format(
        depth_path, args.source_idx))
    source_depth = np.asarray(source_depth) / depth_scale

    # Unproject depth to vertex map (H, W, 3) and reshape to a point cloud (H*W, 3)
    source_vertex_map = transforms.unproject(source_depth, intrinsic)
    source_points = source_vertex_map.reshape((-1, 3))

    # Load normal map (H, W, 3) and reshape to point cloud normals (H*W, 3)
    source_normal_map = np.load('{}/{}.npy'.format(normal_path,
                                                   args.source_idx))
    source_normals = source_normal_map.reshape((-1, 3))

    # Similar preparation for target, but keep the image format for projective association
    target_depth = o3d.io.read_image('{}/{}.png'.format(
        depth_path, args.target_idx))
    target_depth = np.asarray(target_depth) / depth_scale
    target_vertex_map = transforms.unproject(target_depth, intrinsic)
    target_normal_map = np.load('{}/{}.npy'.format(normal_path,
                                                   args.target_idx))

    # Visualize before ICP
    o3d_utility.visualize_icp(source_points, target_vertex_map.reshape(
        (-1, 3)), np.eye(4))

    # TODO: fill-in components in ICP
    T = icp(source_points,
            source_normals,
            target_vertex_map,
            target_normal_map,
            intrinsic,
            np.eye(4),
            debug_association=False)

    # Visualize after ICP
    o3d_utility.visualize_icp(source_points, target_vertex_map.reshape(
        (-1, 3)), T)
