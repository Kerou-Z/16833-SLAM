'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular
from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np
import matplotlib.pyplot as plt


def solve_default(A, b):
    from scipy.sparse.linalg import spsolve
    x = spsolve(A.T @ A, A.T @ b)
    return x, None


def solve_pinv(A, b):
    # TODO: return x s.t. Ax = b using pseudo inverse.
    N = A.shape[1]
    x = np.zeros((N, ))
    x = inv(A.T @ A)@A.T @ b
    return x, None


def solve_lu(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
    x = np.zeros((N, ))
    U = eye(N)

    superLU= splu(csc_matrix(A.T@A),permc_spec='NATURAL')
    U = superLU.U
    # print("nonzero amount of LU ", superLU.nnz)
    x = superLU.solve(A.T@b)
    return x, U


def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutration_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
    x = np.zeros((N, ))
    U = eye(N)

    superLU = splu(csc_matrix(A.T @ A), permc_spec='COLAMD')
    U = superLU.U
    # print("nonzero amount of LU_colamd ",superLU.nnz)
    x = superLU.solve(A.T @ b)
    return x, U

def solve_lu_colamd_hand(A, b):
    # TODO: hand written forward/backward substitution to compute x

    N = A.shape[1]
    x = np.zeros((N, ))
    y = np.zeros((N,))
    U = eye(N)

    AA = csc_matrix(A.T @ A)
    bb= A.T@b

    lu = splu(AA, permc_spec='COLAMD')
    U = lu.U
    L = lu.L

    Pr = csc_matrix((np.ones(N), (lu.perm_r, np.arange(N))))
    Pc = csc_matrix((np.ones(N), (np.arange(N), lu.perm_c)))
    pr_mul_b = Pr @ bb

    # y = spsolve_triangular(L, pr_mul_b, lower=True)
    y[0]=pr_mul_b[0]/L[0,0]
    for i in range(1,N):
        sum = L[i,:i]*y[:i]
        y[i] = (pr_mul_b[i]-sum)/L[i,i]

    # x = spsolve_triangular(U, y, lower=False)
    x[N-1] = y[N-1]/U[N-1,N-1]
    for j in range(N-2,-1,-1):
        sum = U[j,j+1:]*x[j+1:]
        x[j] = (y[j]-sum)/U[j,j]

    x = Pc@x

    return x, U


def solve_qr(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
    # https://github.com/theNded/PySPQR
    N = A.shape[1]
    x = np.zeros((N, ))
    R = eye(N)

    z, R, E, rank = rz(A,b,permc_spec='NATURAL')
    x = spsolve_triangular(R,z,lower=False)
    return x, R


def solve_qr_colamd(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2, with reordered QR decomposition (E is the permutation matrix).
    # https://github.com/theNded/PySPQR

    N = A.shape[1]
    x = np.zeros((N, ))
    R = eye(N)

    z, R, E, rank = rz(A, b, permc_spec='COLAMD')
    pre_x = spsolve_triangular(R, z, lower=False)

    x = permutation_vector_to_matrix(E)@pre_x

    return x, R


def solve(A, b, method='default'):
    '''
    \param A (M, N) Jacobian matirx
    \param b (M, 1) residual vector
    \return x (N, 1) state vector obtained by solving Ax = b.
    '''
    M, N = A.shape

    fn_map = {
        'default': solve_default,
        'pinv': solve_pinv,
        'lu': solve_lu,
        'qr': solve_qr,
        'lu_colamd': solve_lu_colamd,
        'qr_colamd': solve_qr_colamd,
        'lu_colamd_hand': solve_lu_colamd_hand,
    }

    return fn_map[method](A, b)
