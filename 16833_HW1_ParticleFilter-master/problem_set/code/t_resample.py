'''
    Test for Resampling
    Created by Kerou Zhang (kerouz@andrew.cmu.edu), March 2021
'''

from resampling import Resampling
import numpy as np

resampler = Resampling()
X_bar_old = np.array([[1, 1, 1, 0.1], [2, 2, 2, 0.3], [3, 3, 3, 0.5], [4, 4, 4, 0.1]])
# X_bar_old=np.array([[ 4.23145049e+03 , 4.88460697e+02,  3.97330028e-01 , 0.00000000e+00],
#  [ 3.56651485e+03,  6.01121983e+03 ,-1.39517995e+00,  0.00000000e+00],
#  [ 6.74274565e+03 , 2.79737332e+03 ,-2.84567227e+00 , 0.00000000e+00],
#  [ 3.95994472e+03 , 2.90155626e+03, -6.45471164e-01,  0.00000000e+00],
#  [ 3.06260549e+03,  1.90083570e+02 ,-1.21025363e+00 , 0.00000000e+00],
#  [ 5.26535625e+03 , 6.13898387e+03 , 3.96037607e-02 , 0.00000000e+00],
#  [ 6.63270889e+03 , 3.75897283e+03 , 4.12651794e-01 , 0.00000000e+00],
#  [ 3.77851761e+03,  3.87077587e+03 , 5.38755712e-01,  0.00000000e+00],
#  [ 5.24716543e+03,  2.61908913e+03 ,-2.11457452e+00,  0.00000000e+00],
#  [ 4.68473018e+03,  1.35781453e+03,  3.60145275e-01 , 0.00000000e+00]])
X_bar_lowvar = resampler.low_variance_sampler(X_bar_old)
X_bar_multi = resampler.multinomial_sampler(X_bar_old)
print("X_bar_multi", X_bar_multi)
print("X_bar_lowvar", X_bar_lowvar)
