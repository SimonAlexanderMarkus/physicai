
from filterpy.common import Q_discrete_white_noise
from filterpy import kalman as kf

import numpy as np
import math
from numpy.random import randn

# model of one-dim movement
def compute_dog_data(z_var, process_var, count=1, dt=1.):
    x, vel = 0., 1.
    z_std = math.sqrt(z_var)
    p_std = math.sqrt(process_var)
    xs, zs = [], []
    for _ in range(count):
        v = vel + (randn() * p_std)
        x += v*dt
        xs.append(x)
        zs.append(x + randn() * z_std)
    return np.array(xs), np.array(zs)

def pos_vel_filter(x, P, R, Q=0, dt=1.):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """
    f = kf.KalmanFilter(dim_x=2, dim_z=1)
    f.x = np.array([x[0], x[1]])
    f.F = np.array([[1., dt],
                    [0., 1.]])
    f.H = np.array([[1., 0]])
    f.R *= R
    if np.isscalar(P):
        f.P *= P
    else:
        f.P = P[:]         # [:] makes deep copy
    if np.isscalar(Q):
        f.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
    else:
        f.Q = Q[:]
    return f


if __name__ == '__main__':
    dt = 1.
    z_var = 4.  # Sensor variance
    p_var = 2.  # Process variance

    x0 = np.array([.0, 1.])  # initial state position and velocity (as mean of normal distr.)
    P0 = np.diag([500., 49.])  # initial state covariance, (velocity <= 21m/s: 3σ=21 -> σ²=49)

    F = np.array([[1., dt],
                  [0., 1]])  # state transition function
    Q = Q_discrete_white_noise(dim=2, dt=dt, var=p_var)  # process covariance (white noise for movement)
    H = np.array([[1., 0.]])  # measurement function, H [x, v] = x
    R = np.array([[z_var]])  # measurement variance

    f = pos_vel_filter(x=x0, P=P0, R=R, Q=Q, dt=dt)
    xs_act, zs = compute_dog_data(z_var=z_var, process_var=p_var, count=20, dt=dt)
    print(f)

    print("\n\n", "-"*5+">", "done.")
