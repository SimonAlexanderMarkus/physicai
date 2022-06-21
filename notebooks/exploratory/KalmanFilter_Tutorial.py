# https://nbviewer.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/04-One-Dimensional-Kalman-Filters.ipynb

# import filterpy.stats as stats

import plotly.express as px
import numpy as np
import scipy.stats as stats

from collections import namedtuple

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: f"N(μ={s[0]:.3f}, σ²={s[1]:.3f})"


class DogSimulation(object):
    def __init__(self, x0=0, velocity=1,
                 measurement_var=0.0,
                 process_var=0.0):
        """ x0 : initial position
            velocity: (+=right, -=left)
            measurement_var: variance in measurement m^2
            process_var: variance in process (m/s)^2
        """
        self.x = x0
        self.velocity = velocity
        self.meas_std = np.sqrt(measurement_var)
        self.process_std = np.sqrt(process_var)

    def move(self, dt=1.0):
        """Compute new position of the dog in dt seconds."""
        dx = self.velocity + np.random.randn() * self.process_std
        self.x += dx * dt

    def sense_position(self):
        """ Returns measurement of new position in meters."""
        measurement = self.x + np.random.randn() * self.meas_std
        return measurement

    def move_and_sense(self):
        """ Move dog, and return measurement of new position in meters"""
        self.move()
        return self.sense_position()


def gaussian_product(g1, g2):
    mean = (g1.mean * g2.var + g2.mean * g1.var) / (g1.var + g2.var)
    var = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, var)


def plot_gaussians(gs):
    x = np.linspace(6, 14, 100)
    y = [stats.norm(g.mean, np.sqrt(g.var)).pdf(x) for g in gs]
    fig = px.line(x=x, y=y, range_y=[0, 3])
    fig.show()


def update_alt(measure, prior):
    return gaussian_product(measure, prior)


def predict_alt(pos, movement):
    return gaussian(pos.mean + movement.mean, pos.var + movement.var)


def predict(posterior, movement):
    x, P = posterior  # mean and variance of estimated state
    dx, Q = movement  # mean and variance model
    return gaussian(mean=x + dx, var=P + Q)


def update(prior, measurement):
    x, P = prior  # mean and variance of predicted state (prior)
    z, R = measurement  # mean and variance of measurement

    K = P / (P + R)  # Kalman Gain (uncertainty of prior)
    r = z - x  # Residual

    mean = x + K * r  # Posterior weighted mean (K=0 => prior, K=1 => measure)
    var = P * (1 - K)  # Posterior variance

    return gaussian(mean=mean, var=var)


if __name__ == '__main__':
    # global parameters
    process_variance = 1.  # variance in the movement
    sensor_variance = 2.  # variance in sensor

    # initial state
    x = gaussian(0., 20. ** 2)  # position
    velocity = 1.
    dt = 1.  # timestep in seconds
    process_model = gaussian(velocity * dt, process_variance)  # displacement to add to x

    dog = DogSimulation(
        x0=x.mean,
        velocity=process_model.mean,
        measurement_var=sensor_variance,
        process_var=process_model.var
    )

    zs = [dog.move_and_sense() for _ in range(10)]

    # perform Kalman Filter on measurements zs
    print('PREDICT\t\t \tUpdate')
    print('     x      var\t\t  z\t    x      var')
    for z in zs:
        prior = predict(x, process_model)
        likelihood = gaussian(mean=z, var=sensor_variance)
        x = update(likelihood, prior)

        print(f"{prior.mean:.3f}\t{prior.var:.3f}\t{z:.3f}\t{x.mean:.3f}\t{x.var:.3f}")

    print("done")
