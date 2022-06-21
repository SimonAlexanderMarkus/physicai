import numpy as np

import plotly as py
import plotly.express as px
import plotly.graph_objects as go

import filterpy as py
from filterpy.discrete_bayes import normalize
from filterpy.discrete_bayes import predict
from filterpy.discrete_bayes import update
import pandas as pd


class Train(object):

    def __init__(self, track_len, kernel=[1.], sensor_acc=0.9):
        self.track_len = track_len
        self.kernel = kernel
        self.sensor_acc = sensor_acc
        self.pos = 0

    def move(self, dist=1):
        # move the train forward
        self.pos += dist

        # insert random movement error to kernel
        r = np.random.random()
        s = 0
        offset = -(len(self.kernel) - 1) / 2
        for k in self.kernel:
            s += k
            if r <= s:
                break
            offset += 1
        self.pos = int((self.pos + offset) % self.track_len)
        return self.pos

    def sense(self):
        pos = self.pos
        if np.random.random() > self.sensor_acc:
            if np.random.random() > 0.5:
                pos += 1
            else:
                pos -= 1
        return pos % self.track_len


def likelihood(track, measurement, acc):
    scale = acc / (1. - acc)
    result = np.ones(len(track))
    result[track == measurement] *= scale
    return result


def train_filter(iterations, kernel, sensor_acc, move_dist, do_print=True):
    track = np.array(range(10))
    robot = Train(len(track), kernel, sensor_acc)

    # x0-state
    prior = np.array([1.] + [0.0] * 9)
    posterior = prior[:]
    normalize(prior)

    xs = [0.]
    zs = [0.]
    priors = [prior]
    posteriors = [posterior]
    indices = [0]

    for i in range(iterations):
        # give move-order to robot
        robot.move(move_dist)

        # perform prediction
        prior = predict(posterior, move_dist, kernel)

        # perform measurement update
        z = robot.sense()
        lh = likelihood(track, z, sensor_acc)
        posterior = update(lh, prior)
        index = np.argmax(posterior)

        if do_print:
            print(f'time {i+1}: pos {robot.pos}, sensed {z}, at position {track[robot.pos]}')
            conf = posterior[index] * 100
            print(f'        estimated position is {index} with confidence {conf:.4f}%:')

        xs.append(robot.pos)
        zs.append(z)
        priors.append(prior)
        posteriors.append(posterior)
        indices.append(index)

    return np.array(xs), np.array(zs), np.array(priors), np.array(posteriors), np.array(indices)


def plot_filter(df):
    fig = px.bar(
        df,
        x='pos',
        y=['x', 'z', 'prior', 'posterior'],
        animation_frame='iter',
        title='Train Filter',
        barmode='group',
        color_discrete_map={
            'x': 'green',
            'z': 'blue',
            'prior': 'red',
            'posterior': 'orange',
        },
    )
    fig.show()

def plot_accuracy(xs, indices):
    fig = px.line(x=range(len(xs)), y=xs-indices)
    fig.show()

def create_DataFrame(xs, zs, priors, posteriors, indices):
    iters = priors.shape[0]
    poss = priors.shape[1]
    data = [[iter, pos, int(xs[iter] == pos), int(zs[iter] == pos), priors[iter, pos], posteriors[iter, pos], int(indices[iter] == pos)] for iter
            in range(iters) for pos in range(poss)]
    df = pd.DataFrame(data, columns=['iter', 'pos', 'x', 'z', 'prior', 'posterior', 'index'])
    return df

if __name__ == '__main__':
    xs, zs, priors, posteriors, indices = train_filter(iterations=300, kernel=[.05, .1, .6, .2, .05], sensor_acc=.65, move_dist=1, do_print=True)
    df = create_DataFrame(xs, zs, priors, posteriors, indices)
    plot_filter(df)
    plot_accuracy(xs, indices)
    print("done")
