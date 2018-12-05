import numpy as np
import random
import os
import pickle
from collections import deque
import cv2
import tensorflow as tf


#A Buffer of tuples (s,a,r,s') to be reused for training
class ReplayBuffer(object):
    def __init__(self, maxlength):
        #maxlength: max number of tuples in the buffer
        self.buffer = deque()
        self.number = 0
        self.maxlength = maxlength

    def append(self, experience):
        #experience: a tuple of the form (s,a,r,s')
        self.buffer.append(experience)
        self.number += 1

    def pop(self):
        #pop out the oldest tuples if self.number > self.maxlength
        while self.number > self.maxlength:
            self.buffer.popleft()
            self.number -= 1

    def sample(self, batchsize):
        #batchsize: size of the minibatch to be sampled
        return random.sample(self.buffer, batchsize)


# Similar to replay buffer, but just to save the last n states
class LastObservations(object):
    def __init__(self, length=4):
        self.length = length
        self.buffer = deque([np.zeros((84,84))]*length)

    def append(self, obs):
        obs_p = preprocess(obs)
        self.buffer.append(obs_p)
        self.buffer.popleft()

    def reset(self):
        self.__init__(length=self.length)

    def to_array(self):
        return np.stack(self.buffer, axis=-1)


# subroutine to update the target network from the principal.
# Below, a soft update is implemented
def build_target_update(from_scope, to_scope, b = 0.2):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=to_scope)
    op = []
    for v1, v2 in zip(from_vars, to_vars):
        op.append(v2.assign(v1*b + v2*(1-b)))
    return op


# transform the original images (mean, reshape, and 0-1 scale)
def preprocess(observation):
    obs = observation.mean(axis=2)
    obs = cv2.resize(obs, (84, 84))/255
    return(obs)
    #return np.reshape(obs, (84,84,1))


# go 4 steps further to fasten the game
def make_4_steps(env, action):
    r_tot = 0
    for i in range(4):
        new_obs, r, d, _ = env.step(action)
        r_tot+=r
        if d :
            break
    return new_obs, r_tot, d


def save_scores(scores, env):
    os.makedirs('scores', exist_ok=True)
    path = os.path.join('scores', 'DDQN_{}.txt'.format(env))

    with open(path, "wb") as fp:   #Pickling
        pickle.dump(scores, fp)
