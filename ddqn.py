import numpy as np
import sys
import os
import tensorflow as tf
import gym

from utils import *


# define neural net Q_theta(s,a) as a class
class Qfunction(object):

    def __init__(self, actsize, sess, optimizer, batchsize=32):
        """
        actsize: dimension of action space
        sess: sess to execute this Qfunction
        """

        # build the prediction graph
        state = tf.placeholder(tf.float32, [None, 84, 84, 4])

        kernels = {# 8x8 conv, 4 input, 32 outputs
                   'k1': tf.Variable(tf.truncated_normal([8, 8, 4, 32])),
                   # 4x4 conv, 32 inputs, 64 outputs
                   'k2': tf.Variable(tf.truncated_normal([4, 4, 32, 64])),
                   # 3x3 conv, 64 inputs, 64 outputs
                   'k3': tf.Variable(tf.truncated_normal([3, 3, 64, 64]))
                  }

        # layer 1
        x_1 = tf.nn.conv2d(state, kernels['k1'], strides=[1, 4, 4, 1], padding='VALID')
        bn_1 = tf.layers.batch_normalization(x_1, axis=-1, momentum=0.1, epsilon=1e-5)
        l_1 = tf.nn.relu(bn_1)

        # layer 2
        x_2 = tf.nn.conv2d(l_1, kernels['k2'], strides=[1, 2, 2, 1], padding='VALID')
        bn_2 = tf.layers.batch_normalization(x_2, axis=-1, momentum=0.1, epsilon=1e-5)
        l_2 = tf.nn.relu(bn_2)

        # layer 3
        x_3 = tf.nn.conv2d(l_2, kernels['k3'], strides=[1, 1, 1, 1], padding='VALID')
        bn_3 = tf.layers.batch_normalization(x_3, axis=-1, momentum=0.1, epsilon=1e-5)
        l_3 = tf.nn.relu(bn_3)


        flat = tf.reshape(l_3, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)

        Qvalues = tf.layers.dense(inputs=dense, units=actsize, activation=tf.nn.relu)  # make sure it has size [None, actsize]

        # build the targets and actions
        # targets represent the terms E[r+gamma Q] in Bellman equations
        # actions represent a_t
        targets = tf.placeholder(tf.float32, [None])
        actions = tf.placeholder(tf.int32, [None])

        values = []  # A list of values corresponding to the respective coordinate in indices.
        for i in range(batchsize):
            values.append(Qvalues[i,actions[i]])
        Qpreds = tf.stack(values)

        loss = tf.reduce_mean(tf.square(Qpreds - targets))

        # optimization
        self.train_op = optimizer.minimize(loss)

        # some bookkeeping
        self.Qvalues = Qvalues
        self.state = state
        self.actions = actions
        self.targets = targets
        self.loss = loss
        self.sess = sess

    def compute_Qvalues(self, states):
        """
        states: numpy array as input to the neural net, states should have
        size [numsamples, obssize], where numsamples is the number of samples
        output: Q values for these states. The output should have size
        [numsamples, actsize] as numpy array
        """
        return self.sess.run(self.Qvalues, feed_dict={self.state: states})

    def train(self, states, actions, targets):
        """
        states: numpy array as input to compute loss (s)
        actions: numpy array as input to compute loss (a)
        targets: numpy array as input to compute loss (Q targets)
        """
        return self.sess.run([self.loss,self.train_op], feed_dict={self.state:states, self.actions:actions, self.targets:targets})


def eval_score(env_name, Qp):
    eval_episodes = 1
    record = []
    env = gym.make(env_name)
    last_obs = LastObservations(4)

    for ite in range(eval_episodes):
        obs = env.reset()
        last_obs.reset()
        last_obs.append(obs)
        state = last_obs.to_array()
        done = False
        rsum = 0

        while not done:
            values = Qp.compute_Qvalues(np.expand_dims(state,0))
            action = np.argmax(values.flatten())

            new_obs, r, done = make_4_steps(env, action)
            rsum += r

            last_obs.append(new_obs)
            state = last_obs.to_array()

        record.append(rsum)
    return(np.mean(record))
"""
def BayesReg(phiphiT,phiY,alpha,batchsize):
    sigma=0.001
    sigma_n=1.0
    phiphiT *= (1-alpha) #Forgetting parameter alpha suggest how much of the moment from the past can be used, we set alpha to 1 which means do not use the past moment
    phiY *= (1-alpha)
    for j in range(batchsize):
        batch = buffer.sample(1) # sample a minibatch of size one from replay buffer
        states = np.array([batch[i][0] for i in range(batchsize)])
        actions = np.array([batch[i][1] for i in range(batchsize)])
        rewards = np.array([batch[i][2] for i in range(batchsize)])
        states_prime = np.array([batch[i][3] for i in range(batchsize)])
        bat_done = transitions[0].done
        phiphiT[int(actions)] += nd.dot(dqn_(states).T,dqn_(bat_state))
        phiY[int(actions)] += (dqn_(states)[0].T*(rewards +(1.-bat_done) * opt.gamma * nd.max(nd.dot(E_W_target,target_dqn_(bat_state_next)[0].T))))
    for i in range(actsize):
        inv = np.linalg.inv((phiphiT[i]/sigma_n + 1/sigma*eye).asnumpy())
        E_W[i] = nd.array(np.dot(inv,phiY[i].asnumpy())/sigma_n, ctx = opt.ctx)
        Cov_W[i] = sigma * inv
    return phiphiT,phiY,E_W,Cov_W
"""
def main(env):
    dict_env={'pong':'Pong-v0', 'assault':'Assault-v0', 'alien':'Alien-v0', 'centipede':'Centipede-v0'}

    # parameter initializations
    envname = dict_env[env]
    lr = 0.005  # learning rate for gradient update
    rms_eps = 0.01 # RMSprop epsilon bias
    momentum=0.95 # RMSprop momentum
    decay=0.95 # RMSprop decay
    epsilon = 1.0  # constant for exploration
    epsilon_weakening = 0.995
    gamma = .99  # discount
    batchsize = 32  # batchsize for buffer sampling
    maxlength = int(1e4)  # max number of tuples held by buffer
    initialsize = 10000  # initial time steps before start updating
    episodes = int(1e4)  # number of episodes to run
    episode_max_length = 2000
    tau_target = 1000  # time steps for target update

    # initialize environment
    env = gym.make(envname)
    actsize = env.action_space.n

    # initialize tensorflow session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=decay,
                            momentum=momentum, epsilon=rms_eps, centered=True)

    # initialize networks
    with tf.variable_scope("principal"):
        Qprincipal = Qfunction(actsize, sess, optimizer)
    with tf.variable_scope("target"):
        Qtarget = Qfunction(actsize, sess, optimizer)

    # build ops
    update = build_target_update("principal", "target")  # call sess.run(update) to copy from principal to target

    # initialization of graph and buffer
    sess.run(tf.global_variables_initializer())
    buffer = ReplayBuffer(maxlength)
    last_obs = LastObservations(4)
    sess.run(update)

    counter = 0
    scores=[]
    #feeding buffer with random actions
    while buffer.number < initialsize:
        obs = env.reset()
        last_obs.reset()
        last_obs.append(obs)
        state = last_obs.to_array()
        step = 0
        d = False

        while not d and step < episode_max_length:
            action = env.action_space.sample()
            new_obs, r, d = make_4_steps(env, action)
            last_obs.append(new_obs)
            new_state = last_obs.to_array()

            if d :
                buffer.append((state, action, 0, new_state))
            else :
                buffer.append((state, action, r, new_state))
            state = new_state
            step+=1
            counter += 1

    for episode in range(episodes):
        # reset env
        obs = env.reset()
        last_obs.reset()
        last_obs.append(obs)
        state = last_obs.to_array()
        step = 0
        d = False
        reward_sum = 0

        #evaluation
        if not episode%10:
            score = eval_score(envname, Qprincipal)
            print('Episode {} -> Score : {}'.format(episode,score))
            scores.append(score)

        while not d and step < episode_max_length:
            #get batch
            batch = buffer.sample(batchsize)
            states = np.array([batch[i][0] for i in range(batchsize)])
            actions = np.array([batch[i][1] for i in range(batchsize)])
            rewards = np.array([batch[i][2] for i in range(batchsize)])
            states_prime = np.array([batch[i][3] for i in range(batchsize)])

            values = Qtarget.compute_Qvalues(states_prime)
            max_prime = np.max(values, axis=1)
            targets = rewards + gamma*max_prime
            # train Qprincipal
            Qprincipal.train(states, actions, targets)

            # select new action
            if epsilon > 0.1:
                epsilon*=epsilon_weakening
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                values = Qprincipal.compute_Qvalues(np.expand_dims(state,0))
                action = np.argmax(values.flatten())

            new_obs, r, d  = make_4_steps(env, action)
            last_obs.append(new_obs)
            new_state = last_obs.to_array()

            if d :
                buffer.append((state, action, 0, new_state))
            else :
                buffer.append((state, action, r, new_state))
            state = new_state
            reward_sum += r
            step += 1
            counter += 1

            buffer.pop()

            #Update target network
            if not counter%tau_target:
                sess.run(update)

    #save scores:
    save_scores(scores)
    plt.plot(episodes,scores)
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1])
