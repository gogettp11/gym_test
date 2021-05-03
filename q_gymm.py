import gym
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import copy
import collections
import random

env = gym.make('CartPole-v0')
loss_function = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
optimizer = tf.optimizers.Adam(learning_rate=0.001)
BUFFER_SIZE = 500
max_steps = 2000
MIN_EPSILON = 0.05
GAMMA = 0.9
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state','done'])

class myModel(keras.Model):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.l1 = keras.layers.Dense(units=input_shape, activation='tanh',use_bias=False)
        self.l2 = keras.layers.Dense(units=128, activation='relu',use_bias=False)
        self.l4 = keras.layers.Dense(units=output_shape,activation='sigmoid',use_bias=False)

    def call(self,input):
        input = self.l1(input)
        input = self.l2(input)
        return self.l4(input)

class ExperienceBuffer():
    def __init__(self,capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.mem_pointer = 0
    def append(self, experience): #insert element to the circular buffer
        buffer.insert(mem_pointer%buffer.maxlen,experience)
    def sample(self, n_samples): #return random samples from buffer
        indices = np.random.choice(len(self.buffer), n_samples, replace=False)
        return [buffer[i].state for i in indices],[buffer[i].action for i in indices],
        [buffer[i].reward for i in indices],[buffer[i].next_state for i in indices], [buffer[i].done for i in indices]
    def samples_num():
        return mem_pointer
        

class Agent():
    def __init__(self, target_net, training_net, env, optimizer, loss_function):
        self.target_net = target_net
        self.training_net = training_net
        self.env = env
        self.experience_buffer = ExperienceBuffer(BUFFER_SIZE)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epsilon = 1.00 # default probability for taking random steps
        self.observation = env.reset()
        self.total_reward = 0

    def play_step(self):
        if epsilon > random.random:
            obs_tensor = tf.Variable(observation, shape=(1, len(observation))) #transfrom observation to tensor
            q_actions_tensor = training_net(obs_tensor)
            action = tf.math.argmax(q_actions_tensor, axis=1)
        else:
            action = env.action_space.sample()

        next_observation, reward, done, info = env.step(action)
        total_reward += reward
        experience_buffer.append(Experience(observation,action,reward,next_observation,done))

        if done:
            next_observation = env.reset()
            return self.total_reward

        observation = next_observation

    def decrease_epsilon():
        if MIN_EPSILON < epsilon:
            epsilon -= 0.015

    def training():
        if self.experience_buffer.samples_num > BUFFER_SIZE:
            raw_state, raw_action, raw_reward, raw_next_state, dones = experience_buffer.sample(20) #hardcoded 20 because why not
            updated_q_values = tf.math.max(target_net(raw_next_state),1)*GAMMA + raw_reward
            #TODO dont count q_values for dones
            with tf.GradientTape() as tape:
                q_values = training_net(tf.Variable(raw_state))
                loss = loss_function(updated_q_values ,q_values)
            gradients = tape.gradient(training_net.trainable_weights, loss)
            optimizer.apply_gradients(training_net.trainable_weights, gradients)
    def test_run():
        pass

if name == '__main__':
    pass