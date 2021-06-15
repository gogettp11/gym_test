import gym
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import copy
import collections
import random

from tensorflow.python.keras.backend import dtype

BUFFER_SIZE = 500
max_steps = 2000
MIN_EPSILON = 0.05
GAMMA = 0.9
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state','done'])
const_one_tensor = tf.Variable(initial_value=1, dtype=tf.float32)

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
        if self.mem_pointer < self.buffer.maxlen:
            self.buffer.append(experience)
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
        self.mem_pointer+=1
    def sample(self, n_samples): #return random samples from buffer
        indices = np.random.choice(len(self.buffer), n_samples, replace=False)
        return [self.buffer[i].state for i in indices],[self.buffer[i].action for i in indices], [self.buffer[i].reward for i in indices],[self.buffer[i].next_state for i in indices], [self.buffer[i].done for i in indices]
    def samples_num(self):
        return self.mem_pointer
        

class Agent():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.target_net = myModel(self.env.observation_space.shape[0], self.env.action_space.n)
        self.training_net = myModel(self.env.observation_space.shape[0], self.env.action_space.n)
        self.experience_buffer = ExperienceBuffer(BUFFER_SIZE)
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.epsilon = 1.00 # default probability for taking random steps
        self.observation = self.env.reset()
        self.total_reward = 0

    #return tuple(reward <- int, done <- bool)
    def play_step(self):
        if self.epsilon > random.uniform(0,1):
            obs_tensor = tf.expand_dims(tf.Variable(self.observation),0) #transfrom observation to tensor
            q_actions_tensor = self.training_net(obs_tensor)
            action = int(tf.math.argmax(q_actions_tensor, axis=1))
        else:
            action = self.env.action_space.sample()

        next_observation, reward, done, info = self.env.step(action)
        self.total_reward += reward
        self.experience_buffer.append(Experience(self.observation,action,reward,next_observation,done))

        if done:
            self.observation = self.env.reset()
            temp_reward = self.total_reward
            self.total_reward = 0
            return (temp_reward,done)

        self.observation = next_observation
        return (reward, done)

    def play_episode(self):
        while True:
            reward_done = self.play_step()
            if(reward_done[1]):
                self.decrease_epsilon()
                return reward_done
        

    def decrease_epsilon(self):
        if MIN_EPSILON < self.epsilon:
            self.epsilon -= 0.015

    def training(self, iter_no):
        for i in range(iter_no):
            self.play_episode()
            self.play_episode()
            if self.experience_buffer.samples_num() >= BUFFER_SIZE:
                raw_state, raw_action, raw_reward, raw_next_state, dones = self.experience_buffer.sample(20) #hardcoded 20 because why not
                next_state_value = tf.math.top_k(input=self.target_net(tf.Variable(raw_next_state)), sorted=False)
                updated_q_values = (next_state_value.values*GAMMA)*(const_one_tensor-tf.Variable(dones, dtype=tf.float32)) + tf.Variable(raw_reward,dtype=tf.float32) #applying mask

                with tf.GradientTape() as tape:
                    q_values = tf.gather(self.training_net(tf.Variable(raw_state)), tf.expand_dims(raw_action, 1))
                    loss = self.loss_function(updated_q_values ,q_values)
                print("loss: " + loss)
                gradients = tape.gradient(self.training_net.trainable_weights, loss)
                self.optimizer.apply_gradients(self.training_net.trainable_weights, gradients)
        #self.target_net.set_weights(self.training_net.trainable_weights)
    def test_run(self):
        self.observation = self.env.reset()
        while True:
            self.env.render()
            action = self.env.action_space.sample()
            next_observation, reward, done, info = self.env.step(action)
            self.total_reward += reward

            if done:
                print(self.total_reward)
                self.total_reward = 0
                break

            self.observation = next_observation

if __name__ == '__main__':
    agent = Agent()
    while True:
        agent.training(20)
        agent.test_run()