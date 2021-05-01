import gym
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import copy
import collections
import random

env = gym.make('CartPole-v0')
loss_function = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
optimizer = tf.optimizers.Adam(learning_rate=0.01)
BUFFER_SIZE = 500
max_steps = 2000
MIN_EPSILON = 0.05
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state'])

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
        buffer.insert(mem_pointer%buffer.maxlen,namedtuple)
    def sample(self, n_samples): #return random samples from buffer
        indices = np.random.choice(len(self.buffer), n_samples, replace=False)
        return [buffer[i].state for i in indices],[buffer[i].action for i in indices],[buffer[i].reward for i in indices],[buffer[i].next_state for i in indices]
        

class Agent():
    def __init__(self, target_net, training_net, env, optimizer, loss_function):
        self.target_net = target_net
        self.training_net = training_net
        self.env = env
        self.experience_buffer = ExperienceBuffer()
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epsilon = 1.00 # default probability for taking random steps
        self.observation = env.reset()

    def play_step(self):
        if epsilon > random.random:
            obs_tensor = tf.Variable(observation, shape=(1, len(observation))) #transfrom observation to tensor
            q_actions_tensor = training_net(obs_tensor)
            action = tf.math.argmax(q_actions_tensor, axis=1)
        else:
            action = env.action_space.sample()

        next_observation, reward, done, info = env.step(action)
        experience_buffer.append(Experience(observation,action,reward,next_observation))

        if done:
            next_observation = env.reset()

        observation = next_observation
        return done

    def decrease_epsilon():
        if MIN_EPSILON < epsilon:
            epsilon -= 0.015

    def training():
        raw_state, raw_action, raw_reward, raw_next_state = experience_buffer.sample(20) #hardcoded 20 because why not
        with tf.GradientTape() as tape:
            training_net

        


if name == '__main__':
    pass
model_training = myModel(env.observation_space.shape[0],env.action_space.n)
model_target = myModel(env.observation_space.shape[0],env.action_space.n)

episodes = 16
softmax_l = keras.layers.Softmax()


obs_history = []
reward_history = []
action_prob_history = []

while True:
# getting data for training
    for i_episode in range(episodes):
        observation = env.reset()
        ep_obs = []
        ep_action = []
        ep_reward = 0
        for t in range(max_steps):

            ep_obs.append(observation)

            obs = tf.constant(observation, shape=(1, len(observation)))
            action_prob = model(obs)
            action_prob_softmax = softmax_l(action_prob)
            action = np.random.choice(len(np.squeeze(action_prob)),p=np.squeeze(action_prob_softmax))

            observation, reward, done, info = env.step(action)
            ep_action.append(action)
            ep_reward += reward
            if done:
                reward_history.append(ep_reward)
                obs_history.append(ep_obs)
                action_prob_history.append(ep_action)
                break
# filtering the best attempts from batch
    reward_treshold = np.percentile(reward_history, PERCENTILE)
    best_obs = []
    best_action_prob = []
    batches = 0

    print("mean reward: " + np.mean(reward_history).__str__())
    print("max reward: " + np.max(reward_history).__str__())
    print("min reward: " + np.min(reward_history).__str__())

    for index in range(episodes):
        if(reward_history[index] >= reward_treshold):
            best_obs.append(obs_history[index])
            best_action_prob.append(action_prob_history[index])
            batches += 1
    #clear history
    print("training examples: " + batches.__str__())
    reward_history.clear()
    obs_history.clear()
    action_prob_history.clear()

#training
    for i in range(batches):
        with tf.GradientTape() as tape:
            action_prob = model(tf.constant(best_obs[i]))
            loss = loss_function(best_action_prob[i], action_prob)
        # Backpropagation
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
# test run
    test_obs = env.reset()
    test_run_reward=0
    for i in range(max_steps):
        env.render()
        action_prob = softmax_l(model(tf.constant(test_obs, shape=(1, len(observation)))))
        action_prob = np.squeeze(action_prob)
        action = np.random.choice(len(action_prob), p=action_prob)
        observation, reward, done, info = env.step(action)
        test_obs = observation
        test_run_reward+=reward
        if(done):
            print("test reward: " + test_run_reward.__str__() + '\n')
            break

    if reward > 190:
        break
      
env.close()