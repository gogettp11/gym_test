import gym
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import copy
import collections

env = gym.make('CartPole-v0')
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(learning_rate=0.01)
PERCENTILE = 70
max_steps = 2000

class myModel(keras.Model):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.l1 = keras.layers.Dense(units=input_shape, activation='tanh',use_bias=False)
        self.l2 = keras.layers.Dense(units=128, activation='relu',use_bias=False)
        self.l4 = keras.layers.Dense(units=output_shape,activation='linear',use_bias=False)

    def call(self,input):
        input = self.l1(input)
        input = self.l2(input)
        return self.l4(input)

model = myModel(env.observation_space.shape[0],env.action_space.n)
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