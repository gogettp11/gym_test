import gym
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import copy

env = gym.make('CartPole-v0')
loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(learning_rate=0.01)
PERCENTILE = 70
max_steps = 2000
open('logs.txt', 'w').close()
log_file = open("logs.txt", 'a')

#log_file.write('!--------------------------------------!\n')

class myModel(keras.Model):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.l1 = keras.layers.Dense(units=input_shape, use_bias=False)
        self.l2 = keras.layers.Dense(units=128, activation='relu',use_bias=False)
        self.l4 = keras.layers.Dense(units=output_shape, activation='softmax',use_bias=False)

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
        #log_file.write('------------------------------\n')
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
            ep_action.append(np.squeeze(action_prob))
            ep_reward += reward
            if done:
                #log_file.write("    Episode finished after {} timesteps, reward: {}\n".format(t+1, ep_reward))
                reward_history.append(ep_reward)
                obs_history.append(ep_obs)
                action_prob_history.append(ep_action)
                break
# filtering the best attempts from batch
    reward_treshold = np.percentile(reward_history, PERCENTILE)
    best_obs = []
    best_action_prob = []
    batches = 0

    log_file.write("reward_treshold: " + reward_treshold.__str__() + '\n')
    for index in range(episodes):
        if(reward_history[index] > reward_treshold):
            best_obs.append(obs_history[index])
            best_action_prob.append(action_prob_history[index])
            batches += 1
    #clear history
    reward_history.clear()
    obs_history.clear()
    action_prob_history.clear()

#training
    for i in range(batches):
        with tf.GradientTape() as tape:
            action_prob = model(tf.Variable(best_obs[i]))
            loss = loss_function(tf.Variable(best_action_prob[i]), action_prob)
            # Backpropagation
        log_file.write("loss: " + loss.__str__() + '\n')
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
# test run
    test_obs = env.reset()
    for i in range(max_steps):
        env.render()
        action_prob = softmax_l(model(tf.constant([test_obs])))
        action_prob = np.squeeze(action_prob)
        action = np.random.choice(len(action_prob), p=action_prob)
        observation, reward, done, info = env.step(action)
        if(done):
            break

    #log_file.write("weights: " + model.trainable_variables.__str__() + '\n')
    log_file.flush()

    if reward > 190:
        break
    
log_file.close()           
env.close()