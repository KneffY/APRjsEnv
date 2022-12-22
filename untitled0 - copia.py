# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 06:21:51 2022

@author: Manuel Rozas
"""

#importar llibrerias para navegación web
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.keys import Keys

from selenium.webdriver.chrome.service import Service

#importar librerias para modelo apr
import numpy as np
import json
import sys
#from config import *
#import tensorflow as tf
#from tensorflow.keras.layers import Dense
#import tensorflow_probability as tfp
import gym
import pyautogui
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
from tensorflow.keras import optimizers as opt
import random
import time
import tqdm


#ingreso a homepage DRIVE
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.maximize_window()
#inicialización AMBIENTE
driver.get('https://kneffy.github.io/APRjsEnv/')

#ENV
#espacio de observación
def observation_space():
    #distancias a los obstáculos
    dist1 = driver.find_element(By.ID, 'bb1').text
    dist2 = driver.find_element(By.ID, 'bb2').text
    dist3 = driver.find_element(By.ID, 'bb3').text
    #altura obstáculos
    alt1 = driver.find_element(By.ID, 'a1').text
    alt2 = driver.find_element(By.ID, 'a2').text
    alt3 = driver.find_element(By.ID, 'a3').text
    #distancias al objetivo
    distw = driver.find_element(By.ID, 'ww1').text
    #posicion agente
    pos_x = driver.find_element(By.ID, 'pos1').text
    pos_y = driver.find_element(By.ID, 'pos2').text

    obs = [dist1, dist2, dist3, alt1, alt2, alt3, distw, pos_x, pos_y]
    return obs

#espacio de acciones caminante
def action_space1():
    act1 = driver.find_element(By.ID, 'btn1')
    act2 = driver.find_element(By.ID, 'btn2')
    act3 = driver.find_element(By.ID, 'btn3')
    act4 = driver.find_element(By.ID, 'btn4')
    act5 = driver.find_element(By.ID, 'btn5')
    act6 = driver.find_element(By.ID, 'btn6')
    act7 = driver.find_element(By.ID, 'btn7')
    act8 = driver.find_element(By.ID, 'btn8')
    act9 = driver.find_element(By.ID, 'btn9')
    act10 = driver.find_element(By.ID, 'btn10')
    act11 = driver.find_element(By.ID, 'btn11')
    act12 = driver.find_element(By.ID, 'btn12')
    acts = [act1,act2,act3,act4,act5,act6,act7,act8,act9,act10,act11,act12]
    return acts

btn_elements1 = [btn1,btn2,btn3,btn4,btn5,btn6,btn7,btn8,btn9,btn10,btn11,btn12]
actions1 = action_space1()

#espacio de acciones volador
def action_space2():
    act1 = driver.find_element(By.ID, 'btn13').text
    act2 = driver.find_element(By.ID, 'btn14').text
    act3 = driver.find_element(By.ID, 'btn15').text
    act4 = driver.find_element(By.ID, 'btn16').text
    act5 = driver.find_element(By.ID, 'btn17').text
    act6 = driver.find_element(By.ID, 'btn18').text
    act7 = driver.find_element(By.ID, 'btn19').text
    act8 = driver.find_element(By.ID, 'btn20').text
    acts = [act1,act2,act3,act4,act5,act6,act7,act8]
    return acts

def reward():
    rwd = driver.find_element(By.ID, 'rwrd').text
    return [rwd]

def done():
    done = driver.find_element(By.ID, 'episode').text
    return [done]

def reset():
    wait = WebDriverWait(driver, 10)
    reset = driver.find_element(By.ID, 'reset')
    wait.until(ec.visibility_of(reset))
    reset.click()

BATCH_SIZE = 64
MIN_SIZE_BUFFER = 100 # Minimum size of the buffer to start learning, until then random actions
BUFFER_CAPACITY = 1000000

ACTOR_HIDDEN_0 = 512
ACTOR_HIDDEN_1 = 256

CRITIC_HIDDEN_0 = 512
CRITIC_HIDDEN_1 = 256

LOG_STD_MIN = -20 # exp(-10) = 4.540e-05
LOG_STD_MAX = 2 # exp(2) = 7.389
EPSILON = 1e-6

GAMMA = 0.99
ACTOR_LR = 0.0005
CRITIC_LR = 0.0005
#ACTOR_LR = 0.01
#CRITIC_LR = 0.005

TAU = 0.05 # For soft update the target network

REWARD_SCALE = 2 # Scale factor for rewards

THETA=0.15
DT=1e-1

MAX_EPISODES = 100000
SAVE_FREQUENCY = 200

actions_dim = len(acts)

#%%
######################################################################
#BUFFER

class ReplayBuffer1():
    def __init__(self, buffer_capacity=BUFFER_CAPACITY, batch_size=BATCH_SIZE, min_size_buffer=MIN_SIZE_BUFFER):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.min_size_buffer = min_size_buffer
        self.buffer_counter = 0
        self.n_episodes = 0
        
        self.states = np.zeros((self.buffer_capacity, observation_space().shape))
        self.actions = np.zeros((self.buffer_capacity, action_space1().shape))
        self.rewards = np.zeros((self.buffer_capacity))
        self.next_states = np.zeros((self.buffer_capacity, observation_space().shape))
        self.dones = np.zeros((self.buffer_capacity), dtype=bool)
        

        
    def __len__(self):
        return self.buffer_counter

    def add_record(self, state, action, reward, next_state, done):
        # Set index to zero if counter = buffer_capacity and start again (1 % 100 = 1 and 101 % 100 = 1) so we substitute the older entries
        index = self.buffer_counter % self.buffer_capacity

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done
        
        # Update the counter when record something
        self.buffer_counter += 1
    
    def check_buffer_size(self):
        return self.buffer_counter >= self.batch_size and self.buffer_counter >= self.min_size_buffer
    
    def update_n_episodes(self):
        self.episodes += 1
    
    def get_minibatch(self):
        # If the counter is less than the capacity we don't want to take zeros records, 
        # if the cunter is higher we don't access the record using the counter 
        # because older records are deleted to make space for new one
        buffer_range = min(self.buffer_counter, self.buffer_capacity)
        
        batch_index = np.random.choice(buffer_range, self.batch_size, replace=False)

        # Convert to tensors
        state = self.states[batch_index]
        action = self.actions[batch_index]
        reward = self.rewards[batch_index]
        next_state = self.next_states[batch_index]
        done = self.dones[batch_index]
        
        return state, action, reward, next_state, done

#%%
######################################################################
#networks

class Critic(tf.keras.Model):
    def __init__(self, name, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1):
        super(Critic, self).__init__()
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.net_name = name

        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.q_value = Dense(1, activation=None)

    def call(self, state, action):
        state_action_value = self.dense_0(tf.concat([state, action], axis=1))
        state_action_value = self.dense_1(state_action_value)

        q_value = self.q_value(state_action_value)

        return q_value

class CriticValue(tf.keras.Model):
    def __init__(self, name, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1):
        super(CriticValue, self).__init__()
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.net_name = name
        
        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.value = Dense(1, activation=None)

    def call(self, state):
        value = self.dense_0(state)
        value = self.dense_1(value)

        value = self.value(value)

        return value

class Actor(tf.keras.Model):
    def __init__(self, name, upper_bound, actions_dim, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1, epsilon=EPSILON, log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX):
        super(Actor, self).__init__()
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.actions_dim = actions_dim
        self.net_name = name
        self.upper_bound = upper_bound
        self.epsilon = epsilon
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.mean = Dense(self.actions_dim, activation=None)
        self.log_std = Dense(self.actions_dim, activation=None)

    def call(self, state):
        policy = self.dense_0(state)
        policy = self.dense_1(policy)

        mean = self.mean(policy)
        log_std = self.log_std(policy)

        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_action_log_probs(self, state, reparameterization_trick=True):
        mean, log_std = self.call(state)
        std = tf.exp(log_std)
        normal_distr = tfp.distributions.Normal(mean, std)
        # Reparameterization trick
        z = tf.random.normal(shape=mean.shape, mean=0., stddev=1.)

        if reparameterization_trick:
            actions = mean + std * z
        else:
            actions = normal_distr.sample()

        action = tf.math.tanh(actions) * self.upper_bound
        log_probs = normal_distr.log_prob(actions) - tf.math.log(1 - tf.math.pow(action,2) + self.epsilon)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs

#%%
######################################################################
#Agent

class Agent:
    def __init__(self, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, tau=TAU, reward_scale=REWARD_SCALE):
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer()
        self.actions_dim = action_space().shape[0]
        self.upper_bound = action_space().high[0]
        self.lower_bound = action_space().low[0]
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor = Actor(actions_dim=self.actions_dim, name='actor', upper_bound=action_space().high)
        self.critic_0 = Critic(name='critic_0')
        self.critic_1 = Critic(name='critic_1')
        self.critic_value = CriticValue(name='value')
        self.critic_target_value = CriticValue(name='target_value')

        self.actor.compile(optimizer=opt.Adam(learning_rate=self.actor_lr))
        self.critic_0.compile(optimizer=opt.Adam(learning_rate=self.critic_lr))
        self.critic_1.compile(optimizer=opt.Adam(learning_rate=self.critic_lr))
        self.critic_value.compile(optimizer=opt.Adam(learning_rate=self.critic_lr))
        self.critic_target_value.compile(optimizer=opt.Adam(learning_rate=self.critic_lr))

        self.reward_scale = reward_scale

        self.critic_target_value.set_weights(self.critic_value.weights)
        
    def update_target_networks(self, tau):
        critic_value_weights = self.critic_value.weights
        critic_target_value_weights = self.critic_target_value.weights
        for index in range(len(critic_value_weights)):
            critic_target_value_weights[index] = tau * critic_value_weights[index] + (1 - tau) * critic_target_value_weights[index]

        self.critic_target_value.set_weights(critic_target_value_weights)
        
    def add_to_replay_buffer(self, state, action, reward, new_state, done):
        self.replay_buffer.add_record(state, action, reward, new_state, done)
        
    def get_action(self, observation):
        state = tf.convert_to_tensor([observation])
        actions, _ = self.actor.get_action_log_probs(state, reparameterization_trick=False)

        return actions[0]

    def learn(self):
        if self.replay_buffer.check_buffer_size() == False:
            return

        state, action, reward, new_state, done = self.replay_buffer.get_minibatch()

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            value = tf.squeeze(self.critic_value(states), 1)
            target_value = tf.squeeze(self.critic_target_value(new_states), 1)

            policy_actions, log_probs = self.actor.get_action_log_probs(states, reparameterization_trick=False)
            log_probs = tf.squeeze(log_probs,1)
            q_value_0 = self.critic_0(states, policy_actions)
            q_value_1 = self.critic_1(states, policy_actions)
            q_value = tf.squeeze(tf.math.minimum(q_value_0, q_value_1), 1)

            value_target = q_value - log_probs
            value_critic_loss = 0.5 * tf.keras.losses.MSE(value, value_target)

        value_critic_gradient = tape.gradient(value_critic_loss, self.critic_value.trainable_variables)
        self.critic_value.optimizer.apply_gradients(zip(value_critic_gradient, self.critic_value.trainable_variables))


        with tf.GradientTape() as tape:
            new_policy_actions, log_probs = self.actor.get_action_log_probs(states, reparameterization_trick=True)
            log_probs = tf.squeeze(log_probs, 1)
            new_q_value_0 = self.critic_0(states, new_policy_actions)
            new_q_value_1 = self.critic_1(states, new_policy_actions)
            new_q_value = tf.squeeze(tf.math.minimum(new_q_value_0, new_q_value_1), 1)
        
            actor_loss = log_probs - new_q_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))
        

        with tf.GradientTape(persistent=True) as tape:
            q_pred = self.reward_scale * reward + self.gamma * target_value * (1-done)
            old_q_value_0 = tf.squeeze(self.critic_0(state, action), 1)
            old_q_value_1 = tf.squeeze(self.critic_1(state, action), 1)
            critic_0_loss = 0.5 * tf.keras.losses.MSE(old_q_value_0, q_pred)
            critic_1_loss = 0.5 * tf.keras.losses.MSE(old_q_value_1, q_pred)
    
        critic_0_network_gradient = tape.gradient(critic_0_loss, self.critic_0.trainable_variables)
        critic_1_network_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)

        self.critic_0.optimizer.apply_gradients(zip(critic_0_network_gradient, self.critic_0.trainable_variables))
        self.critic_1.optimizer.apply_gradients(zip(critic_1_network_gradient, self.critic_1.trainable_variables))

        self.update_target_networks(tau=self.tau)
        
        self.replay_buffer.update_n_episodes()

#%%
######################################################################
#main

rwd = []
evaluation = True

for episode in range(MAX_EPISODES+1):
    reset()
    ep_reward = 0
    done = 0
    actual_episode = episode
    states = observation()
    while done != 1:
        action = Agent.get_action(states)
        cont=0
        #envía una acción al ambiente
        for a in actions1:
            if action==a:
                btn = btn_elements1[cont]
                wait = WebDriverWait(driver, 10)
                act_btn = driver.find_element(By.ID, btn)
                wait.until(ec.visibility_of(act_btn))
                act_btn.click()
            cont=cont+1
        new_states = observation()
        reward = reward()
        done = done()
        ep_reward += reward
        Agent.add_to_replay_buffer(states,action,reward,new_states,done)
        Agent.learn()
        states = new_states
    rwd.append(ep_reward)
    ReplayBuffer1.update_n_episodes()
    
