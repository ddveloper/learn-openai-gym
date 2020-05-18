
'''
code reference from https://deeplizard.com/learn/video/HGeI30uATws
'''


import numpy as np
import gym
import random
import time
from IPython.display import clear_output

env = gym.make("FrozenLake-v0")
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
#print(q_table)

num_episodes = 10000
max_steps_per_episode = 100
learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1 # epsilon
max_exp_rate = 1
min_exp_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_ep = []

# Q-learning algorithm
for episode in range(num_episodes):
    # initialize new episode params
    state = env.reset()
    done = False
    rewards_cur_episode = 0

    for step in range(max_steps_per_episode):
        dice = random.uniform(0,1)
        if dice > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()
    
        new_state, reward, done, info = env.step(action)

        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))

        state = new_state
        rewards_cur_episode += reward

        if done == True:
            break

    exploration_rate = min_exp_rate + \
        (max_exp_rate-min_exp_rate) * np.exp(-exploration_decay_rate * episode)
    rewards_all_ep.append(rewards_cur_episode)

# calculate and print the average reward per 1000 episodes
reward_per_1000 = np.split(np.array(rewards_all_ep), num_episodes/1000)
count = 1000
print("********average reward per 1000 episodes *********\n")
for r in reward_per_1000:
    print(count, ":", str(sum(r/1000)))
    count += 1000

# print updated Q-table
print("\n\n ******Q-table*********\n")
print(q_table)


for epi in range(3):
    state = env.reset()
    done = False
    print("****EPISODE ", episode+1, "******\n\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)

        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)

        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print("******You reached the goal!!*****")
                time.sleep(3)
            else:
                print("******You fell through a hole!***")
                time.sleep(3)
            clear_output(wait=True)
            break

        state = new_state
env.close()