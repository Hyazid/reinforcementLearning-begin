import gym
import numpy as np
import random
import time
from IPython.display import clear_output
# initialisation environement
env  = gym.make("FrozenLake-v0")
# creating Qtable
#get env propostion Q(s,a)
# states
state_space_size = env.observation_space.n
# actions
action_space_size = env.action_space.n
q_table = np.zeros((state_space_size,action_space_size))
#mettre la table a zeros chaque Q = 0
print(q_table)
#initialisation q learning algo
num_episode= 10000 # nombre d'episode que l'agent a le droit de jouer
max_step_per_episode = 100 #le nombre de cout dans un episode
learning_rate =0.1
discount_rate = 0.99
exploration_rata=1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01
reward_all_episodes=[]# sotcker tout les recompance 
#Q learning algorithme
for episode in range(num_episode):
    "initialisation new episode "
    state = env.reset()
    done = False#variable ppour savoir si l'episode est fini

    reward_current_episode =0#stocker la recompance de l'episode corant
    for step in range(max_step_per_episode):

        #  exploration/exploitation
        exploration_rate_threhold = random.uniform(0,1)
        if exploration_rate_threhold>exploration_rata:# epsilion greedy 
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()
            
        #     set new state 
        new_state,reward,done,info  =env.step(action)

        #     update q table
        # Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        " transition to the next state"
        state = new_state
        reward_current_episode+=reward
        if done== True:
            break
        "exploration rate decay  "
        # Exploration rate decay
        "diminue le exploitation rate "
        exploration_rate = min_exploration_rate +(max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
        #     get new reward
        reward_all_episodes.append(reward_current_episode)
        # Calculate and print the average reward per thousand episodes
        # reward_per_thousen_episodes = np.split(reward_all_episodes,3,0)
        # count =1000
        
        

        
        
    pass 
print(q_table)   
for episode  in range(3):
    state = env.reset()
    done  = False
    print("********--",episode+1,"--********")
    time.sleep(1)
    for step in range(max_step_per_episode):
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)

        action =np.argmax(q_table[state,:])
        new_state,reward,done,info = env.step(action)
        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print("**********BRAVO*******")
            else:
                print("*****YOU LOSE**********")
                time.sleep(3)
            clear_output(wait=True)
            break
        state = new_state

env.close()