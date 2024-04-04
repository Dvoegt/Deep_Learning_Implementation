#Import Required Libraries
import gym
import random
import matplotlib.pyplot as plt
import numpy as np

#Epsilon Greedy Functions (ChooseAction, UpdateAction)
def chooseAction(env, state, epsilon, qtable):         #Given the current state, epsilon value and Qtable, if random int <epsilon choose random action else choose best action
    if random.uniform(0,1) < epsilon:
        return  env.action_space.sample()
    else:
        return np.argmax(qtable[state,:])
                                                #Update Qtable based on Q-learning update formula as shown in Week 9 Lectures
def updateAction(new_state, state, q_values, action, reward, learning_rate, discount_factor):
    q_values[state , action] += learning_rate * (reward + discount_factor * np.max(q_values[new_state,:]) - q_values[state , action])
        

    #Initialise the Scenario
    env = gym.make('Taxi-v3')
    state_size = env.observation_space.n
    action_size = env.action_space.n
    Qtable = np.zeros((state_size, action_size))
    learning_rate = 0.9
    discount_factor = 0.8
    epsilon = 0.1


    rewards_collection = {}  # Track average rewards
    for i in range(100):
        

        state = env.reset()
        total_reward = 0
        state = state[0]

        while True:
            # Choose an action using epsilon-greedy strategy
            action = chooseAction(state, epsilon, Qtable)

            # Take a step in the environment
            holder = env.step(action)
            new_state, reward, done, _ = holder[:4]
            
            # Update Q-values using epsilon-greedy update
            updateAction(new_state, state, Qtable, action, reward, learning_rate, discount_factor)
            total_reward += reward
            state = new_state
            

            if done:
                break
                
    
        if (i+1) % 5 == 0:
            rewards_collection.update({i+1: total_reward})
        

        print(f"Iteration: {i + 1},  Total Reward: {total_reward}")
        
        
            

    env.close()  # Close the environment





# Extracting keys and values
keys = list(rewards_collection.keys())
values = list(rewards_collection.values())

# Plotting the line graph
plt.figure(figsize=(8, 6))
plt.plot(keys, values, marker='o', linestyle='-', color='blue')
plt.xlabel('Keys')
plt.ylabel('Values')
plt.title('Line Graph of Dictionary Keys vs. Values')
plt.grid(True)
plt.show()
