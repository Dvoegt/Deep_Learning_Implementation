
#task 1_4: Alternative to Epsilon Greedy = Softmax Exploration

import gym
import numpy as np
import matplotlib.pyplot as plt

#Softmax
def softmax(q_values, temperature):
    exp_values = np.exp((q_values - np.max(q_values)) / temperature)
    softmax_probabilities = exp_values / np.sum(exp_values, axis=0)
    return softmax_probabilities

def updateAction(new_state, state, q_values, action, reward, learning_rate, discount_factor):
    q_values[state , action] += learning_rate * (reward + discount_factor * np.max(q_values[new_state,:]) - q_values[state , action])


    
#Main Method for running the model ---- With Linear Decay for Epsilon 
def runModel(flag, iterations, learning_rate, discount_factor, starting_temperature, decay_rate):
    #Initialise the Scenario, create Qtable, create standard learning rate, discount factor and epsilon
    env = gym.make('Taxi-v3')
    state_size = env.observation_space.n
    action_size = env.action_space.n
    Qtable = np.zeros((state_size, action_size))
    evaluatory_int = 0
    temp = starting_temperature + decay_rate

    rewards_collection = {}  # Track average rewards

    #Run the model n times
    for i in range(iterations):
        state = env.reset()
        total_reward = 0
        state = state[0]

        while True:
            if temp > 0.001: #Minimum Value for the Temperature
                temp = temp - decay_rate
            softmax_probabilities = softmax(Qtable[state], temp)   #Choose Action Using Softmax
            action = np.random.choice(len(Qtable[0]), p=softmax_probabilities)
            holder = env.step(action)                      #Conduct action, recording the output
            new_state, reward, done, _ = holder[:4]
        
            updateAction(new_state, state, Qtable, action, reward, learning_rate, discount_factor)  #Update Qtable
            total_reward += reward  #Update total reward for performance visualisation
            evaluatory_int += reward
            state = new_state   #
        
            if done:        #When Taxi Env end conditions are met, break the loop and end the iteration
                break

        if (i+1) % 5 == 0:
            rewards_collection.update({i+1: total_reward})

    if flag == True:
        #get x and y
        keys = list(rewards_collection.keys())
        values = list(rewards_collection.values())

        #Create Graph
        graph(keys, values)

        #holder for now
        print("Model Successful")
    else:
        return evaluatory_int/iterations ####Returns Average Reward ----> for evaluating which learning rate and discount factor is best


def graph(x, y):

    # Plotting the line graph
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Line Graph of X AND y')
    plt.grid(True)
    plt.show()

'''
learning_rates = [0.9, 0.7, 0.5, 0.3, 0.1]
discount_factors = [0.9, 0.7, 0.5, 0.3, 0.1]
lowest_score = 100000
best_LR = 0
best_DF = 0
for i in learning_rates:
    for j in discount_factors:
        print("Hypertuning Parameters, Evaluating Learning Rate: ", i, " Discount Factor: ", j)
        result =runModel(False, 1000, i, j, 1, 0.001)
        if result < lowest_score:
            lowest_score = result
            best_LR = i
            best_DF = j

#Run Model on Hypertuned Params
print("Running Model on Optimal Parameters")    #0.1, 0.1
print("Optimal Values: ", "Learning Rate: ", best_LR, "Discount Factor: ", best_DF)
'''

result =runModel(True , 00, 0.1, 0.1, 1, 0.001)

