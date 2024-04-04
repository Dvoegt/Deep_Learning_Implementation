#Import Required Libraries
import gym
import random
import numpy as np
import matplotlib.pyplot as plt


#Code Partially adapted from: https://www.gocoder.one/blog/rl-tutorial-with-openai-gym/
#Epsilon Greedy Functions (ChooseAction, UpdateAction)
def chooseAction(env, state, epsilon, qtable):         #Given the current state, epsilon value and Qtable, if random int <epsilon choose random action else choose best action
    if random.uniform(0,1) < epsilon:
        return  env.action_space.sample()
    else:
        return np.argmax(qtable[state,:])
                                                #Update Qtable based on Q-learning update formula as shown in Week 9 Lectures
def updateAction(new_state, state, q_values, action, reward, learning_rate, discount_factor):
    q_values[state , action] += learning_rate * (reward + discount_factor * np.max(q_values[new_state,:]) - q_values[state , action])


    
#Main Method for running the model ---- With Linear Decay for Epsilon 
def runModel(flag, iterations, learning_rate, discount_factor, starting_epsilon, decay_rate):
    #Initialise the Scenario, create Qtable, create standard learning rate, discount factor and epsilon
    env = gym.make('Taxi-v3')
    state_size = env.observation_space.n
    action_size = env.action_space.n
    Qtable = np.zeros((state_size, action_size))
    epsilon = starting_epsilon + decay_rate
    epsilon_minimum = 0.1
    evaluatory_int = 0
    #Run the model n times

    rewards_collection = {}  # Track average rewards
    for i in range(iterations):
        if epsilon > epsilon_minimum:
            epsilon = epsilon - decay_rate
        state = env.reset()
        total_reward = 0
        state = state[0]

        while True:
            action = chooseAction(env, state, epsilon, Qtable) # Choose next action to take
            holder = env.step(action)                      #Conduct action, recording the output
            new_state, reward, done, _ = holder[:4]
        
            updateAction(new_state, state, Qtable, action, reward, learning_rate, discount_factor)  #Update Qtable
            total_reward += reward  #Update total reward for performance visualisation
            evaluatory_int += reward
            state = new_state   #
            

            if done:        #When Taxi Env end conditions are met, break the loop and end the iteration
                break

            #if i == 5 or i==500 or i == 9999:
            #    print('ep ',i,'- action is ', action)

        if (i+1) % 5 == 0:
            rewards_collection.update({i+1: total_reward})
        
        print(total_reward)
    
    if flag == True:
        #get x and y
        keys = list(rewards_collection.keys())
        values = list(rewards_collection.values())

        #Create Graph
        graph(keys, values)

        #holder for now
        print("Model Successful")
        return Qtable
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


#Parameter Hypertuning
'''
learning_rates = [0.9, 0.7, 0.5, 0.3, 0.1]
discount_factors = [0.9, 0.7, 0.5, 0.3, 0.1]
lowest_score = 100000
best_LR = 0
best_DF = 0
for i in learning_rates:
    for j in discount_factors:
        print("Hypertuning Parameters, Evaluating Learning Rate: ", i, " Discount Factor: ", j)
        result =runModel(False, 1000, learning_rate=i, discount_factor=j, starting_epsilon=1, decay_rate= 0.01)
        if result < lowest_score:
            lowest_score = result
            best_LR = i
            best_DF = j



#Run Model on Hypertuned Params
print("Running Model on Optimal Parameters")
print("Optimal Values: ", "Learning Rate: ", best_LR, "Discount Factor: ", best_DF)  #0.1, 0.1
runModel(True, 10000, learning_rate=best_LR, discount_factor=best_DF, starting_epsilon=1, decay_rate= 0.01)
'''

print("Running Model on Optimal Parameters")
print("Optimal Values - Learning Rate: 0.1   Discount Factor: 0.1")  #0.1, 0.1
tqtable = runModel(True, 10000, learning_rate=0.1, discount_factor=0.1, starting_epsilon=1, decay_rate= 0.01)





#Initialise the Scenario, create Qtable, create standard learning rate, discount factor and epsilon
env = gym.make('Taxi-v3', render_mode = "human")

state_size = env.observation_space.n
action_size = env.action_space.n


#np.copyto(tstate_size, taction_size)
Qtable = np.zeros((state_size, action_size))
np.copyto(tqtable,Qtable)
epsilon = 0.1
epsilon_minimum = 0.1
evaluatory_int = 0
#Run the model n times

rewards_collection = {}  # Track average rewards
for i in range(10):
    state = env.reset()
    total_reward = 0
    state = state[0]

    while True:
        action = chooseAction(env, state, epsilon, Qtable) # Choose next action to take
        holder = env.step(action)                      #Conduct action, recording the output
        new_state, reward, done, _ = holder[:4]
    
        updateAction(new_state, state, Qtable, action, reward, 0.1, 0.1)  #Update Qtable
        total_reward += reward  #Update total reward for performance visualisation
        evaluatory_int += reward
        state = new_state   #
        

        if done:        #When Taxi Env end conditions are met, break the loop and end the iteration
            break