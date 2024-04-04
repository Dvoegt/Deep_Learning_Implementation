

import gym
import numpy as np
#from tensorflow.keras import models, layers
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import gym
import matplotlib.pyplot as plt
import random

def chooseAction_N(q_values):
    return np.argmax(q_values)

def tensor_state(random_state):
    #convert random state into tensor for nn
    state_array = np.array([random_state])
    state_tensor = tf.convert_to_tensor(state_array, dtype=tf.int32)
    return state_tensor

def nn_output(random_state, model):
    state_tensor = tensor_state(random_state)
    #get Qvalues from given state
    q_values = model.predict([state_tensor], verbose = 0) 
    max_qvalue = np.amax(q_values)
    action = np.argmax(q_values)
    return(q_values, max_qvalue, action) #return the predicted action and q value with it

# Create the Taxi environment
env = gym.make('Taxi-v3', render_mode = "human")
learning_rate = 0.9
discount_factor = 0.1

# Get the observation space information
observation_space = env.observation_space
num_states = observation_space.n

# Create the neural network model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_dim = 1),
    layers.Dense(64, activation='relu'),
    layers.Dense(env.action_space.n)  # Output layer: Q-value for each action
])

# Compile the model with appropriate optimizer and loss function before training
model.compile(optimizer='adam', loss='mse')  # Example, suitable for Q-learning



random_state = env.reset()  # Get a random initial state


current_state = random_state[0]
q_values, max_qvalue, action = nn_output(current_state, model)
print(max_qvalue, action)


holder = env.step(action)                      #Conduct action, recording the output
new_state, reward, done, _ = holder[:4]
next_qvalues, next_max_qvalue, next_action = nn_output(new_state, model) #just need next q value for bellman equation

target_value = learning_rate * (reward + (discount_factor * next_max_qvalue))

print('target value is ',target_value)
print('real value was ',max_qvalue)

print('predicted q values: ', q_values)
target_qvalues = np.copy(q_values)
max_qvalue_index = np.argmax(target_qvalues)
target_qvalues[0][max_qvalue_index] = target_value


current_tensor_state = tensor_state(current_state)
#model.fit(current_tensor_state, target_qvalues, epochs=10, batch_size=32, verbose = 0)

print('success')


'''
for i in range(10000):
    
    state = env.step(1)

    new_state, reward, done, _ = state[:4]
    print('Action : ',i ,'new state is ',new_state, 'reward is ',reward, 'done is ', done)
    state = new_state
'''





'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import gym

# Convert the current state to a one-hot encoded vector
def state_to_one_hot(state, num_states):
    one_hot = np.zeros(num_states)
    one_hot[state, :] = 1
    return one_hot

env = gym.make('Taxi-v3')

num_states = env.observation_space.n
num_actions = env.action_space.n

# Neural Network model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(1, num_states )),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_actions)  # Output layer: Q-value for each action
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

'''




















num_episodes = 10
epsilon = 0.1  # Epsilon-greedy parameter
gamma = 0.99  # Discount factor
batch_size = 32


























'''
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Random action
        else:
            # Predict Q-values for current state
            current_state_vector = state_to_one_hot(state, num_states)
            q_values = model.predict(np.array([current_state_vector]))
            action = np.argmax(q_values)
        
        # Take action and observe new state and reward
        next_state, reward, done, _ = env.step(action)
        
        # Calculate target Q-value using Bellman equation
        target = reward + gamma * np.max(model.predict(np.array([[next_state]])))
        
        # Update Q-value for the chosen action
        q_values[0][action] = target
        
        # Collect experience for training
        model.fit(np.array([[state]]), q_values, verbose=0)
        
        total_reward += reward
        state = next_state
    
    # Print total rewards per episode
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
'''