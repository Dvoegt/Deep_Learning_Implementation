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

num_episodes = 10
epsilon = 0.1  # Epsilon-greedy parameter
gamma = 0.99  # Discount factor
batch_size = 32

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