import gym
import robogym
import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Flatten
from tensorflow.python.keras.optimizers import adam_v2
from robogym.envs.rearrange.blocks_train import make_env

# Hyperparameters
LR_ACTOR = 0.0001  # Learning rate for the actor network
LR_CRITIC = 0.001  # Learning rate for the critic network
GAMMA = 0.99  # Discount factor
NUM_EPISODES = 1000  # Number of episodes
MAX_STEPS = 1000  # Maximum number of steps per episode

# Define the Actor-Critic network
def build_actor_critic(state_dim, action_dim: np.ndarray):
    # Actor network
    input_state = Input(shape=state_dim)
    flatten= Flatten()(input_state)
    dense1 = Dense(64, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output_actions=[]
    for i in range(action_dim.size):
            output_actions.append( Dense(action_dim[i],activation='softmax')(dense2))

    # Critic network
    dense3 = Dense(64, activation='relu')(flatten)
    dense4 = Dense(64, activation='relu')(dense3)
    output_value = Dense(1, activation='linear')(dense4)

    # Create the actor-critic model
    actor = Model(inputs=input_state, outputs=output_actions)
    critic = Model(inputs=input_state, outputs=output_value)

    # Compile the actor and critic models
    actor.compile(optimizer=adam_v2.Adam(lr=LR_ACTOR), loss='categorical_crossentropy')
    critic.compile(optimizer=adam_v2.Adam(lr=LR_CRITIC), loss='mean_squared_error')

    actor.summary()
    critic.summary()
    return actor, critic

# A2C agent
class A2CAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor, self.critic = build_actor_critic(state_dim, action_dim)

    
    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        test= self.actor.predict(state)
        action_probs =[]
        for i in test:
            action_probs.append(i.flatten().tolist())
        actions = []
        for arr in action_probs:
            arr = np.array(arr) / np.sum(arr)
            actions.append(np.random.choice(len(arr), 1, p=arr)[0])
            

        return actions

    def train(self, states, actions, rewards, next_states, dones):
        states = np.array(states['obj_pos'])
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        next_states= np.expand_dims(next_states, axis=0)

        # Compute TD targets
        next_values = self.critic.predict(next_states)
        td_targets = rewards + GAMMA * next_values * (1 - dones)

        # Train critic
        self.critic.fit(states[0], td_targets, verbose=0)

        # Compute advantages
        values = self.critic.predict(states)
        advantages = td_targets - values

        # One-hot encode actions
        actions_onehot = np.eye(self.action_dim)[actions]

        # Train actor
        self.actor.fit(states, actions_onehot, sample_weight=advantages.flatten(), verbose=0)

# Create the RoboGym environment
env = make_env(
    constants={
        'success_reward': 5.0,
        'success_pause_range_s': [0.0, 0.5],
        'max_timesteps_per_goal_per_obj': 600,
         'vision': True,  # use False if you don't want to use vision observations
    'vision_args': {
        'image_size': 200,
        'camera_names': ['vision_cam_front'],
        'mobile_camera_names': ['vision_cam_wrist'],
    },
    'goal_args': {
        'rot_dist_type': 'full',
        'randomize_goal_rot': True,
        'p_goal_hide_robot': 1.0,
    },
    'success_threshold': {'obj_pos': 0.04, 'obj_rot': 0.2},
    },
    parameters={
        'simulation_params': {
        'num_objects': 1,
        'max_num_objects': 32,
        'object_size': 0.0254,
        'used_table_portion': 1.0,
        'goal_distance_ratio': 1.0,
        'cast_shadows': False,
        'penalty': {
            # Penalty for collisions (wrist camera, table)
            'wrist_collision': 0.0,
            'table_collision': 0.0,
            
            # Penalty for safety stops
            'safety_stop_penalty': 0.0,
        }
    }
    }
)

# Get the state and action dimensions
state_dim = env.observation_space.spaces['goal_obj_pos'].shape
action_dim = env.action_space.nvec

# Create the A2C agent
agent = A2CAgent(state_dim, action_dim)

# Training loop
for episode in range(NUM_EPISODES):
    state = env.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):
        # Get an action from the agent
        action = agent.get_action(state['obj_pos'])

        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)

        # Store the transition in memory
        agent.train(state, action, reward, next_state['obj_pos'], done)

        # Update the current state and episode reward
        state = next_state
        episode_reward += reward

        if done:
            # Print the episode reward
            print(f"Episode: {episode+1}, Reward: {episode_reward}")
            break

        # Update the agent
        if len(agent.train) >= agent.batch_size:
            agent.update()

