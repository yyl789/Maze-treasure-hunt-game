"""
Q-learning algorithm implementation.
"""
import numpy as np
import random
import pickle
import os

class QLearningAgent:
    def __init__(self, env=None, n_states=None, n_actions=None,
                 learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        """
        Initialize the Q-learning agent.

        Args:
            env: environment instance
            learning_rate: learning rate (alpha)
            discount_factor: discount factor (gamma)
            exploration_rate: exploration rate (epsilon)
            exploration_decay: exploration decay
            min_exploration: minimum exploration rate
        """
        if env is None:
            if n_states is None or n_actions is None:
                raise TypeError("env or (n_states, n_actions) is required")
            self.env = None
            self.n_states = int(n_states)
            self.n_actions = int(n_actions)
        else:
            self.env = env
            self.n_states = env.get_total_states()
            self.n_actions = len(env.actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        
        # Initialize Q-table
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        # Track training history
        self.rewards_history = []
        self.steps_history = []
        
    def choose_action(self, state, training=True):
        """
        Choose an action based on the current state.

        Args:
            state: current state
            training: whether in training mode (controls exploration)

        Returns:
            action: selected action
        """
        self._require_env()
        state_idx = self.env.get_state_index(state)
        
        if training and random.random() < self.exploration_rate:
            # Explore: choose a random legal action
            legal_actions = self.env.get_legal_actions(state)
            if legal_actions:
                return random.choice(legal_actions)
            else:
                return random.choice(self.env.actions)
        else:
            # Exploit: choose the action with the highest Q value
            legal_actions = self.env.get_legal_actions(state)
            if not legal_actions:
                return random.choice(self.env.actions)
            
            # Only consider legal actions
            q_values = [self.q_table[state_idx, a] for a in legal_actions]
            max_q = max(q_values)
            
            # If multiple actions share the max Q value, pick one at random
            best_actions = [a for i, a in enumerate(legal_actions) if q_values[i] == max_q]
            return random.choice(best_actions)
    
    def learn(self, state, action, reward, next_state, done):
        """
        Update the Q-table.

        Args:
            state: current state
            action: executed action
            reward: received reward
            next_state: next state
            done: whether the episode ended
        """
        self._require_env()
        state_idx = self.env.get_state_index(state)
        next_state_idx = self.env.get_state_index(next_state)
        
        # Q-learning update rule
        if done:
            target = reward
        else:
            # Max Q value of the next state
            next_max_q = np.max(self.q_table[next_state_idx])
            target = reward + self.discount_factor * next_max_q
        
        # Update current Q value
        current_q = self.q_table[state_idx, action]
        self.q_table[state_idx, action] = current_q + self.learning_rate * (target - current_q)
    
    def update_exploration_rate(self):
        """Decay the exploration rate."""
        self.exploration_rate = max(self.min_exploration, 
                                    self.exploration_rate * self.exploration_decay)
    
    def train_episode(self, max_steps=100):
        """
        Train a single episode.

        Args:
            max_steps: max steps

        Returns:
            total_reward: total reward
            steps: steps taken
        """
        self._require_env()
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Choose action
            action = self.choose_action(state, training=True)
            
            # Execute action
            next_state, reward, done, _ = self.env.step(action)
            
            # Learn
            self.learn(state, action, reward, next_state, done)
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Update exploration rate
        self.update_exploration_rate()
        
        # Record history
        self.rewards_history.append(total_reward)
        self.steps_history.append(steps)
        
        return total_reward, steps
    
    def test_episode(self, max_steps=100, render=False):
        """
        Test one episode (no exploration, exploit only).

        Args:
            max_steps: max steps
            render: whether to render

        Returns:
            total_reward: total reward
            steps: steps taken
            path: path
        """
        self._require_env()
        state = self.env.reset()
        total_reward = 0
        steps = 0
        path = [state]
        
        for step in range(max_steps):
            if render:
                self.env.render()
                print(f"Step {step+1}, State: {state}, Reward: {total_reward}")
                input("Press Enter to continue...")
            
            # Choose action (no exploration)
            action = self.choose_action(state, training=False)
            
            # Execute action
            next_state, reward, done, _ = self.env.step(action)
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward
            steps += 1
            path.append(state)
            
            if done:
                if render:
                    print(f"🎉 Reached the goal! Total reward: {total_reward}, Steps: {steps}")
                break
            elif step == max_steps - 1:
                if render:
                    print(f"❌ Failed to reach the goal within {max_steps} steps")
        
        return total_reward, steps, path
    
    def save_model(self, filename='q_learning_model.pkl'):
        """Save the model."""
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'exploration_rate': self.exploration_rate,
                'rewards_history': self.rewards_history,
                'steps_history': self.steps_history
            }, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='q_learning_model.pkl'):
        """Load the model."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.exploration_rate = data['exploration_rate']
                self.rewards_history = data['rewards_history']
                self.steps_history = data['steps_history']
            print(f"Model loaded from {filename}")
            return True
        else:
            print(f"File {filename} does not exist")
            return False
    
    def get_policy(self):
        """Get the policy (best action for each state)."""
        self._require_env()
        policy = {}
        for i in range(self.n_states):
            row = i // self.env.cols
            col = i % self.env.cols
            state = (row, col)
            
            # Only consider legal actions
            legal_actions = self.env.get_legal_actions(state)
            if legal_actions:
                # Choose the max Q value among legal actions
                q_values = [self.q_table[i, a] for a in legal_actions]
                best_action_idx = np.argmax(q_values)
                best_action = legal_actions[best_action_idx]
                policy[state] = best_action
            else:
                policy[state] = None
        
        return policy

    def _require_env(self):
        if self.env is None:
            raise ValueError("env is required for this operation")