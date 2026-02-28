"""
Train the Q-learning agent.
"""
import numpy as np
import matplotlib.pyplot as plt
from maze_env import MazeEnv
from q_learning import QLearningAgent

def train_agent(env_type='simple', episodes=1000):
    """Train the agent."""
    print(f"Starting training for {env_type} maze...")
    
    # Create environment and agent
    env = MazeEnv(maze_type=env_type)
    agent = QLearningAgent(
        env=env,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=1.0,
        exploration_decay=0.997,
        min_exploration=0.02
    )
    
    # Training parameters
    max_steps = 200
    
    # Track training progress
    episode_rewards = []
    episode_steps = []
    success_rate_history = []
    
    for episode in range(episodes):
        # Train one episode
        total_reward, steps = agent.train_episode(max_steps=max_steps)
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # Test every 100 episodes
        if (episode + 1) % 100 == 0 or episode == 0:
            # Test 10 episodes to compute success rate
            success_count = 0
            for _ in range(10):
                test_reward, test_steps, _ = agent.test_episode(max_steps=max_steps)
                if test_reward > 0:  # Reached the goal
                    success_count += 1
            
            success_rate = success_count / 10
            success_rate_history.append(success_rate)
            
            print(f"Episode {episode+1}/{episodes}: "
                f"Reward={total_reward:.1f}, "
                f"Steps={steps}, "
                f"Exploration={agent.exploration_rate:.3f}, "
                f"Success={success_rate:.1%}")
    
    # Save model
    model_filename = f'q_learning_model_{env_type}.pkl'
    agent.save_model(model_filename)
    
    # Visualize training results
    visualize_training(episode_rewards, episode_steps, success_rate_history, env_type)
    
    return agent, env

def visualize_training(rewards, steps, success_rates, env_type):
    """Visualize training results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Reward curve
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title(f'{env_type} maze - reward per episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    
    # 2. Step curve (moving average)
    window_size = 50
    if len(steps) >= window_size:
        steps_smooth = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
        axes[0, 1].plot(range(window_size-1, len(steps)), steps_smooth, color='orange')
    axes[0, 1].plot(steps, alpha=0.3)
    axes[0, 1].set_title(f'{env_type} maze - steps per episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    
    # 3. Success rate
    axes[1, 0].plot(success_rates)
    axes[1, 0].set_title(f'{env_type} maze - success rate')
    axes[1, 0].set_xlabel('Checkpoint (every 100 episodes)')
    axes[1, 0].set_ylabel('Success rate')
    axes[1, 0].set_ylim([0, 1])
    
    # 4. Reward distribution for the last 100 episodes
    if len(rewards) >= 100:
        last_100_rewards = rewards[-100:]
        axes[1, 1].hist(last_100_rewards, bins=20, edgecolor='black')
        axes[1, 1].set_title(f'{env_type} maze - last 100 episode reward distribution')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'training_results_{env_type}.png', dpi=300)
    plt.show()
    
    print(f"Training results saved to training_results_{env_type}.png")

def test_trained_agent(env_type='simple'):
    """Test the trained agent."""
    print(f"\nTesting {env_type} maze...")
    
    env = MazeEnv(maze_type=env_type)
    agent = QLearningAgent(env)
    
    model_filename = f'q_learning_model_{env_type}.pkl'
    if agent.load_model(model_filename):
        # Test 10 episodes
        test_results = []
        for i in range(10):
            reward, steps, path = agent.test_episode(max_steps=100, render=False)
            test_results.append((reward, steps))
            print(f"Test {i+1}: Reward={reward:.1f}, Steps={steps}")
        
        # Compute average performance
        avg_reward = np.mean([r for r, _ in test_results])
        avg_steps = np.mean([s for _, s in test_results])
        print(f"\nAverage performance: Reward={avg_reward:.1f}, Steps={avg_steps:.1f}")
        
        # Show best path
        print("\nShow best path:")
        reward, steps, path = agent.test_episode(max_steps=100, render=True)
        
        return agent, env

if __name__ == "__main__":
    # User selects training type
    print("Select training type:")
    print("1. Simple maze (5x5)")
    print("2. Medium maze (8x8)")
    print("3. Hard maze (10x10)")
    
    choice = input("Enter option (1-3): ").strip()
    
    if choice == '1':
        env_type = 'simple'
        episodes = 500
    elif choice == '2':
        env_type = 'medium'
        episodes = 1000
    elif choice == '3':
        env_type = 'hard'
        episodes = 5000
    else:
        print("Invalid choice, default to simple maze")
        env_type = 'simple'
        episodes = 500
    
    # Train agent
    agent, env = train_agent(env_type=env_type, episodes=episodes)
    
    # Test trained agent
    test_trained_agent(env_type)