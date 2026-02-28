"""
Main entry point - Maze Treasure Q-learning project.
"""
import os

def main():
    """Main function."""
    print("=" * 50)
    print("Maze Treasure Q-learning Project")
    print("=" * 50)
    
    while True:
        print("\nSelect an option:")
        print("1. Train a new agent")
        print("2. Play with a trained agent")
        print("3. View Q-table")
        print("4. Exit")
        
        choice = input("Enter option (1-4): ").strip()
        
        if choice == '1':
            # Run training script
            from train import train_agent, test_trained_agent
            
            print("\nChoose maze difficulty:")
            print("1. Simple (5x5)")
            print("2. Medium (8x8)")
            print("3. Hard (10x10)")
            
            difficulty = input("Select difficulty (1-3): ").strip()
            
            if difficulty == '1':
                env_type = 'simple'
                episodes = 500
            elif difficulty == '2':
                env_type = 'medium'
                episodes = 1000
            elif difficulty == '3':
                env_type = 'hard'
                episodes = 2000
            else:
                print("Invalid choice, using simple difficulty")
                env_type = 'simple'
                episodes = 500
            
            print(f"\nStarting training for {env_type} maze, {episodes} episodes...")
            agent, env = train_agent(env_type, episodes)
            
            # Ask whether to test
            test_choice = input("\nTraining completed. Test now? (y/n): ").strip().lower()
            if test_choice == 'y':
                test_trained_agent(env_type)
                
        elif choice == '2':
            # Use trained agent
            from train import test_trained_agent
            
            print("\nChoose maze difficulty:")
            print("1. Simple (5x5)")
            print("2. Medium (8x8)")
            print("3. Hard (10x10)")
            
            difficulty = input("Select difficulty (1-3): ").strip()
            
            if difficulty == '1':
                env_type = 'simple'
            elif difficulty == '2':
                env_type = 'medium'
            elif difficulty == '3':
                env_type = 'hard'
            else:
                print("Invalid choice, using simple difficulty")
                env_type = 'simple'
            
            # Check if model file exists
            model_file = f'q_learning_model_{env_type}.pkl'
            if not os.path.exists(model_file):
                print(f"Error: model file not found: {model_file}")
                print("Please train an agent or ensure the model file exists")
            else:
                test_trained_agent(env_type)
                
        elif choice == '3':
            # View Q-table
            print("\nView Q-table:")
            print("1. Simple maze")
            print("2. Medium maze")
            print("3. Hard maze")
            
            difficulty = input("Select maze (1-3): ").strip()
            
            if difficulty == '1':
                env_type = 'simple'
            elif difficulty == '2':
                env_type = 'medium'
            elif difficulty == '3':
                env_type = 'hard'
            else:
                print("Invalid choice, using simple maze")
                env_type = 'simple'
            
            # Load model and show Q-table
            from maze_env import MazeEnv
            from q_learning import QLearningAgent
            
            env = MazeEnv(maze_type=env_type)
            agent = QLearningAgent(env)
            
            model_file = f'q_learning_model_{env_type}.pkl'
            if agent.load_model(model_file):
                print(f"\n{env_type} maze Q-table summary:")
                print(f"Q-table shape: {agent.q_table.shape}")
                print(f"Q-value range: [{agent.q_table.min():.2f}, {agent.q_table.max():.2f}]")
                print(f"Non-zero Q-value ratio: {(agent.q_table != 0).sum() / agent.q_table.size:.1%}")
                
                # Show first few state Q-values
                print("\nQ-values for first 5 states:")
                for i in range(min(5, agent.n_states)):
                    row = i // env.cols
                    col = i % env.cols
                    print(f"State({row},{col}): {agent.q_table[i]}")
                
                # Get and show policy
                policy = agent.get_policy()
                print("\nSample policy:")
                count = 0
                for state, action in policy.items():
                    if action is not None and count < 10:
                        action_name = env.action_names[action]
                        print(f"State{state} -> {action_name}")
                        count += 1
                        
        elif choice == '4':
            print("Thanks for using the app. Goodbye!")
            break
            
        else:
            print("Invalid option, please try again")

if __name__ == "__main__":
    main()