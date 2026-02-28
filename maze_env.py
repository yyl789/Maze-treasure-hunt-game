"""
Maze environment.
Defines game state, actions, rewards, and related logic.
"""
import numpy as np

class MazeEnv:
    def __init__(self, maze_type='simple'):
        """
        Initialize maze environment.

        Args:
            maze_type: maze type ('simple', 'medium', 'hard')
        """
        self.maze_type = maze_type
        self.reset()
        
        # Action space: up, down, left, right
        self.actions = [0, 1, 2, 3]  # 0: up, 1: down, 2: left, 3: right
        self.action_names = ['Up', 'Down', 'Left', 'Right']
        
        # Define cell types
        self.EMPTY = 0      # empty
        self.WALL = 1       # wall
        self.AGENT = 2      # agent
        self.GOAL = 3       # goal
        self.TRAP = 4       # trap
        self.BONUS = 5      # bonus
        
    def reset(self):
        """Reset the environment to the initial state."""
        # Create maps for different maze types
        if self.maze_type == 'simple':
            # 5x5 simple maze
            self.maze = [
                [0, 0, 0, 0, 3],  # row 1: goal at (0,4)
                [0, 1, 0, 1, 0],  # row 2: walls
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [2, 0, 0, 0, 0]   # row 5: start at (4,0)
            ]
            self.agent_pos = (4, 0)  # agent start position
            self.goal_pos = (0, 4)   # goal position
            
        elif self.maze_type == 'medium':
            # 8x8 medium maze with traps and bonuses
            self.maze = [
                [0, 0, 0, 1, 0, 0, 0, 3],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1],
                [0, 1, 4, 1, 5, 0, 0, 0],
                [2, 0, 0, 1, 0, 1, 4, 0]
            ]
            self.agent_pos = (7, 0)
            self.goal_pos = (0, 7)
            
        elif self.maze_type == 'hard':
            # 10x10 hard maze
            self.maze = [
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 3],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
                [0, 5, 0, 0, 0, 4, 0, 0, 5, 0],
                [2, 0, 0, 1, 0, 1, 0, 1, 0, 4]
            ]
            self.agent_pos = (9, 0)
            self.goal_pos = (0, 9)
            
        self.maze = np.array(self.maze)
        self.rows, self.cols = self.maze.shape
        
        return self.get_state()
    
    def get_state(self):
        """Get current state (position coordinates)."""
        return self.agent_pos
    
    def step(self, action):
        """
        Execute action and return new state, reward, and done flag.

        Args:
            action: action index (0-3)

        Returns:
            new_state: new state
            reward: reward value
            done: whether episode ended
            info: additional info
        """
        row, col = self.agent_pos
        
        # Compute new position based on action
        if action == 0:  # up
            new_row, new_col = row - 1, col
        elif action == 1:  # down
            new_row, new_col = row + 1, col
        elif action == 2:  # left
            new_row, new_col = row, col - 1
        elif action == 3:  # right
            new_row, new_col = row, col + 1
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Check out-of-bounds or wall
        if (new_row < 0 or new_row >= self.rows or 
            new_col < 0 or new_col >= self.cols or
            self.maze[new_row, new_col] == self.WALL):
            # Out of bounds or hit a wall, stay put and penalize
            new_row, new_col = row, col
            reward = -5
            done = False
        else:
            # Move to new position
            self.agent_pos = (new_row, new_col)
            
            # Check what is at the new position
            cell_value = self.maze[new_row, new_col]
            
            if cell_value == self.GOAL:  # reached goal
                reward = 100
                done = True
            elif cell_value == self.TRAP:  # stepped on trap
                reward = -50
                done = False
            elif cell_value == self.BONUS:  # collected bonus
                reward = 8
                self.maze[new_row, new_col] = self.EMPTY  # bonus disappears
                done = False
            else:  
                reward = -1  # small step penalty to encourage shorter paths
                done = False
        
        return self.get_state(), reward, done, {}
    
    def render(self):
        """Print current maze state (console version)."""
        print("-" * (self.cols * 2 + 1))
        for i in range(self.rows):
            row_str = "|"
            for j in range(self.cols):
                if (i, j) == self.agent_pos:
                    row_str += "A|"
                elif self.maze[i, j] == self.WALL:
                    row_str += "#|"
                elif self.maze[i, j] == self.GOAL:
                    row_str += "G|"
                elif self.maze[i, j] == self.TRAP:
                    row_str += "X|"
                elif self.maze[i, j] == self.BONUS:
                    row_str += "B|"
                else:
                    row_str += " |"
            print(row_str)
        print("-" * (self.cols * 2 + 1))
    
    def get_state_index(self, state):
        """Convert state to a unique index (for Q-table)."""
        row, col = state
        return row * self.cols + col
    
    def get_total_states(self):
        """Get total number of states."""
        return self.rows * self.cols
    
    def get_legal_actions(self, state):
        """Get legal actions from the current position."""
        row, col = state
        legal_actions = []
        
        # Check four directions for legality
        if row > 0 and self.maze[row-1, col] != self.WALL:  # up
            legal_actions.append(0)
        if row < self.rows-1 and self.maze[row+1, col] != self.WALL:  # down
            legal_actions.append(1)
        if col > 0 and self.maze[row, col-1] != self.WALL:  # left
            legal_actions.append(2)
        if col < self.cols-1 and self.maze[row, col+1] != self.WALL:  # right
            legal_actions.append(3)
            
        return legal_actions