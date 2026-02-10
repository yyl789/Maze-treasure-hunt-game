"""
Pygame UI
Visualize maze and agent movement
"""
import pygame
import sys
from maze_env import MazeEnv
from q_learning import QLearningAgent

# Initialize Pygame
pygame.init()

# Color definitions
COLORS = {
    'EMPTY': (255, 255, 255),      # white - empty
    'WALL': (100, 100, 100),       # gray - wall
    'AGENT': (0, 150, 255),        # blue - agent
    'GOAL': (0, 200, 0),           # green - goal
    'TRAP': (255, 50, 50),         # red - trap
    'BONUS': (255, 200, 0),        # yellow - bonus
    'GRID': (200, 200, 200),       # light gray - grid
    'TEXT': (0, 0, 0),             # black - text
    'BACKGROUND': (240, 240, 240), # light gray - background
    'PATH': (200, 230, 255),       # light blue - path
}

class MazeUI:
    def __init__(self, env_type='simple', cell_size=60):
        """
        Initialize UI
        
        Args:
            env_type: maze type
            cell_size: cell size (px)
        """
        self.env = MazeEnv(maze_type=env_type)
        self.env.reset()
        
        # Compute window size
        self.cell_size = cell_size
        self.width = self.env.cols * cell_size + 300  # space for info panel
        self.height = self.env.rows * cell_size + 100
        
        # Create window
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Maze Treasure - {env_type} mode")
        
        # Fonts
        self.font = pygame.font.SysFont(None, 28)
        self.title_font = pygame.font.SysFont(None, 36)
        
        # load agent if available
        self.agent = QLearningAgent(self.env)
        self.model_file = f'q_learning_model_{env_type}.pkl'
        self.agent_loaded = self.agent.load_model(self.model_file)
        
        # Game state
        self.running = True
        self.auto_play = False
        self.step_delay = 500  # ms
        self.last_step_time = 0
        
        # stats
        self.total_reward = 0
        self.total_steps = 0
        self.path = []
        
    def draw_maze(self):
        """Draw maze"""
        # background
        self.screen.fill(COLORS['BACKGROUND'])
        
        # cells
        for i in range(self.env.rows):
            for j in range(self.env.cols):
                # cell rect
                rect = pygame.Rect(
                    j * self.cell_size + 10,
                    i * self.cell_size + 10,
                    self.cell_size - 2,
                    self.cell_size - 2
                )
                
                # choose color by cell type
                cell_type = self.env.maze[i, j]
                if cell_type == self.env.WALL:
                    color = COLORS['WALL']
                elif cell_type == self.env.GOAL:
                    color = COLORS['GOAL']
                elif cell_type == self.env.TRAP:
                    color = COLORS['TRAP']
                elif cell_type == self.env.BONUS:
                    color = COLORS['BONUS']
                else:
                    color = COLORS['EMPTY']
                
                # draw cell
                pygame.draw.rect(self.screen, color, rect)
                
                # grid border
                pygame.draw.rect(self.screen, COLORS['GRID'], rect, 1)
                
                # path marker
                if (i, j) in self.path:
                    path_rect = pygame.Rect(
                        j * self.cell_size + 15,
                        i * self.cell_size + 15,
                        self.cell_size - 12,
                        self.cell_size - 12
                    )
                    pygame.draw.rect(self.screen, COLORS['PATH'], path_rect)
        
        # agent
        agent_row, agent_col = self.env.agent_pos
        agent_rect = pygame.Rect(
            agent_col * self.cell_size + 15,
            agent_row * self.cell_size + 15,
            self.cell_size - 12,
            self.cell_size - 12
        )
        pygame.draw.rect(self.screen, COLORS['AGENT'], agent_rect)
        
        # simple eyes
        eye_size = 5
        pygame.draw.circle(self.screen, (255, 255, 255), 
                          (agent_rect.left + 15, agent_rect.top + 15), 
                          eye_size)
        pygame.draw.circle(self.screen, (255, 255, 255), 
                          (agent_rect.right - 15, agent_rect.top + 15), 
                          eye_size)
    
    def draw_info_panel(self):
        """Draw info panel"""
        # panel origin
        info_x = self.env.cols * self.cell_size + 20
        
        # title
        title = self.title_font.render("Maze Treasure", True, COLORS['TEXT'])
        self.screen.blit(title, (info_x, 20))
        
        # current state
        state_text = self.font.render(f"Position: {self.env.agent_pos}", True, COLORS['TEXT'])
        self.screen.blit(state_text, (info_x, 70))
        
        # reward and steps
        reward_text = self.font.render(f"Reward: {self.total_reward:.1f}", True, COLORS['TEXT'])
        self.screen.blit(reward_text, (info_x, 100))
        
        steps_text = self.font.render(f"Steps: {self.total_steps}", True, COLORS['TEXT'])
        self.screen.blit(steps_text, (info_x, 130))
        
        # agent status
        if self.agent_loaded:
            agent_status = self.font.render("Agent: loaded", True, (0, 150, 0))
        else:
            agent_status = self.font.render("Agent: not trained", True, (150, 0, 0))
        self.screen.blit(agent_status, (info_x, 160))
        
        # play mode
        status_text = self.font.render(
            f"Auto-play: {'on' if self.auto_play else 'off'}",
            True, COLORS['TEXT']
        )
        self.screen.blit(status_text, (info_x, 190))
        
        # instructions
        instructions = [
            "Controls:",
            "Arrow keys: move agent",
            "Space: toggle auto/manual",
            "R: reset",
            "N: next step (auto)",
            "1/2/3: switch maze",
            "Esc: quit"
        ]
        
        for i, text in enumerate(instructions):
            instruction = self.font.render(text, True, COLORS['TEXT'])
            self.screen.blit(instruction, (info_x, 230 + i * 30))
        
        # legend
        legend_y = 230 + len(instructions) * 30 + 20
        
        # legend items
        legend_items = [
            ("Agent", COLORS['AGENT']),
            ("Goal", COLORS['GOAL']),
            ("Trap", COLORS['TRAP']),
            ("Bonus", COLORS['BONUS']),
            ("Wall", COLORS['WALL'])
        ]
        
        for i, (name, color) in enumerate(legend_items):
            # color box
            pygame.draw.rect(self.screen, color, 
                            (info_x, legend_y + i * 30, 20, 20))
            pygame.draw.rect(self.screen, COLORS['GRID'], 
                            (info_x, legend_y + i * 30, 20, 20), 1)
            
            # text
            legend_text = self.font.render(name, True, COLORS['TEXT'])
            self.screen.blit(legend_text, (info_x + 30, legend_y + i * 30))
    
    def draw(self):
        """Draw UI"""
        self.draw_maze()
        self.draw_info_panel()
        pygame.display.flip()
    
    def handle_events(self):
        """Handle events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    
                elif event.key == pygame.K_r:
                    # reset
                    self.env.reset()
                    self.total_reward = 0
                    self.total_steps = 0
                    self.path = [self.env.agent_pos]
                    
                elif event.key == pygame.K_SPACE:
                    # toggle auto/manual
                    self.auto_play = not self.auto_play
                    
                elif event.key == pygame.K_n and self.auto_play:
                    # next step (auto)
                    self.step_agent()
                    
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3]:
                    # switch maze
                    if event.key == pygame.K_1:
                        self.change_maze('simple')
                    elif event.key == pygame.K_2:
                        self.change_maze('medium')
                    elif event.key == pygame.K_3:
                        self.change_maze('hard')
                
                elif not self.auto_play:
                    # manual control
                    action = None
                    if event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_DOWN:
                        action = 1
                    elif event.key == pygame.K_LEFT:
                        action = 2
                    elif event.key == pygame.K_RIGHT:
                        action = 3
                    
                    if action is not None:
                        self.step_agent(action)
    
    def step_agent(self, action=None):
        """
        Step once
        
        Args:
            action: specified action, None = agent chooses
        """
        if action is None and self.agent_loaded:
            # agent chooses
            state = self.env.get_state()
            action = self.agent.choose_action(state, training=False)
        
        if action is not None:
            # apply action
            next_state, reward, done, _ = self.env.step(action)
            
            # update stats
            self.total_reward += reward
            self.total_steps += 1
            self.path.append(next_state)
            
            # check done
            if done:
                print("Reached the goal!")
                print(f"Total reward: {self.total_reward}, steps: {self.total_steps}")
                self.auto_play = False  # stop auto-play
    
    def change_maze(self, maze_type):
        """Switch maze"""
        self.env = MazeEnv(maze_type=maze_type)
        self.env.reset()
        
        # reload agent
        self.agent = QLearningAgent(self.env)
        self.model_file = f'q_learning_model_{maze_type}.pkl'
        self.agent_loaded = self.agent.load_model(self.model_file)
        
        # reset state
        self.total_reward = 0
        self.total_steps = 0
        self.path = [self.env.agent_pos]
        
        # update title
        pygame.display.set_caption(f"Maze Treasure - {maze_type} mode")
        
        print(f"Switched to {maze_type} maze")
    
    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        
        while self.running:
            # events
            self.handle_events()
            
            # auto-play
            if self.auto_play:
                current_time = pygame.time.get_ticks()
                if current_time - self.last_step_time > self.step_delay:
                    self.step_agent()
                    self.last_step_time = current_time
            
            # draw
            self.draw()
            
            # frame rate
            clock.tick(30)
        
        pygame.quit()
        sys.exit()

def main():
    """Entry point"""
    print("Maze Treasure - Pygame UI")
    print("Choose maze difficulty:")
    print("1. Simple (5x5)")
    print("2. Medium (8x8)")
    print("3. Hard (10x10)")

    choice = input("Enter option (1-3): ").strip()
    
    if choice == '1':
        env_type = 'simple'
    elif choice == '2':
        env_type = 'medium'
    elif choice == '3':
        env_type = 'hard'
    else:
        print("Invalid choice, default to simple")
        env_type = 'simple'
    
    # Run the game
    game = MazeUI(env_type=env_type)
    game.run()

if __name__ == "__main__":
    main()