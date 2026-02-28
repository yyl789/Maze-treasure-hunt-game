# Maze Treasure Q-learning Project

## Overview
This project is a maze treasure hunt game based on the Q-learning algorithm. The agent learns to navigate the maze, avoid traps, and reach the goal through reinforcement learning.

## Features
1. **Multiple difficulty levels**: Simple (5x5), Medium (8x8), Hard (10x10)
2. **Complete Q-learning implementation**: balances exploration and exploitation
3. **Visualization**: Pygame UI
4. **Training and testing**: supports training agents and evaluating performance
5. **Data visualization**: generates training curves and result analysis

## Install Dependencies
```bash
pip install -r requirements.txt
```

## First training 5*5
The agent goes directly to the finish line.

![First training 5*5](https://github.com/user-attachments/assets/67d338f1-f37e-4f5d-8047-76934bec2776)

## First training 8*8
1. **Problem**: Agent stuck in local optimum.
2. **Cause**: Poor reward function design.
3. **Solution**: Adjust reward values, add step penalty.

![First training 8*8](https://github.com/user-attachments/assets/ca776690-97cf-4a03-ae6b-4db99e4324e7)

## Modify the reward function
```bash
elif cell_value == self.BONUS: 
    reward = 5  # Reduce from 20 to 5
    done = False
else:  
    reward = -1  # Increase from -0.1 to -1
    done = False
```
1. Reduce yellow point value: from 20 to 5, reduce temptation.  
2. Increase step penalty: from -0.1 to -1, encourage faster completion.  
3. Make the goal relatively more valuable.

## Second training 8*8
The agent goes directly to the finish line

![Second training 8*8](https://github.com/user-attachments/assets/545526f5-cb68-474d-bc92-59234b89ab1b)

## First training 10*10
1. **Problem**: Agent performs poorly in difficult mazes.  
2. **Cause**: Large state space, insufficient training, reward function needs improvement.  
3. **Solution**: Increase training episodes, adjust reward function, make reward points disappear.
   
![First training 10*10](https://github.com/user-attachments/assets/15c6d808-f5db-4f38-8fd8-aaf3391789c2)

## Modify reward function and training parameters
```bash
elif cell_value == self.BONUS: 
    reward = 3  # Reduce from 5 to 3
    self.maze[new_row, new_col] = self.EMPTY  # Reward points disappear
    done = False
else: 
    reward = -2  # Reduce from -1 to -2
    done = False
```
```bash
agent = QLearningAgent(
    env=env,
    learning_rate=0.1,
    discount_factor=0.85,  # Change to 0.85
    exploration_rate=1.0,
    exploration_decay=0.998,  # Change to 0.998
    min_exploration=0.005  # Change to 0.005
)
# Modify the number of training rounds
if choice == '3':
    env_type = 'hard'
    episodes = 5000  # Change to 5000
```
1. Prevent infinite loops: from 5 to 3, reward point disappearance mechanism.  
2. Increase step penalty: from -1 to -2, encourage faster completion.  
3. Adjust parameters for different difficulty levels.

## Second training 10*10
1. **Initial back-and-forth movement**: inefficient.  
2. **Complete disregard for yellow reward points**: too "utilitarian," focusing only on the goal.

![Second training 10*10](https://github.com/user-attachments/assets/0bef063e-86a1-4922-99c0-b1a8d5752130)

## Modify reward function and training parameters
```bash
elif cell_value == self.BONUS: 
    reward = 8  # Increase from 3 to 8
    self.maze[new_row, new_col] = self.EMPTY  # Bonus points disappear
    done = False
else: 
    reward = -1  # Reduce from -2 to -1
    done = False
```
```bash
agent = QLearningAgent(
    env=env,
    learning_rate=0.1,
    discount_factor=0.9,
    exploration_rate=1.0,
    exploration_decay=0.997,  # Adjust appropriately
    min_exploration=0.02  # Slightly increase the minimum exploration rate
)
```
1. Collects reward points along the way: reward value of 8 is attractive enough, but step penalty of -1 prevents excessive detours  
2. More intelligent exploration

## Third training 10*10
The agent ate the reward point and went to the finish line.

![Third training 10*10](https://github.com/user-attachments/assets/0b2b44f2-2ad1-4d86-87c9-11a11b017ea2)




