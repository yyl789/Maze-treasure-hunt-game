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

## 第一次训练5*5
智能体直接去终点

![5*5第一次训练](https://github.com/user-attachments/assets/67d338f1-f37e-4f5d-8047-76934bec2776)

## 第一次训练8*8
1. **发现问题**: 智能体陷入局部最优
2. **原因分析**: 奖励函数设计问题
3. **解决方案**: 调整奖励值、增加步数惩罚

![第一次训练](https://github.com/user-attachments/assets/ca776690-97cf-4a03-ae6b-4db99e4324e7)

## 修改奖励函数
```bash
elif cell_value == self.BONUS:  # 吃到奖励点
    reward = 5  # 从20降低到5
    done = False
else:  # 普通空地
    reward = -1  # 从-0.1增加到-1，鼓励尽快找到目标
    done = False
```
1. 降低黄色点价值：从20降到5，减少诱惑
2. 增加步数惩罚：从-0.1到-1，鼓励尽快完成游戏
3. 使终点相对价值更高

## 第二次训练8*8
智能体直接去终点

![第二次训练](https://github.com/user-attachments/assets/545526f5-cb68-474d-bc92-59234b89ab1b)

## 第一次训练10*10
1. **发现问题**: 智能体在困难迷宫中表现不佳
2. **原因分析**: 状态空间大，训练不足，奖励函数仍需优化
3. **解决方案**: 增加训练回合，调整奖励函数，实现奖励点消失
   
![10*10第一次训练](https://github.com/user-attachments/assets/15c6d808-f5db-4f38-8fd8-aaf3391789c2)

## 修改奖励函数+修改训练参数
```bash
elif cell_value == self.BONUS:  # 吃到奖励点
    reward = 5  # 改为 3
    self.maze[new_row, new_col] = self.EMPTY  # 添加这行，让奖励点消失
    done = False
else:  # 普通空地
    reward = -1  # 改为 -2
    done = False
```
```bash
agent = QLearningAgent(
    env=env,
    learning_rate=0.1,
    discount_factor=0.85,  # 改为0.85
    exploration_rate=1.0,
    exploration_decay=0.998,  # 改为0.998
    min_exploration=0.005  # 改为0.005
)
# 修改训练回合数：
if choice == '3':
    env_type = 'hard'
    episodes = 5000  # 改为5000
```
1. 防止无限循环：从5降到3，奖励点消失机制
2. 增加步数惩罚：从-1到-2，鼓励尽快完成游戏
3. 针对不同难度调整不同参数

## 第二次训练10*10
智能体直接去终点

