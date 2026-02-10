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

## 第一次训练
1. **发现问题**: 智能体陷入局部最优
2. **原因分析**: 奖励函数设计问题
3. **解决方案**: 调整奖励值、增加步数惩罚

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

## 第二次训练
智能体直接去终点



