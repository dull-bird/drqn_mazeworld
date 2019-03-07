import numpy as np
#绘制一个迷宫
class gameEnv():
    def __init__(self):
        self.state_dim = 2
        self.action_dim = 4

        self.bonus = False
        self.get_bonus = False

        self.maze = np.zeros((6, 6))
        self.maze[0,0] = 1
        self.maze[0,1] = 1
        self.maze[1,1] = 1
        self.maze[2,1] = 1
        self.maze[3,0] = 1
        self.maze[3,1] = 1
        self.maze[3,2] = 1
        self.maze[3,3] = 1
        self.maze[3,4] = 1
        self.maze[3,5] = 1
        self.maze[4,2] = 1
        self.maze[5,2] = 1
        self.maze[5,3] = 1
        self.maze[5,4] = 1
        self.maze[5,5] = 1
        self.maze[4,5] = 1
        self.maze[2,5] = 1
        self.maze[0,3] = 1
        self.maze[0,4] = 1
        self.maze[0,5] = 1
        self.maze[1,3] = 1
        self.maze[2,3] = 1

    def reset(self):
        self.state = [0, 0]
        self.bonus = False
        self.get_bonus = False
        return self.state

#进行一次“试验”，返回新的状态
#如果走不动，就保持在原处
    def step(self, a):
        reward = 0
        #special case:
        if self.bonus == False and self.get_bonus == False and self.state[0] == 3 and self.state[1] == 4 and a == 0:
            self.bonus = True

        elif self.bonus == True and self.state[0] == 3 and self.state[1] == 4 and a == 0:
            reward += 100
            self.bonus = False
            self.get_bonus = True
        #up:
        if a == 0 and self.state[0] > 0 and self.maze[self.state[0] - 1, self.state[1]] == 1:
            self.state = (self.state[0] - 1, self.state[1])
        #down:
        elif a == 1 and self.state[0] < 5 and self.maze[self.state[0] + 1, self.state[1]] == 1:
            self.state = (self.state[0] + 1, self.state[1])
        #left
        elif a == 2 and self.state[1] > 0 and self.maze[self.state[0], self.state[1] - 1] == 1:
            self.state = (self.state[0], self.state[1] - 1)
        #right
        elif a == 3 and self.state[1] < 5 and self.maze[self.state[0], self.state[1] + 1] == 1:
            self.state = (self.state[0], self.state[1] + 1)

        if self.state[0] != 2 or self.state[1] != 5:
            done = False
            reward -= 5
        else:
            done = True
            reward += 100
        return self.state, reward, done
    

if __name__ == "__main__":
    m = gameEnv()
    s = m.reset()

    s1 = m.step(3)
    print(m.step(1))
    print(m.step(1))
    print(m.step(1))
    print(m.step(3))
    print(m.step(3))
    print(m.step(3))
    print(m.step(0))
    print(m.step(0))
    print(m.step(0))
    print(m.step(0))
    print(m.step(3))
    print(m.step(3))