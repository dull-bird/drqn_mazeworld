import matplotlib.pyplot as plt

filename = 'records\\DQN_level3.txt'
X,Y = [],[]

with open(filename, 'r') as f:#1
    lines = f.readlines()#2
    for line in lines:#3
        value = [s for s in line.split()]#4
        X.append(float(value[1]))#5
        Y.append(float(value[2]))

plt.axhline(150,0,len(X), color='r', linestyle='--')
plt.plot(X, Y)
plt.yticks([-3000, -2000, -1000, 0, 150])
print(max(Y))
plt.xlabel("episode")
plt.ylabel("total rewards")
plt.show()