import matplotlib.pyplot as plt

filename = 'C:\\Users\\DAI Zhiwen\\Desktop\\records\\DQN_level1.txt'
X,Y = [],[]

with open(filename, 'r') as f:#1
    lines = f.readlines()#2
    for line in lines:#3
        value = [s for s in line.split()]#4
        X.append(float(value[1]))#5
        Y.append(float(value[2]))

print(X)
print(Y)

plt.plot(X, Y)
plt.show()