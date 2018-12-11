import matplotlib.pyplot as plt
import numpy as np
import sys 

# colors = ["red", "yellow", "blue", "green", "black", "brown", "cyan", "magenta", "blueviolet", "coral"]


pop_8 = np.array([69, 35, 40, 40, 40, 60, 104, 52, 20, 80])
pop_16 = np.array([49, 20, 75, 65, 80, 45, 40, 67, 58, 60])
pop_32 = np.array([34, 23, 18, 58, 21, 26, 32, 26, 32, 48])

final = np.append(pop_8, pop_16)
final = np.append(final, pop_32)
final = final.reshape(3, -1)
# final = final.transpose()
# print final[1]
# sys.exit()

labels = ['Pop size 8', 'Pop size 16', 'Pop size 32']

for i in range(len(final)):
    # x = [i]*10
    y = final[i]
    x = [j for j in range(len((y)))]
    
    print "mean:"
    print np.mean(y)
    print "std:"
    print np.std(y)

    plt.plot(x, y, label=labels[i])#, color=colors[i])
    plt.scatter(x, y)#, color=colors[i])

# plt.xticks(np.arange(0, 10, 1))

plt.legend(loc='upper left')
plt.title("Differences in iteration per population size")
plt.xlabel('Curves')
plt.ylabel('Iterations')

plt.grid(True)
plt.show()