import matplotlib.pyplot as plt
import numpy as np
import time
loss_data=[]
r=[]
loss_data1=[]
r1=[]
for i in range(100):
    loss_data.append(i)
    r.append(i*i)
    loss_data1.append(i)
    r1.append(i*i*2)
    x=np.array(loss_data)
    y=np.array(r)
    x1=np.array(loss_data1)
    y1=np.array(r1)
    #plt.plot(x,y,x1,y1)scatter
    if(i%20==0):
        plt.scatter(x,y,x1,y1)
        plt.draw()
        plt.pause(0.1)
        plt.savefig("./result_pic/examples1.jpg")
        print(i)
    #time.sleep(1)
    

'''
import matplotlib.pyplot as plt

fig = plt.figure()
x = [1, 2, 3, 4, 5, 6, 7]
y = [1, 3, 4, 2, 5, 8, 6]

# below are all percentage
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax1 = fig.add_axes([left, bottom, width, height])  # main axes
ax1.plot(x, y, 'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('title')

ax2 = fig.add_axes([0.2, 0.6, 0.25, 0.25])  # inside axes
ax2.plot(y, x, 'b')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('title inside 1')


# different method to add axes
####################################
plt.axes([0.6, 0.2, 0.25, 0.25])
plt.plot(y[::-1], x, 'g')
plt.xlabel('x')
plt.ylabel('y')
plt.title('title inside 2')

plt.show()
'''
'''
import matplotlib.pyplot as plt
import numpy as np

n = 1024    # data size
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
T = np.arctan2(Y, X)    # for color later on

plt.scatter(X, Y, s=75, c=T, alpha=.5)

plt.xlim(-1.5, 1.5)
plt.xticks(())  # ignore xticks
plt.ylim(-1.5, 1.5)
plt.yticks(())  # ignore yticks

plt.show()
'''