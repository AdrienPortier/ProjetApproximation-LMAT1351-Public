import numpy as np
import matplotlib.pyplot as plt

P1 = [[0,0],[0,3.5],[2.5,3],[0,-3.5]]
P2 = [[0,0],[0,3.5],[-2.5,3],[0,-3.5]]


def B(t,P):
    if len(P) == 2:
        return (1-t)*P[0][0]+t*P[1][0],(1-t)*P[0][1]+t*P[1][1]
    else:
        return (1-t)*B(t,P[:-1])+t*B(t,P[1:])

t = np.linspace(0,1,1000)
x,y = B(t,P1)
plt.plot(x,y,"red")
x,y = B(t,P2)
plt.plot(x,y,"red")
plt.title("Heart")
plt.xlim(-2,2)
plt.ylim(-4,3)
plt.show()
