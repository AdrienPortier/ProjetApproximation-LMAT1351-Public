import numpy as np
import matplotlib.pyplot as plt

P = [[0,0],[1,1],[2,0],[3,1],[4,0]]
Q = [[0.5,1],[1.5,7],[2.5,3],[3.5,1]]

def B(t,P):
    if len(P) == 2:
        return (1-t)*P[0][0]+t*P[1][0],(1-t)*P[0][1]+t*P[1][1]
    else:
        return (1-t)*B(t,P[:-1])+t*B(t,P[1:])

t = np.linspace(0,1,1000)
for i in range(0,len(P)-1):
    x,y = B(t,[P[i],Q[i],P[i+1]])
    plt.plot(x,y,"blue")
plt.show()