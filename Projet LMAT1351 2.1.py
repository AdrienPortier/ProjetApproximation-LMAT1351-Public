import numpy as np
import matplotlib.pyplot as plt

P = [[0,0],[-1,2],[4,2],[6,0],[7,1.75]]

def B(t,P):
    if len(P) == 2:
        return (1-t)*P[0][0]+t*P[1][0],(1-t)*P[0][1]+t*P[1][1]
    else:
        return (1-t)*B(t,P[:-1])+t*B(t,P[1:])

t = np.linspace(0,1,1000)
x,y = B(t,P)
plt.plot(x,y)
for p in P:
    plt.plot([p[0]],[p[1]],"black",marker=".")
for i in range(0,len(P)-1):
    x,y = B(t,[P[i],P[i+1]])
    plt.plot(x,y,"--",color="red")
plt.show()