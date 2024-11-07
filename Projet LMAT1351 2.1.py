import numpy as np
import matplotlib.pyplot as plt

P = np.array([[-1,0], [-0.5,15], [0, -5], [2, 0], [1, 3],[4,5],[-2,8]])

def B(t,P):
    if len(P) == 2:
        return (1-t)*P[0][0]+t*P[1][0],(1-t)*P[0][1]+t*P[1][1]
    else:
        return (1-t)*B(t,P[:-1])+t*B(t,P[1:])

t = np.linspace(0,1,1000)
x,y = B(t,P)
plt.plot(x,y, label = "Courbe de Bézier", color = "red")

for p in P:
    if (p == P[0]).all():
        plt.scatter(p[0], p[1], color='mediumseagreen', s=100, marker='*', label="Point de départ", zorder=5)
    elif (p == P[-1]).all():
        plt.scatter(p[0], p[1], color='royalblue', s=100, marker='*', label="Point d'arrivée", zorder=5)
    else:
        plt.scatter(p[0], p[1], color='black', s=75, marker='o', zorder=5)

for i in range(0,len(P)-1):
    x,y = B(t,[P[i],P[i+1]])
    plt.plot(x,y,"--",color="black")
plt.legend()
plt.show()
plt.savefig("BezierExemple.svg")