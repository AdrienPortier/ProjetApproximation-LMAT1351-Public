import numpy as np
from matplotlib import pyplot as plt
from math import factorial

def f(x):
    return 1/(1+25*x**2)

def quotient(f,N,C,depth=0):
    if depth < len(C):
        return (quotient(f,N,C,depth+1)-C[len(C)-1-depth])/(N[len(C)]-N[len(C)-1-depth])
    else:
        return f(N[len(C)])
    
def fillC(f,N,C):
    for i in range(0,len(N)):
        C.append(quotient(f,N,C))
    return C

def approximation(f,N,L):
    C = fillC(f,N,[])
    x = np.linspace(-L,L,1000)
    y = f(x)
    p = 0
    for i in range(0,len(C)):
        z = 1
        for j in range(0,i):
            z *= x-float(N[j])
        z *= float(C[i])
        p += z
    plt.plot(x,y,"blue")
    plt.plot(x,p,"red")
    plt.legend(["1/(1+25x²)","p(x)"])
    #plt.ylim(min(y)-1,max(y)+1)
    plt.show()

def pol(f,N):
    C = fillC(f,N,[])
    p = f"{np.around(C[0],2)}+{np.around(C[1],2)}x"
    for i in range(2,len(C)):
        p += f"+{np.around(C[i],2)}x^{i}"
    return p

def Chebyshev(n,s):
    N = []
    for i in range(0,n+1):
        angle = np.pi*(2*i+1)/(2*n+2)
        N.append(s*np.cos((2*i+1)*np.pi/(2*n+2)))
    return N

N = [[-1,1],[-1,0,1],[-1,-0.5,0,0.5,1],[-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]]
for n in N:
    print(f"polynome interpolateur de f(x) pour N = {n} : {pol(f,n)}")
    approximation(f,n,1)

for n in [2,3,5,7,9]:
    C = Chebyshev(n,1)
    print(f"polynome interpolateur de f(x) pour N = {C} : {pol(f,C)}")
    approximation(f,C,1)