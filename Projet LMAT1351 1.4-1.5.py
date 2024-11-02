import numpy as np
import numdifftools as nd
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
    p = 0
    for i in range(0,len(C)):
        z = 1
        for j in range(0,i):
            z *= x-float(N[j])
        z *= float(C[i])
        p += z
    return p

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
x=np.linspace(-1,1,1000)
WW = []
for nodes in N:
    W = 1
    for e in nodes:
        W *= (x-e)
    WW.append(W)
plt.plot(x,f(x),"black",label="1/1+25x²")
for i in range(len(N)):
    df = nd.Derivative(f,n=len(N))
    dy = abs(df(x))
    print(f"polynome interpolateur de 1/1+25x² pour N = {N[i]} : {pol(f,N[i])}")
    p = approximation(f,N[i],1)
    print(all((abs(f(x)-p)<=max(dy)*abs(WW[i])/factorial(len(N)))[:-1])*np.isclose(abs(f(x)-p)[-1],max(dy)*abs(WW[i][-1])/factorial(len(N))))
    plt.plot(x,p,["red","orange","green","blue"][i],linewidth=0.5,label=f"p(x), N={N[i]}")
plt.title("Approximation de 1/1+25x²")
plt.legend()
plt.show()

x=np.linspace(-1,1,1000)
plt.plot(x,f(x),"black",label="1/1+25x²")
N = [2,3,5,7,9]
for i in range(len(N)):
    C = Chebyshev(N[i],1)
    print(f"polynome interpolateur de f(x) pour N = {C} : {pol(f,C)}")
    p = approximation(f,C,1)
    plt.plot(x,p,["red","orange","green","blue","purple"][i],linewidth=0.5,label=f"p(x), N={N[i]}")
plt.title("Approximation de 1/1+25x² via Chebyshev")
plt.legend()
plt.show()