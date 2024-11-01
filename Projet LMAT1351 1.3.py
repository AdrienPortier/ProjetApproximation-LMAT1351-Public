import numpy as np
import mpmath
import matplotlib.pyplot as plt

def getNewtonBasis(nodes):
    " entrée : une liste nodes de noeuds"
    " sortie : la liste de fonctions [1,(x-x0),(x-x0)...(x-x_{n-1})] qui calculent la valeur de la base prod (x-x_j) en x "
    if len(nodes) == 0:
        return None
    listbasis = [lambda x : 1]
    for k in range(len(nodes)-1): # -1 car x_n n'est pas pris dans la base
        last = listbasis[-1]
        listbasis.append( lambda x, last = last, k = k : last(x)*(x-nodes[k]) )
    return listbasis

def getNewtonCoefficients(nodes, f):
    " entrée : une liste nodes de noeuds de taille n + 1 et une fonction f"
    " sortie : la liste de coefficients [a_0,...,a_n] de l'interpolation de f par la base de Newton de degré n"
    if len(nodes) == 0:
        return None
    listcoef = []
    for i in range(len(nodes)):
        ai = 0
        for k in range(i+1):
            ai += f(nodes[k])/prodkj(nodes,k,i)
        listcoef.append(ai)
    return listcoef

def prodkj(nodes,k,i):
    "entrée : une liste nodes de noeuds, k l'indice du noeud fixe du produit et i l'indice final du noeud libre"
    "sortie : la valeur du produit (x_k-x_j) tel que k != j, et j €  {0,...,i}"
    prod = 1
    for j in range(0,i+1):
        if j != k:
            prod *= nodes[k]-nodes[j]
    return prod

t = [1/4,1/8,1/16]
# Pour les différents choix de noeuds sur le polynôme de Newton
for tcas in t:
    x = np.linspace(-1,1,1000)
    plt.figure(figsize=(10, 6))

    # Pour les différents degrés des polynômes

    for n in range(1, 4+1):  # n = 1,2,3,4
        plt.subplot(2,2,n)
        plt.ylim(-1,1)
        # Fonction sin de base 
        plt.plot(x, [mpmath.sin(x) for x in x], label="sin(x)", color="red", linewidth=0.25)

        # Approximation de Taylor en 0 de degré n
        taylor_coeffs = mpmath.taylor(mpmath.sin, 0, n)
        taylor_poly = mpmath.polyval(taylor_coeffs[::-1],x)
        plt.plot(x, taylor_poly, label= "Polynôme de Taylor de degré " + str(n), color = "blue", linewidth = 0.5 )

        # Approximation de Newton sur [-t,t] de degré n
        nodes = np.linspace(-tcas,tcas,n+1) # Car polynôme de degré n <-> n + 1 noeuds
        basis = getNewtonBasis(nodes)
        coef = getNewtonCoefficients(nodes, np.sin)
        print(coef)
        NewtonInter = lambda x: sum(c * b(x) for c, b in zip(coef, basis))
        NewtonVals = [NewtonInter(y) for y in x]
        
        plt.plot(x, NewtonVals, label= "Polynôme de Newton de degré " + str(n), color = "black", linestyle = "dotted" )

        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
    plt.suptitle("Approximations de Taylor (en 0) et de Newton (sur [-t,t] avec t = " + str(tcas) + " ) de la fonction sin")
    plt.show()

