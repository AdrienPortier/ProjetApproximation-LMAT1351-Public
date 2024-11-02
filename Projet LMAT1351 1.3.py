import numpy as np
import mpmath
import matplotlib.pyplot as plt

# Précision du module mpmath
mpmath.mp.dps = 25

def getNewtonBasis(nodes):
    "entrée : une liste nodes de noeuds"
    "sortie : la liste de fonctions [1, (x-x0), (x-x0)...(x-x_{n-1})] qui calculent la valeur de la base prod (x-x_j) en x"
    if len(nodes) == 0:
        return None
    listbasis = [lambda x: mpmath.mpf(1)]
    for k in range(len(nodes) - 1):  # -1 car x_n n'est pas pris dans la base
        last = listbasis[-1]
        listbasis.append(lambda x, last=last, k=k: last(x) * (x - nodes[k]))
    return listbasis

def getNewtonCoefficients(nodes, f):
    "entrée : une liste nodes de noeuds de taille n + 1 et une fonction f"
    "sortie : la liste de coefficients [a_0, ..., a_n] de l'interpolation de f par la base de Newton de degré n"
    if len(nodes) == 0:
        return None
    listcoef = []
    for i in range(len(nodes)):
        ai = mpmath.mpf(0)
        for k in range(i + 1):
            ai += mpmath.mpf(f(nodes[k])) / prodkj(nodes, k, i)
        listcoef.append(ai)
    return listcoef

def prodkj(nodes, k, i):
    "entrée : une liste nodes de noeuds, k l'indice du noeud fixe du produit et i l'indice final du noeud libre"
    "sortie : la valeur du produit (x_k - x_j) tel que k != j, et j ∈ {0, ..., i}"
    prod = mpmath.mpf(1)
    for j in range(0, i + 1):
        if j != k:
            prod *= (nodes[k] - nodes[j])
    return prod

t = [1/4, 1/8, 1/16]

# Pour les différents choix de noeuds sur le polynôme de Newton
for tcas in t:
    x = np.linspace(-1, 1, 1000)
    plt.figure(figsize=(10, 6))

    # Pour les différents degrés des polynômes
    for n in range(1, 4 + 1):  # n = 1, 2, 3, 4
        plt.subplot(2, 2, n)
        plt.ylim(-1, 1)
        plt.plot(x, [mpmath.sin(mpmath.mpf(val)) for val in x], label="sin(x)", color="red", linewidth=0.25)

        # Approximation de Taylor en 0 de degré n
        taylor_coeffs = mpmath.taylor(mpmath.sin, 0, n)
        taylor_poly = [mpmath.polyval(taylor_coeffs[::-1], mpmath.mpf(val)) for val in x]
        plt.plot(x, taylor_poly, label="Polynôme de Taylor de degré " + str(n), color="blue", linewidth=0.5)

        # Approximation de Newton sur [-t,t] de degré n
        nodes = [mpmath.mpf(val) for val in np.linspace(-tcas, tcas, n + 1)]
        basis = getNewtonBasis(nodes)
        coef = getNewtonCoefficients(nodes, mpmath.sin)
        
        print(f"Coefficients pour t={tcas} et pour degré ={n}: {[str(c) for c in coef]}")
        
        NewtonInter = lambda x: sum(c * b(x) for c, b in zip(coef, basis))
        NewtonVals = [NewtonInter(mpmath.mpf(val)) for val in x]

        plt.plot(x, NewtonVals, label="Polynôme de Newton de degré " + str(n), color="black", linestyle="dotted")

        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
    
    plt.suptitle("Approximations de Taylor (en 0) et de Newton (sur [-t, t] avec t = " + str(tcas) + " ) de la fonction sin")
    plt.savefig(f"approx_t_{tcas}.jpg", format='jpg', dpi=300)  # dpi pour la qualité
    plt.close()


def Wnplusone(nodes):
    "entrée : une liste de noeuds nodes"
    "sortie : une fonction lambda représentant W_{n+1} -> prod (x-x_j)"
    return lambda x: mpmath.fprod([(x - nodes[j]) for j in range(len(nodes))])

# Tracés des erreurs d'approximation
colors = ['blue', 'green', 'orange', 'purple']

for n in range(1, 5):
    plt.figure(figsize=(10, 6))
    
    for idx, tcas in enumerate(t):
        x = np.linspace(-1, 1, 1000)
        
        # Création des noeuds pour l'interpolation de Newton
        nodes = [mpmath.mpf(val) for val in np.linspace(-tcas, tcas, n + 1)]
        
        # Calcul des bases et des coefficients de Newton pour l'interpolation
        basis = getNewtonBasis(nodes)
        coef = getNewtonCoefficients(nodes, mpmath.sin)
        
        # Fonction d'interpolation de Newton construite à partir des bases et coefficients
        NewtonInter = lambda x: sum(c * b(x) for c, b in zip(coef, basis))
        NewtonVals = [NewtonInter(mpmath.mpf(val)) for val in x]

        # Calcul de l'erreur d'approximation pour chaque point x
        error_vals = [abs(mpmath.sin(mpmath.mpf(val)) - NewtonInter(mpmath.mpf(val))) for val in x]
        
        # Calcul de la borne de Lagrange pour l'erreur théorique
        Sine_n_derivative = lambda x: mpmath.diff(mpmath.sin, x, n + 1)  # Dérivée n+1-ème de sin(x)
        max_derivative = max([abs(Sine_n_derivative(val)) for val in x])  # Borne sup de la dérivée
        factorial = mpmath.factorial(n + 1)  # Facteur (n+1)!
        lagrange_bound = [(max_derivative / factorial) * abs(Wnplusone(nodes)(val)) for val in x]  # Formule de la borne
        
        # Tracés de l'erreur d'approximation et de la borne de Lagrange pour chaque valeur de t
        color = colors[idx]
        plt.plot(x, error_vals, label=f"Erreur d'approximation, t={tcas}", linestyle="solid", color=color)
        plt.plot(x, lagrange_bound, label=f"Reste de Lagrange, t={tcas}", linestyle="dashed", color=color)
        
        plt.xlabel("x")
        plt.ylabel("|f(x) - p(x)| (échelle logarithmique)")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)

    plt.title(f"Comparaison des erreurs d'approximation et des bornes théoriques de Lagrange pour n={n}")
    plt.savefig(f"erreur_n_{n}.jpg", format='jpg', dpi=300)
    plt.close()

