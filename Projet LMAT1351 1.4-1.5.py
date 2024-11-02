import numpy as np
import mpmath
import matplotlib.pyplot as plt

mpmath.mp.dps = 25

def f(x):
    return 1 / (1 + 25 * x**2)

def lagrange_interpolation(f, nodes, x):
    p = 0
    for i, node in enumerate(nodes):
        term = f(node)
        for j, other_node in enumerate(nodes):
            if j != i:
                term *= (x - other_node) / (node - other_node)
        p += term
    return p

def Wnplusone(nodes):
    return lambda x: mpmath.fprod([(x - nodes[j]) for j in range(len(nodes))])

def uniform_nodes(n, a=-1, b=1):
    return np.linspace(a, b, n + 1)

def chebyshev_nodes(n):
    return [np.cos((2 * i + 1) * np.pi / (2 * n + 2)) for i in range(n + 1)]

n_values = [5, 10, 20, 40]
x = np.linspace(-1, 1, 1000)

# Couleurs pour les graphiques
colors = ['blue', 'green', 'orange', 'purple']

# Graphique pour les nœuds uniformément distribués
plt.figure(figsize=(12, 6))
plt.plot(x, f(x), "black", label="f(x) = 1 / (1 + 25x²)")
plt.yscale("log")

for i, ncas in enumerate(n_values):
    nodes = uniform_nodes(ncas)
    p = lagrange_interpolation(f, nodes, x)
    plt.plot(x, p, label=f"N={ncas}", color=colors[i])  # Utilisation des couleurs définies
    plt.scatter(nodes, f(np.array(nodes)), color='black', s=25, zorder=5)

plt.title("Approximation de f avec des nœuds uniformément répartis par l'interpolation de Lagrange")
plt.xlabel("x")
plt.ylabel("f(x) (échelle logarithmique)")
plt.legend()
plt.grid()
plt.savefig("approximation_uniform_nodes.svg")  # Enregistrer le graphique en SVG
plt.close()  # Fermer la figure

# Graphique pour les nœuds de Chebyshev
plt.figure(figsize=(12, 6))
plt.plot(x, f(x), "black", label="f(x) = 1 / (1 + 25x²)")
plt.yscale("log")

for i, ncas in enumerate(n_values):
    nodes = chebyshev_nodes(ncas)
    p = lagrange_interpolation(f, nodes, x)
    plt.plot(x, p, label=f"N={ncas}", color=colors[i])  # Utilisation des couleurs définies
    plt.scatter(nodes, f(np.array(nodes)), color='black', s=25, zorder=5)

plt.title("Approximation de f avec des nœuds de Tchebychev par l'interpolation de Lagrange")
plt.xlabel("x")
plt.ylabel("f(x) (échelle logarithmique)")
plt.legend()
plt.grid()
plt.savefig("approximation_chebyshev_nodes.svg")  # Enregistrer le graphique en SVG
plt.close()  # Fermer la figure

# Erreur pour les nœuds uniformément distribués
plt.figure(figsize=(12, 6))

for i, ncas in enumerate(n_values):
    nodes = uniform_nodes(ncas)
    LagrangeVals = [lagrange_interpolation(f, nodes, mpmath.mpf(val)) for val in x]
    error_vals = [abs(f(val) - LagrangeVals[i]) for i, val in enumerate(x)]
    
    max_derivative = max([abs(mpmath.diff(f, val, ncas)) for val in x])
    factorial_value = mpmath.factorial(ncas)
    lagrange_bound = [(max_derivative / factorial_value) * abs(Wnplusone(nodes)(val)) for val in x]

    plt.plot(x, error_vals, label=f"Erreur d'approximation, n={ncas}", linestyle="solid", color=colors[i])  # Utilisation des couleurs définies
    plt.plot(x, lagrange_bound, label=f"Reste de Lagrange, n={ncas}", linestyle="dashed", color=colors[i])  # Utilisation des couleurs définies

plt.xlabel("x")
plt.ylabel("|f(x) - p(x)| (échelle logarithmique)")
plt.ylim(10**-5, 10**5)
plt.title("Comparaison des erreurs d'approximation et des bornes théoriques de Lagrange pour les nœuds uniformément distribués")
plt.legend()
plt.grid()
plt.savefig("errors_uniform_nodes.svg")  # Enregistrer le graphique en SVG
plt.close()  # Fermer la figure

# Erreur pour les nœuds de Tchebychev
plt.figure(figsize=(12, 6))

for i, ncas in enumerate(n_values):
    nodes = chebyshev_nodes(ncas)
    LagrangeVals = [lagrange_interpolation(f, nodes, mpmath.mpf(val)) for val in x]
    error_vals = [abs(f(val) - LagrangeVals[i]) for i, val in enumerate(x)]
    
    max_derivative = max([abs(mpmath.diff(f, val, ncas)) for val in x])
    factorial_value = mpmath.factorial(ncas)
    lagrange_bound = [(max_derivative / factorial_value) * abs(Wnplusone(nodes)(val)) for val in x]

    plt.plot(x, error_vals, label=f"Erreur d'approximation, n={ncas}", linestyle="solid", color=colors[i])  # Utilisation des couleurs définies
    plt.plot(x, lagrange_bound, label=f"Reste de Lagrange, n={ncas}", linestyle="dashed", color=colors[i])  # Utilisation des couleurs définies

plt.yscale("log")
plt.xlabel("x")
plt.ylabel("|f(x) - p(x)| (échelle logarithmique)")
plt.ylim(10**-5, 10**5)
plt.title("Comparaison des erreurs d'approximation et des bornes théoriques de Lagrange pour les nœuds de Tchebychev")
plt.legend()
plt.grid()
plt.savefig("errors_chebyshev_nodes.svg")  # Enregistrer le graphique en SVG
plt.close()  # Fermer la figure
