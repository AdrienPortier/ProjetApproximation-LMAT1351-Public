from mpmath import polyroots, mp
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x = sp.Symbol('x')

def Legendre(n):
    """Génère les polynômes de Legendre normalisés"""
    P = []  # p_n = (2n-1)/n x p_n-1 - (n-1)/n p_n-2
    for i in range(n+1):
        if i == 0:
            P.append(np.array([1]))
        elif i == 1:
            P.append(np.array([1,0]))
        else:
            arr = np.array([0])
            minus1 = np.concatenate((P[i-1], arr))
            minus2 = np.concatenate((arr,arr, P[i-2]))
            P.append(((2*i-1)/i) * minus1 - ((i-1)/i) * minus2)
    return P

n = 30
L = Legendre(n)

# Coefficients
A = [sp.Poly(L[i], x).all_coeffs() for i in range(n+1)]
C = [[float(e) for e in coeffs] for coeffs in A]

X = np.linspace(-1,1,1000)
plt.figure(figsize=(10, 6))
for i in range(n+1):
    poly_coeffs = list(L[i])
    Y = [mp.polyval(poly_coeffs, x) for x in X]
    plt.plot(X, Y)

plt.title('Polynômes de Legendre ($p_n$) pour n allant de 1 à 30')
plt.xlabel('x')
plt.ylabel('$p_n(x)$')
plt.grid(True)
plt.savefig("3.2_rep.svg")
plt.show()


mp.dps = 10

Roots = []
for coeffs in C:
    try:
        r = polyroots(coeffs)
    except:
        r = sp.Poly(coeffs, x).all_roots()
    Roots.append([float(r_i) for r_i in r])

# Rassemblement des racines
all_roots = np.concatenate(Roots)


# Distribution
for degree, roots in enumerate(Roots):
    plt.scatter(roots, [degree] * len(roots), color='black', s=10, label=f"Degree {degree}" if degree == 0 else "")
plt.title('Distribution des racines des polynômes de Legendre')
plt.xlabel('Racines')
plt.ylabel('Degré du polynôme')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("3.3_distrib_racines.svg")
plt.show()

# Histogramme
plt.hist(all_roots, bins=30, density=False, alpha=0.6, color='skyblue', edgecolor='black')
plt.ylabel("Fréquence")
plt.xlabel("Valeurs des racines")
plt.title("Histogramme des racines des polynômes de Legendre pour n de 1 à 30")
plt.savefig("3.3_histo_racines.svg")
plt.show()