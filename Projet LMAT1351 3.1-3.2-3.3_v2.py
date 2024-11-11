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

# Create a list of normalized Legendre polynomials
n = 30
L = Legendre(n)

# Coefficients
A = [sp.Poly(L[i], x).all_coeffs() for i in range(n)]
C = [[float(e) for e in coeffs] for coeffs in A]

X = np.linspace(-1,1,1000)
plt.figure(figsize=(10, 6))
for i in range(n+1):
    # Convertir le polynôme NumPy en liste pour le passer à mp.polyval
    poly_coeffs = list(L[i])
    Y = [mp.polyval(poly_coeffs, x) for x in X]  # Evaluation du polynôme
    plt.plot(X, Y)

plt.title('Polynômes de Legendre normalisés')
plt.xlabel('x')
plt.ylabel('P_n(x)')
plt.grid(True)
plt.show()


# Adjust precision for numerical calculations
mp.dps = 10

Roots = []
for coeffs in C:
    try:
        r = np.roots(coeffs)
    except:
        r = sp.Poly(coeffs, x).all_roots()
    Roots.append([float(r_i) for r_i in r])

# Rassemblement des racines
all_roots = np.concatenate(Roots)


# Plotting
for degree, roots in enumerate(Roots):
    plt.scatter(roots, [degree] * len(roots), color='black', s=10, label=f"Degree {degree}" if degree == 0 else "")

plt.title('Distribution des racines des polynômes de Legendre')
plt.xlabel('Racines')
plt.ylabel('Degré du polynôme')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Histogram
plt.hist(all_roots, bins=30, density=False, alpha=0.6, color='skyblue', edgecolor='black')
plt.show()