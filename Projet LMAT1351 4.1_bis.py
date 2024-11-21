import numpy as np
import sympy as sp
from mpmath import polyroots
x = sp.Symbol('x')

# Paramètres
n = 0  # noeuds x0, ... , xn => n+1 noeuds

# Fonctions à intégrer
f1 = sp.sympify("1/(1+25*x**2)")  # Intégrale vaut 0.54936
f2 = sp.sympify(sp.Abs(x))  # Intégrale vaut 1
f3 = sp.sympify(sp.exp(x))  # Intégrale vaut 2.3504

# Subterfuge pour intégrer les fi avec le poids de Tchebychev
g1 = sp.sympify(sp.sqrt(1-x**2)*f1)
g2 = sp.sympify(sp.sqrt(1-x**2)*f2)
g3 = sp.sympify(sp.sqrt(1-x**2)*f3)

#Exemple : n = 0, Legendre estim = 2, Cheby estim = pi
def Lagrange(f, nodes):
    l = sp.simplify(0)
    for k, knode in enumerate(nodes):
        p = 1
        for j, jnode in enumerate(nodes):
            if j != k:
                p *= (x - jnode) / (knode - jnode)
        p *= f.subs(x, knode)
        l += p
    return l
def Legendre(n):
    P = [1,x]
    for i in range(2,n+1):
        P.append(sp.simplify((P[i-1]*x*(2*(i-1)+1)-(i-1)*P[i-2])/(i)))
    return P

def IntegrateLagrange(l):
    res = sp.integrate(l, (x, -1, 1))
    return res.evalf()

# Méthode de Golub-Welsch pour calculer racines et poids
from numpy.polynomial.legendre import leggauss

def legendre_roots_weights(n):
    roots, weights = leggauss(n)  # Calcul direct via numpy
    return roots, weights

# Intégration via quadrature
def gauss_quad(f, roots, weights):
    return sum(w * f.subs(x, r) for r, w in zip(roots, weights))

true_values = {
    '1': 2/5 * (np.arctan(5)),
    '2': 1.0,
    '3': np.exp(1)-np.exp(-1)
}

# Comparaison entre Legendre et Chebychev
results_legendre = {}
results_Chebychev = {}

for name, fonct in zip(['1', '2', '3'], [f1, f2, f3]):
    # Legendre
    roots, weights = legendre_roots_weights(n+1)
    estimation_legendre = gauss_quad(fonct, roots, weights)
    true_value = true_values[name]
    
    results_legendre[name] = {
        'Vraie valeur': true_value,
        'Estimation par Legendre': estimation_legendre,
        'Erreur par Legendre': np.abs(true_value - estimation_legendre)
    }
for name, fonct in zip(['1', '2', '3'], [g1, g2, g3]):
    # Chebychev
    estimation_Chebychev = np.sum([( np.pi/(n+1) )*fonct.subs(x, np.cos(((2*k+1)*np.pi)/(2*n+2)) ) for k in range(0,n+1)]) # Racines pour k = 0,..n, => n+1 racines, formule exercice 5.6.3 du syllabus.
    true_value = true_values[name]
    
    results_Chebychev[name] = {
        'Vraie valeur': true_value,
        'Estimation par Chebychev': estimation_Chebychev,
        'Erreur par Chebychev': np.abs(true_value - estimation_Chebychev)
    }

def plot():
    import matplotlib.pyplot as plt
    # Plot pour la décroissance des erreurs en fonction de n
    n_values = range(0, 24+1) # 0,...,24 -> nombre de noeuds : 1,...,25
    errors_legendre = {'1': [], '2': [], '3': []}
    errors_chebyshev = {'1': [], '2': [], '3': []}
    legendre_poly = Legendre(n_values[-1]+1)

    for n in n_values:
        # Recalcul des erreurs pour chaque n
        L = legendre_poly[n+1]
        A = sp.Poly(L, x).all_coeffs()
        C = [float(e) for e in A]
        try:
            Roots = polyroots(C)
        except:
            Roots = np.roots(C)

        for name, fonct in zip(['1', '2', '3'], [f1, f2, f3]):
            l = Lagrange(fonct, Roots)
            legendre_estimation = IntegrateLagrange(l)
            true_value = true_values[name]
            errors_legendre[name].append(np.abs(true_value - legendre_estimation))

        for name, fonct in zip(['1', '2', '3'], [g1, g2, g3]):
            Chebyshev_approx = np.sum([(np.pi / (n+1)) * fonct.subs(x, np.cos(((2 * k + 1) * np.pi) / (2 * n + 2))) for k in range(n+1)])
            true_value = true_values[name]
            errors_chebyshev[name].append(np.abs(true_value - Chebyshev_approx))
    number_nodes = n_values+np.ones(len(n_values)) 
    for name in ['1', '2', '3']:
        plt.figure()
        plt.plot(number_nodes, errors_legendre[name], label='Erreur Legendre', marker='o')
        plt.plot(number_nodes, errors_chebyshev[name], label='Erreur Tchebychev', marker='x')
        plt.xlabel('Nombre de noeuds (n+1)')
        plt.ylabel('Erreur')
        plt.title(f'Erreur de quadrature pour la fonction {name}')
        plt.xticks(number_nodes)
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()


def noeuds_min():
    def find_min_n_optimized(fonct, method, threshold, n_min=1, n_max=1500):
        while n_min < n_max:
            n_mid = (n_min + n_max) // 2
            
            if method == 'Legendre':
                roots, weights = legendre_roots_weights(n_mid)
                estimation = gauss_quad(fonct, roots, weights)
            else:
                estimation = np.sum([
                        (np.pi / (n_mid+1)) * fonct.subs(x, np.cos(((2 * k + 1) * np.pi) / (2 * n_mid+2)))
                        for k in range(n_mid+1)
                    ])
            true_value = true_values[name]
            error = abs(true_value - estimation)
            
            if error <= threshold:
                n_max = n_mid
            else:
                n_min = n_mid + 1
        
        return n_min

    tolerance_levels = [1e-2, 1e-4, 1e-6]
    results_thresholds = {'Legendre': {}, 'Chebychev': {}}
    method = 'Legendre'
    for name, fonct in zip(['1', '2', '3'], [f1, f2, f3]):
        for tol in tolerance_levels:
            min_n = find_min_n_optimized(fonct, method, tol)
            results_thresholds[method].setdefault(name, {})[tol] = min_n
    method = "Chebychev"
    for name, fonct in zip(['1', '2', '3'], [g1, g2, g3]):
        for tol in tolerance_levels:
            min_n = find_min_n_optimized(fonct, method, tol)
            results_thresholds[method].setdefault(name, {})[tol] = min_n

    # Affichage des résultats
    for method in ['Legendre', 'Chebychev']:
        for func in ['1', '2', '3']:
            print(f"Méthode: {method}, Fonction: {func}")
            for tol in tolerance_levels:
                print(f" - Seuil d'erreur {tol}: # noeuds (n+1) = {results_thresholds[method][func][tol]}")
plot()