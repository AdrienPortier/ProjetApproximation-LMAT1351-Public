import numpy as np
import sympy as sp
from mpmath import polyroots
x = sp.Symbol('x')
# Paramètres
n = 15-1  # x0, ... , xn noeuds, si n = 9, 10 termes.

# Fonctions à intégrer

f1 = sp.sympify("1/(1+25*x**2)") # Intégrale vaut 0.54936
f2 = sp.sympify("(x**2)**(1/2)") # Intégrale vaut 1
f3 = sp.sympify(sp.exp(x)) #Intégrale vaut 2.3504

# Subterfuge pour intégrer les fi avec le poids de Tchebychev
g1 = sp.sympify(sp.sqrt(1-x**2)*f1)
g2 = sp.sympify(sp.sqrt(1-x**2)*f2)
g3 = sp.sympify(sp.sqrt(1-x**2)*f3)

def Lagrange(f,nodes):
    l = sp.simplify(0)
    for k, knode in enumerate(nodes):
        p = 1
        for j, jnode in enumerate(nodes):
            if j != k:
                p *= (x-jnode)/(knode-jnode)
        p *= f.subs(x,knode)
        l+= p
    return l
    
def Legendre(n):
    P = [1,x]
    for i in range(2,n+1):
        P.append(sp.simplify((P[i-1]*x*(2*(i-1)+1)-(i-1)*P[i-2])/(i)))
    return P


def IntegrateLagrange(l):
    res = sp.integrate(l,(x,-1,1))
    return res.evalf()

L = Legendre(n)
A = [sp.Poly(L[i], x).all_coeffs() for i in range(n+1)]
C = [[float(e) for e in coeffs] for coeffs in A]
# Calcul des racines des polynômes de Legendre
Roots = []
try:
    Roots = polyroots(C[-1])
except:
    Roots = np.roots(C[-1])
true_values = {

    '1': 2/5 * (np.arctan(5)),

    '2': 1.0,

    '3': np.exp(1)-np.exp(-1)

}

results_legendre = {}

for name, fonct in zip(['1', '2', '3'], [f1, f2, f3]):
    l = Lagrange(fonct, Roots)
    legendre_estimation = IntegrateLagrange(l)
    true_value = true_values[name]
    results_legendre[name] = {
        'Vraie valeur': true_value,
        'Estimation par Legendre': legendre_estimation,
        'Erreur par Legendre' : np.abs(true_value-legendre_estimation)
    }
    results_legendre["Nombre de noeuds"] = n+1

results = {}

for name, fonct in zip(['1', '2', '3'], [g1, g2, g3]):
    Chebyshev_approx = np.sum([( np.pi/(n+1) )*fonct.subs(x, np.cos(((2*k+1)*np.pi)/(2*n+2)) ) for k in range(0,n+1)]) # Racines pour k = 0,..n, => n+1 racines, formule exercice 5.6.3 du syllabus.
    true_value = true_values[name]
    results[name] = {
        'Vraie valeur': true_value,
        'Estimation par Chebychev': Chebyshev_approx,
        'Erreur par Chebychev' : np.abs(true_value-Chebyshev_approx)
    }
    results["Nombre de noeuds"] = n+1


for k in range(1,4):
    print(results_legendre[str(k)])
    print(results[str(k)])
def plot():
    import matplotlib.pyplot as plt
    # Plot pour la décroissance des erreurs en fonction de n
    n_values = range(1, 24+1) # 1,...,24 -> nombre de noeuds : 2,...,25
    errors_legendre = {'1': [], '2': [], '3': []}
    errors_chebyshev = {'1': [], '2': [], '3': []}

    for n in n_values:
        # Recalcul des erreurs pour chaque n
        L = Legendre(n)
        A = [sp.Poly(L[i], x).all_coeffs() for i in range(n)]
        C = [[float(e) for e in coeffs] for coeffs in A]
        try:
            Roots = polyroots(C[-1])
        except:
            Roots = np.roots(C[-1])

        for name, fonct in zip(['1', '2', '3'], [f1, f2, f3]):
            l = Lagrange(fonct, Roots)
            legendre_estimation = IntegrateLagrange(l)
            true_value = true_values[name]
            errors_legendre[name].append(np.abs(true_value - legendre_estimation))

        for name, fonct in zip(['1', '2', '3'], [g1, g2, g3]):
            Chebyshev_approx = np.sum([(np.pi / (n)) * fonct.subs(x, np.cos(((2 * k + 1) * np.pi) / (2 * n))) for k in range(n)])
            true_value = true_values[name]
            errors_chebyshev[name].append(np.abs(true_value - Chebyshev_approx))

    for name in ['1', '2', '3']:
        plt.figure()
        plt.plot(n_values, errors_legendre[name], label='Erreur Legendre', marker='o')
        plt.plot(n_values, errors_chebyshev[name], label='Erreur Tchebychev', marker='x')
        plt.xlabel('Valeur de n (n+1 noeuds)')
        plt.ylabel('Erreur')
        plt.title(f'Erreur de quadrature pour la fonction {name}')
        plt.xticks(n_values)
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()
def dichotomie():
    tolerance_levels = [1e-2, 1e-4, 1e-6]
    n_min = 1
    n_max = 100  # On fixe une borne supérieure pour la recherche

    results_thresholds = {
        'Legendre': {'1': {}, '2': {}, '3': {}},
        'Chebyshev': {'1': {}, '2': {}, '3': {}}
    }

    def find_min_n(fonct, method, threshold, n_min=1, n_max=100):
        while n_min < n_max:
            n_mid = (n_min + n_max) // 2

            if method == 'Legendre':
                L = Legendre(n_mid)
                A = [sp.Poly(L[i], x).all_coeffs() for i in range(n_mid)]
                C = [[float(e) for e in coeffs] for coeffs in A]
                try:
                    Roots = polyroots(C[-1])
                except:
                    Roots = np.roots(C[-1])
                l = Lagrange(fonct, Roots)
                estimation = IntegrateLagrange(l)
            else:
                estimation = np.sum([
                    (np.pi / (n_mid)) * fonct.subs(x, np.cos(((2 * k + 1) * np.pi) / (2 * n_mid)))
                    for k in range(n_mid)
                ])

            true_value = true_values[name]
            error = np.abs(true_value - estimation)

            if error <= threshold:
                n_max = n_mid
            else:
                n_min = n_mid + 1

        return n_min

    for name, fonct in zip(['1', '2', '3'], [f1, f2, f3]):
        for tol in tolerance_levels:
            n_legendre = find_min_n(fonct, 'Legendre', tol)
            results_thresholds['Legendre'][name][tol] = n_legendre

    for name, fonct in zip(['1', '2', '3'], [g1, g2, g3]):
        for tol in tolerance_levels:
            n_chebyshev = find_min_n(fonct, 'Chebyshev', tol)
            results_thresholds['Chebyshev'][name][tol] = n_chebyshev

    # Affichage des résultats
    for method in ['Legendre', 'Chebyshev']:
        for func in ['1', '2', '3']:
            print(f"Méthode: {method}, Fonction: {func}")
            for tol in tolerance_levels:
                print(f" - Tolérance {tol}: n = {results_thresholds[method][func][tol]}")
dichotomie()

