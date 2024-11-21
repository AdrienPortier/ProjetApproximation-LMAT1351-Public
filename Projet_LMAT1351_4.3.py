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

def Legendre(n):
    """
    Renvoie la liste des polynômes de Legendre normalisés (divisés par leur norme) de degré 0 à n 

    Paramètres
    ----------
    n : int
        Le degré du dernier polynôme de Legendre de la liste

    Retourne
    -------
    List(np.array)
        La liste contenant les polynômes de Legendre normalisés
    """
    P = [1,x]
    for i in range(2,n+1):
        P.append(sp.simplify((P[i-1]*x*(2*(i-1)+1)-(i-1)*P[i-2])/(i)))
    return P

# Méthode de Golub-Welsch pour calculer racines et poids
from numpy.polynomial.legendre import leggauss

def legendre_roots_weights(n):
    """
    Renvoie les racines du polynôme orthogonal de degré n et les poids pour la quadrature de Gauss via les polynômes de Legendre

    Paramètres
    ----------
    n : int
        Le degré du polynôme de Legendre

    Retourne
    -------
    List(float), List(float)
        La liste contenant les racines du polynôme orthogonal de Legendre de degré n,
        La liste contenant les A_k servant de poids pour la quadrature de Gauss via les polynômes de Legendre
    """
    
    roots, weights = leggauss(n)
    return roots, weights

# Intégration via quadrature
def gauss_quad(f, roots, weights):
    """
    Renvoie l'estimation de l'intégrale de f par la règle de quadrature associée aux poids et aux racines du polynôme orthogonal.

    Paramètres
    ----------
    f : function
        La fonction f à évaluer aux racines du polynôme orthogonal
    racines : List(float)
        La liste des racines du polynôme orthogonal de degré n
    poids : List(float)
        La liste des A_k servant à la règle de quadrature
    Retourne
    -------
    float
        L'estimation de l'intégrale par la règle de quadrature
    """
    return sum(w * f.subs(x, r) for r, w in zip(roots, weights))

true_values = {
    '1': 2/5 * (np.arctan(5)),
    '2': 1.0,
    '3': np.exp(1)-np.exp(-1)
}


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
noeuds_min()