import numpy as np
import sympy as sp
from mpmath import polyroots
x = sp.Symbol('x')

# Paramètres
n = 20  # noeuds x0, ... , xn => n+1 noeuds

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
    List(sp.Poly)
        La liste contenant les polynômes de Legendre normalisés
    """
    P = [1,x]
    for i in range(2,n+1):
        ai = i/( np.sqrt(4*(i**2) -1) )
        aim1 = (i-1)/( np.sqrt( 4*((i-1)**2) - 1))
        P.append( sp.Poly( (x/ai)*P[i-1] - (aim1/ai)*P[i-2] ) )
    return P

def obtenir_racines_poids_Legendre(poly_orth):
    """
    Renvoie les racines du polynôme orthogonal de degré n et les poids pour la quadrature de Gauss via les polynômes de Legendre

    Paramètres
    ----------
    poly_orth : List(sp.Poly)
        Les polynômes orthogonaux de Legendre de degré 0 à n + 1

    Retourne
    -------
    List(int), List(int)
        La liste contenant les racines du polynôme orthogonal de Legendre de degré n,
        La liste contenant les A_k servant de poids pour la quadrature de Gauss via les polynômes de Legendre
    """
    racines = [sp.N(r) for r in sp.roots(poly_orth[-1], x).keys()]
    Ak = []
    phi0 = sp.Poly(0,x)
    phi1 = sp.Poly(2,x)
    for k in range(2,len(poly_orth)):
        a_km1 = ((k-1))/np.sqrt((4*((k-1)**2) -1))
        phi1,phi0 = sp.Poly(x*phi1 - (a_km1**2) * phi0) , phi1
    W = sp.Poly(1,x)
    for racine in racines:
        W *= (x-racine)
    Wprime = sp.diff(W, x)
    for k in range(0,len(racines)):
        Ak.append(phi1.subs(x,racines[k])/Wprime.subs(x,racines[k]))
    return racines, Ak


def gauss_quad(f, racines, poids):
    """
    Renvoie l'estimation de l'intégrale de f par la règle de quadrature associée aux poids et aux racines du polynôme orthogonal.

    Paramètres
    ----------
    f : function
        La fonction f à évaluer aux racines du polynôme orthogonal
    racines : List(int)
        La liste des racines du polynôme orthogonal de degré n
    poids : List(int)
        La liste des A_k servant à la règle de quadrature
    Retourne
    -------
    int
        L'estimation de l'intégrale par la règle de quadrature
    """
    return sum(w * f.subs(x, r) for r, w in zip(racines, poids))

valeurs_réelles = {
    '1': 2/5 * (np.arctan(5)),
    '2': 1.0,
    '3': np.exp(1)-np.exp(-1)
}

# Legendre
resultats_legendre = {}

poly_Legendre = Legendre(n+1)
for name, fonct in zip(['1', '2', '3'], [f1, f2, f3]):
    racines, poids = obtenir_racines_poids_Legendre(poly_Legendre)
    estimation_legendre = gauss_quad(fonct, racines, poids)
    vraie_valeur = valeurs_réelles[name]
    
    resultats_legendre[name] = {
        'Vraie valeur': vraie_valeur,
        'Estimation par Legendre': estimation_legendre,
        'Erreur par Legendre': np.abs(vraie_valeur - estimation_legendre)
    }
# Chebychev
resultats_Chebychev = {}
for name, fonct in zip(['1', '2', '3'], [g1, g2, g3]):
    estimation_Chebychev = gauss_quad(fonct, [np.cos((2*k+1)*np.pi/(2*n+2)) for k in range(0,n+1)],  [ np.pi/(n+1) for k in range(0,n+1)]) # Formule exercice 5.6.3 du syllabus.
    vraie_valeur = valeurs_réelles[name]
    
    resultats_Chebychev[name] = {
        'Vraie valeur': vraie_valeur,
        'Estimation par Chebychev': estimation_Chebychev,
        'Erreur par Chebychev': np.abs(vraie_valeur - estimation_Chebychev)
    }

def plot():
    import matplotlib.pyplot as plt
    # Plot pour la décroissance des erreurs en fonction de n
    n_values = range(1, 25) # 1,...,24 -> nombre de noeuds : 2,...,25
    errors_legendre = {'1': [], '2': [], '3': []}
    errors_chebyshev = {'1': [], '2': [], '3': []}
    poly_Legendre = Legendre(n_values[-1]+1)

    for n in n_values:

        for name, fonct in zip(['1', '2', '3'], [f1, f2, f3]):
            racines, poids = obtenir_racines_poids_Legendre(poly_Legendre[:n+1])
            estimation_legendre = gauss_quad(fonct, racines, poids)
            vraie_valeur = valeurs_réelles[name]
            errors_legendre[name].append(np.abs(vraie_valeur - estimation_legendre))
        

        for name, fonct in zip(['1', '2', '3'], [g1, g2, g3]):
            estimation_Chebychev = gauss_quad(fonct, [np.cos((2*k+1)*np.pi/(2*n+2)) for k in range(0,n+1)],  [ np.pi/(n+1) for k in range(0,n+1)]) # Formule exercice 5.6.3 du syllabus.
            vraie_valeur = valeurs_réelles[name]
            errors_chebyshev[name].append(np.abs(vraie_valeur - estimation_Chebychev))

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
plot()