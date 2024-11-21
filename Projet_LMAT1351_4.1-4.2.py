import numpy as np
from mpmath import mp,exp,atan,pi,cos

mp.dps = 50
    
# Paramètres
n = 25  # noeuds x0, ... , xn => n+1 noeuds

# Fonctions à intégrer
f1 = lambda x: 1 / (1 + 25 * x**2)  # Intégrale vaut 0.54936
f2 = lambda x: np.abs(x)            # Intégrale vaut 1
f3 = lambda x: exp(x)            # Intégrale vaut 2.3504

# Subterfuge pour intégrer les fi avec le poids de Tchebychev
g1 = lambda x: np.sqrt(1 - x**2) * f1(x)
g2 = lambda x: np.sqrt(1 - x**2) * f2(x)
g3 = lambda x: np.sqrt(1 - x**2) * f3(x)

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
    P = [np.poly1d([1.0]), np.poly1d([1.0, 0.0])]
    for i in range(2, n+1):
        Pn = ((2 * i - 1) * np.poly1d([1.0, 0.0]) * P[-1] - (i - 1) * P[-2]) / i
        P.append(Pn)
    return P


def obtenir_racines_poids_Legendre(poly):
    """
    Renvoie les racines du polynôme orthogonal de degré n et les poids pour la quadrature de Gauss via les polynômes de Legendre

    Paramètres
    ----------
    poly_orth : List(np.array)
        Les polynômes orthogonaux de Legendre de degré n + 1

    Retourne
    -------
    List(float), List(float)
        La liste contenant les racines du polynôme orthogonal de Legendre de degré n+1,
        La liste contenant les A_k servant de poids pour la quadrature de Gauss via les polynômes de Legendre
    """
    racines = np.roots(poly)  # n + 1 racines
    poly = np.poly1d(np.polyder(poly))
    Ak = []
    for r in racines:
        poids = 2/( (1 - np.power(r,2))*(np.power(poly(r),2)) )
        Ak.append(poids)
    return racines, Ak

def gauss_quad(f, racines, poids):
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
    return sum(w * f(r) for r, w in zip(racines, poids))

valeurs_réelles = {
    '1': 2/5 * (atan(5)),
    '2': 1.0,
    '3': exp(1)-exp(-1)
}

# Legendre
resultats_legendre = {}

poly_Legendre = Legendre(n+1)
for name, fonct in zip(['1', '2', '3'], [f1, f2, f3]):
    racines, poids = obtenir_racines_poids_Legendre(poly_Legendre[-1])
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
    estimation_Chebychev = gauss_quad(fonct, [cos((2*k+1)*pi/(2*n+2)) for k in range(0,n+1)],  [ pi/(n+1) for k in range(0,n+1)]) # Formule exercice 5.6.3 du syllabus.
    vraie_valeur = valeurs_réelles[name]
    
    resultats_Chebychev[name] = {
        'Vraie valeur': vraie_valeur,
        'Estimation par Chebychev': estimation_Chebychev,
        'Erreur par Chebychev': np.abs(vraie_valeur - estimation_Chebychev)
    }

def plot():
    import matplotlib.pyplot as plt
    # Plot pour la décroissance des erreurs en fonction de n
    n_values = range(0, 25) # 0,...,24 -> nombre de noeuds : 1,...,25
    errors_legendre = {'1': [], '2': [], '3': []}
    errors_chebyshev = {'1': [], '2': [], '3': []}
    poly_Legendre = Legendre(n_values[-1]+1) # si valeur n, nous voulons n+1 racines

    for n in n_values:

        for name, fonct in zip(['1', '2', '3'], [f1, f2, f3]):
            racines, poids = obtenir_racines_poids_Legendre(poly_Legendre[n+1])
            estimation_legendre = gauss_quad(fonct, racines, poids)
            vraie_valeur = valeurs_réelles[name]
            errors_legendre[name].append(np.abs(vraie_valeur - estimation_legendre))
        

        for name, fonct in zip(['1', '2', '3'], [g1, g2, g3]):
            estimation_Chebychev = gauss_quad(fonct, [np.cos((2*k+1)*np.pi/((2*n)+2)) for k in range(0,n+1)],  [ np.pi/(n+1) for k in range(0,n+1)]) # Formule exercice 5.6.3 du syllabus.
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
