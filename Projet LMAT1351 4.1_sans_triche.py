import numpy as np
from mpmath import mp,polyroots,exp,atan,pi,cos,fmul,fdiv

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


polyroots_cache = {}

def polyroots_memory(poly, extraprec=50, maxsteps=200):
    poly_tuple = tuple(poly)
    if poly_tuple in polyroots_cache:
        return polyroots_cache[poly_tuple]
    else:
        roots = np.array(polyroots(poly, extraprec=extraprec, maxsteps=maxsteps))
        polyroots_cache[poly_tuple] = roots
        return roots
    

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
    P = [np.array([1.0]), np.array([1.0, 0.0])]
    for i in range(2, n+1):
        Pn = ((2 * (i-1) + 1) * np.polymul(P[-1], [1.0, 0.0]) - (i-1) * np.pad(P[-2], (2, 0), mode='constant')) / i
        P.append(Pn)
    return P



def obtenir_racines_poids_Legendre(poly_orth):
    """
    Renvoie les racines du polynôme orthogonal de degré n et les poids pour la quadrature de Gauss via les polynômes de Legendre

    Paramètres
    ----------
    poly_orth : List(np.array)
        Les polynômes orthogonaux de Legendre de degré 0 à n + 2

    Retourne
    -------
    List(float), List(float)
        La liste contenant les racines du polynôme orthogonal de Legendre de degré n+1,
        La liste contenant les A_k servant de poids pour la quadrature de Gauss via les polynômes de Legendre
    """
    poly = poly_orth[-2]
    racines = np.array(polyroots_memory(poly))  # n + 1 racines
    poly = np.polyder(poly)
    Ak = []
    for r in racines:
        poids = fdiv(2, fmul( (1 - r**2),(np.polyval(poly,r))**2))
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

poly_Legendre = Legendre(n+2)
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
    poly_Legendre = Legendre(n_values[-1]+2) # si valeur n, nous voulons n+1 racines, mais besoin de poly de deg n+2 pour calcul formule Legendre

    for n in n_values:

        for name, fonct in zip(['1', '2', '3'], [f1, f2, f3]):
            racines, poids = obtenir_racines_poids_Legendre(poly_Legendre[:(n+2)+1])
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

def noeuds_min():
    def find_min_n_optimized(fonct, methode, name, seuil,n_max,poly_legendre = None, n_min=2):
        while n_min < n_max:
            n_mid = (n_min + n_max) // 2
            
            if methode == 'Legendre':
                racines, poids = obtenir_racines_poids_Legendre(poly_legendre[:n_mid+2+1])
                estimation = gauss_quad(fonct, racines, poids)
            else:
                estimation = gauss_quad(fonct, [cos((2*k+1)*pi/((2*n_mid)+2)) for k in range(0,n_mid+1)],  [ pi/(n_mid+1) for k in range(0,n_mid+1)]) 
            vraie_valeur = valeurs_réelles[name]
            erreur = abs(vraie_valeur - estimation)
            if erreur <= seuil:
                n_max = n_mid 
            else:
                n_min = n_mid+1
        return n_min
    n_max = 100
    poly_legendre_nmax = Legendre(n_max)
    seuils = [1e-2, 1e-4, 1e-6]
    resultats_erreurs = {'Legendre': {}, 'Chebychev': {}}
    methode = 'Legendre'
    for name, fonct in zip(['1', '2', '3'], [f1, f2, f3]):
        for seuil in seuils:
            min_n = find_min_n_optimized(fonct, methode, name, seuil, n_max, poly_legendre= poly_legendre_nmax)
            resultats_erreurs[methode].setdefault(name, {})[seuil] = min_n
            print(seuil)
    methode = "Chebychev"
    for name, fonct in zip(['1', '2', '3'], [g1, g2, g3]):
        for seuil in seuils:
            min_n = find_min_n_optimized(fonct, methode, name, seuil,n_max = n_max)
            resultats_erreurs[methode].setdefault(name, {})[seuil] = min_n
            print(seuil)

    # Affichage des résultats
    for method in ['Legendre', 'Chebychev']:
        for func in ['1', '2', '3']:
            print(f"Méthode: {method}, Fonction: {func}")
            for seuil in seuils:
                print(f" - Seuil d'erreur {seuil}: # noeuds (n+1) = {resultats_erreurs[method][func][seuil]+1}")
noeuds_min()
