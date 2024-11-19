import numpy as np
import sympy as sp
from mpmath import polyroots
x = sp.Symbol('x')

f1 = sp.sympify("1/(1+25*x**2)") # Intégrale vaut 0.54936
f2 = sp.sympify("(x**2)**(1/2)") # Intégrale vaut 1
f3 = sp.sympify(sp.exp(x)) #Intégrale vaut 2.3504

g1 = sp.sympify(sp.sqrt(1-x**2)*f1)
g2 = sp.sympify(sp.sqrt(1-x**2)*f2)
g3 = sp.sympify(sp.sqrt(1-x**2)*f3)
def Lagrange(N,f):
    l = sp.simplify(0)
    for k in range(0,len(N)):
        p = 1
        for j in range(0,len(N)):
            if j != k:
                p *= (x-N[j])/(N[k]-N[j]) # erreur de signe du dénominateur
        p *= f.subs(x,N[k])
        l+= p
    return l
    
def Legendre(n):
    P = [1,x]
    for i in range(2,n+1):
        P.append(sp.simplify((P[i-1]*x*(2*(i-1)+1)-(i-1)*P[i-2])/(i)))
    return P

def Chebyshev(n):
    P = [1,x]
    for i in range(2,n+1):
        P.append(sp.simplify(2*x*P[i-1]-P[i-2]))
    return P


def A(l,w = 1):
    res = sp.integrate(l*w,(x,-1,1))
    return res.evalf()

n = 10

L = Legendre(n+1)

#print(A(sp.sympify(1/(sp.sqrt(1-x**2))),C))

#Création de la matrice des coefficients (stockés sous forme sympy)
B = []
B.append(sp.Poly(L[-1],x).all_coeffs())

#Création de la matrice des coefficients (stockés sous forme de float)
C = []
for e in B[-1]:
    C.append(float(e))

    
#Calcul des racines des polynômes de Legendre
Roots = [np.roots(C)]
true_values = {

    '1': 0.54936,

    '2': 1.0,

    '3': 2.3504

}

results_legendre = {}

for name, fonct in zip(['1', '2', '3'], [f1, f2, f3]):
    l = Lagrange(Roots[0],fonct)
    legendre_estimation = A(l)
    true_value = true_values[name]
    results_legendre[name] = {
        'Vraie valeur': true_value,
        'Estimation par Legendre': legendre_estimation,
        'Erreur par Legendre' : np.abs(true_value-legendre_estimation)
    }
    results_legendre["Nombre de noeuds"] = n



C = Chebyshev(n+1)
B = []
B.append(sp.Poly(C[-1],x).all_coeffs())
D = []
for e in B[-1]:
    D.append(float(e))
Roots = [np.roots(D)]
'''
#print(((Integral(A(sp.sympify(1/(sp.sqrt(1-x**2))),l),Roots[0],f)).subs("e",2.71828182846)))
print(A(sp.sympify(1/(sp.sqrt(1-x**2))),l).evalf())
l = Lagrange(Roots[0],f2)
print(A(sp.sympify(1/(sp.sqrt(1-x**2))),l).evalf())
'''

results = {}

for name, fonct in zip(['1', '2', '3'], [g1, g2, g3]):
    l = Lagrange(Roots[0],fonct)
    Chebyshev_approx = A(l, sp.sympify( 1/(sp.sqrt(1-x**2)) ) )
    true_value = true_values[name]
    results[name] = {
        'Vraie valeur': true_value,
        'Estimation par Chebychev': Chebyshev_approx,
        'Erreur par Chebychev' : np.abs(true_value-Chebyshev_approx)
    }
    results["Nombre de noeuds"] = n


for k in range(1,4):
    print(results_legendre[str(k)])
    print(results[str(k)])
