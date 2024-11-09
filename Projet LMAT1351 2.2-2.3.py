import numpy as np
import matplotlib.pyplot as plt

def quad_spline(P, control_point):
    """
    Génère une spline quadratique passant par tous les points de contrôle P
    en respectant la continuité C1.
    
    Arguments:
    - P : liste des points de contrôle [P0, P1, ..., Pn] (liste de tuples/liste)
    - control_point : tuple (x, y) représentant le point de contrôle initial Q0

    Retourne une fonction spline S(t) définie sur [0, 1].
    """
    n = len(P) - 1  # Nombre de segments
    if(n == 0):
        return
    Px0, Py0 = P[0]
    ContX, ContY = control_point
    
    # Liste pour stocker les fonctions de Bézier pour chaque segment
    list_Bézier = []
    
    # Construction du premier segment de Bézier
    Px1, Py1 = P[1]
    list_Bézier.append(lambda t, Px0=Px0, Py0=Py0, ContX=ContX, ContY=ContY, Px1=Px1, Py1=Py1: [
        (1 - t)**2 * Px0 + 2 * t * (1 - t) * ContX + t**2 * Px1,
        (1 - t)**2 * Py0 + 2 * t * (1 - t) * ContY + t**2 * Py1
    ])
    
    # Relier le premier point P0 au premier point de contrôle Cont avec une ligne noire
    plt.plot([Px0, ContX], [Py0, ContY], color="black", linewidth=1, linestyle = "--")
    plt.plot([ContX, Px1], [ContY, Py1], color="black", linewidth=1, linestyle = "--")
    
    # Construction des segments suivants en respectant C1
    for k in range(1, n):
        Pxk, Pyk = P[k]
        Pxk1, Pyk1 = P[k + 1]
        
        # Point de contrôle suivant déterminé par continuité C1
        PrevContX, PrevContY = ContX, ContY
        ContX = 2 * Pxk - PrevContX
        ContY = 2 * Pyk - PrevContY
        
        # Relier Px à Cont avec une ligne noire
        plt.plot([Pxk, ContX], [Pyk, ContY], color="black", linewidth=1,linestyle = "--")
        
        # Relier Cont à Px+1 avec une ligne noire et pointillée
        plt.plot([ContX, Pxk1], [ContY, Pyk1], color="black", linestyle="--", linewidth=1)
        if k == n-1:
            if n == 2:
                plt.scatter(ContX, ContY, color="black", label="Point de contrôle déduit")
            else:
                plt.scatter(ContX, ContY, color="black", label="Points de contrôle déduits")
        else:
            plt.scatter(ContX, ContY, color="black")
        
        list_Bézier.append(lambda t, Pxk=Pxk, Pyk=Pyk, ContX=ContX, ContY=ContY, Pxk1=Pxk1, Pyk1=Pyk1: [
            (1 - t)**2 * Pxk + 2 * t * (1 - t) * ContX + t**2 * Pxk1,
            (1 - t)**2 * Pyk + 2 * t * (1 - t) * ContY + t**2 * Pyk1
        ])
    
    
    # Fonction pour appliquer l'indicatrice et combiner les segments
    def res(t):
        resX, resY = np.zeros_like(t), np.zeros_like(t)
        
        for k, Bézier in enumerate(list_Bézier):
            if(k == 0):
                indic = (t >= 0) & (t <= (k + 1) / n)
            else:
                indic = (t > k / n) & (t <= (k + 1) / n)
            
            segment_t = n * t - k
            tempX, tempY = Bézier(segment_t)
            
            resX += indic * tempX
            resY += indic * tempY
            
        return resX, resY
    
    return res

# Exemple fonctionnant pour le cas n = 3
P = [[-1,0],[1,1],[-1,7]]
control_point = [2,0]
t = np.linspace(0, 1, 10**6)
plt.figure(figsize = (10,6))
spline = quad_spline(P, control_point)
curveX, curveY = spline(t)
plt.plot(curveX, curveY, label='Spline quadratique', color="blue")
plt.scatter(*zip(*P), color='red', label='Nœuds')
# Affichage des points de contrôle déduits en noir
plt.scatter(control_point[0], control_point[1], color="black", label="Point de contrôle initial choisi ($Q_{0}$)", marker="*")
plt.legend()
plt.savefig("2.2_example.svg")
plt.show()


# Exemple pour n général
P = [[0,0],[1,1],[2,0],[3,1],[4,0],[6,-4],[7,8],[6,10],[8,-2],[9,15],[4,20],[-1,3],[0,0]]
control_point = [0.3,-0.2]
t = np.linspace(0, 1, 10**6)
plt.figure(figsize = (10,6))
spline = quad_spline(P, control_point)
curveX, curveY = spline(t)
plt.plot(curveX, curveY, label='Spline quadratique', color="blue")
plt.scatter(*zip(*P), color='red', label='Nœuds')
plt.scatter(control_point[0], control_point[1], color="black", label="Point de contrôle initial choisi ($Q_{0}$)", marker="*")
plt.legend()
plt.savefig("2.3_example.svg")
plt.show()

