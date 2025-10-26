import networkx as nx
import matplotlib.pyplot as plt
import time
import dm

def afficherGraphe(sommets, liste_adjacence):
	"""
	Affiche le graphe en utilisant networkx et matplotlib.

	Paramètres:
		sommets : liste des labels des sommets
		liste_adjacence : liste de listes représentant les voisins par indice

	La fonction construit la liste des arêtes sans doublons puis dessine le graphe.
	"""

	G = nx.Graph()
	G.add_nodes_from(sommets)

	aretes = []
	for i in range(len(liste_adjacence)):
		u = sommets[i]
		for v in liste_adjacence[i]:
			# on évite d'ajouter deux fois la même arête (u,v) et (v,u)
			if (u, v) not in aretes and (v, u) not in aretes:
				aretes.append((u, v))
	G.add_edges_from(aretes)

	nx.draw(G, with_labels=True)
	plt.show()
	
def test_temps(f, n, p):
	"""
	Trace le temps moyen d'exécution de la fonction f pour des graphes de tailles 1..n-1.

	Paramètres:
		f : fonction qui calcule une couverture sur (sommets, liste_adjacence)
		n (int) : taille maximale testée
		p (float) : probabilité pour la génération aléatoire

	La mesure utilise 10 répétitions par taille et trace la courbe à la fin.
	"""

	temps = []
	for i in range(1, n):

		print(i)

		somme_temps=0
		for _ in range(10):
			sommets, liste_adjacence = dm.graphe_aleatoire(i,p)
			debut = time.time()
			f(sommets, liste_adjacence)
			somme_temps += time.time()-debut
			
		temps.append(somme_temps/10)
		
	plt.plot([i for i in range(1,n)], temps, label=f.__name__ + " avec p=" + str(p))
	plt.xlabel("Taille du graphe")
	plt.ylabel("Temps")
	
def test_nb_noeuds(f, n, p):
	"""
	Mesure le nombre moyen de nœuds visités par un algorithme de recherche exacte
	(par ex. branchement) sur des graphes aléatoires de tailles 1..n-1.

	Paramètres:
		f : fonction de recherche exacte retournant (couverture, nb_noeuds) quand le drapeau True est passé
		n, p : paramètres de génération des graphes

	La mesure utilise 10 répétitions par taille et trace la courbe.
	"""

	nb_noeuds = []
	for i in range(1, n):
    
		print("generation du test pour i=", i)
        
		somme=0
		for _ in range(10):
			sommets, liste_adjacence = dm.graphe_aleatoire(i,p)
			somme += f(sommets, liste_adjacence, True)[1]
		nb_noeuds.append(somme/10)
        
	plt.plot([i for i in range(1,n)], nb_noeuds, label=f.__name__ + " avec p=" + str(p))
	plt.xlabel("Taille du graphe")
	plt.ylabel("Nombre de nœuds visités")
	
def graphiques(f, n):
	test_temps(f, n, 0.2)
	test_temps(f, n, 0.5)
	test_temps(f, n, 0.7)
	plt.legend()
	plt.show()
	
	test_nb_noeuds(f, n, 0.2)
	test_nb_noeuds(f, n, 0.5)
	test_nb_noeuds(f, n, 0.7)
	plt.legend()
	plt.show()
	
def test_taille_couverture(f1, f2, f3, n, p):
	"""
	Compare les tailles de couverture retournées par trois algorithmes sur des graphes aléatoires.

	Paramètres:
		f1, f2, f3 : fonctions prenant (sommets, liste_adjacence) et renvoyant une couverture
		n : taille maximale des graphes testés
		p : probabilité utilisée pour la génération aléatoire

	La fonction trace ensuite les trois courbes de tailles.
	"""

	tailles_couplages1 = []
	tailles_couplages2 = []
	tailles_couplages3 = []
	for i in range(1, n):
    
		print(i)
        
		sommets, liste_adjacence = dm.graphe_aleatoire(i,p)
        
		c1,c2,c3 = f1(sommets, liste_adjacence), f2(sommets, liste_adjacence), f3(sommets, liste_adjacence)
		tailles_couplages1.append(len(c1))
		tailles_couplages2.append(len(c2))
		tailles_couplages3.append(len(c3))
        
	plt.plot([i for i in range(1,n)], tailles_couplages1, label=f1.__name__ + " avec p=" + str(p))
	plt.plot([i for i in range(1,n)], tailles_couplages2, label=f2.__name__ + " avec p=" + str(p))
	plt.plot([i for i in range(1,n)], tailles_couplages3, label=f3.__name__ + " avec p=" + str(p))
	plt.xlabel("Taille du graphe")
	plt.ylabel("Taille de la couverture")

def evaluer_rapport_approximation(f, approx, n):
	"""
	Évalue le rapport d'approximation moyen entre un algorithme exact f
	et une heuristique approx sur des graphes aléatoires de tailles 10..n-1.

	Pour chaque taille i et chaque probabilité p ∈ {0.2, 0.5, 0.7}, on calcule
	la moyenne de |f|/|approx| sur 10 instances et on trace les courbes.
	"""

	r1 = []
	r2 = []
	r3 = []
	for i in range(10,n):
        
		print(i)
    
		n1,n2,n3=0,0,0
		for _ in range(10):
			s1,l1 = dm.graphe_aleatoire(i,0.2)
			s2,l2 = dm.graphe_aleatoire(i,0.5)
			s3,l3 = dm.graphe_aleatoire(i,0.7)
            
			n1+=len(approx(s1,l1))/len(f(s1,l1))
			n2+=len(approx(s2,l2))/len(f(s2,l2))
			n3+=len(approx(s3,l3))/len(f(s3,l3))
            
		r1.append(n1/10)
		r2.append(n2/10)
		r3.append(n3/10)
    
	xs = [i for i in range(10,n)]
    
	plt.plot(xs, r1, label = f.__name__ + " vs " + approx.__name__ + " avec p=0.2")
	plt.plot(xs, r2, label = f.__name__ + " vs " + approx.__name__ + " avec p=0.5")
	plt.plot(xs, r3, label = f.__name__ + " vs " + approx.__name__ + " avec p=0.7")
	plt.xlabel("Taille du graphe")
	plt.ylabel("Rapport d'approximation")

#q3.1
s,l = dm.chargerGraphe("graphe_q31.txt")
afficherGraphe(s,l)


#q3.2
test_taille_couverture(dm.algo_glouton, dm.algo_couplage, dm.branchement_ameliore2, 20, 0.7)
plt.legend()
plt.show()


test_temps(dm.algo_couplage, 100, 0.7)
test_temps(dm.algo_glouton, 100, 0.7)
plt.legend()
plt.show()

#q4.1.2
#graphiques(dm.branchement, 20)

#q4.2.2
#graphiques(dm.branchement_borne, 20)

#q4.2.3
graphiques(dm.branchement_borne2, 20)
graphiques(dm.branchement_borne3, 20)

#4.3.1 et 4.3.2
graphiques(dm.branchement_ameliore, 20)


#4.3.3
graphiques(dm.branchement_ameliore2, 20)

#q4.4.1
evaluer_rapport_approximation(dm.branchement_ameliore2, dm.algo_glouton, 50)
plt.legend()
plt.show()

evaluer_rapport_approximation(dm.branchement_ameliore2, dm.algo_couplage, 50)
plt.legend()
plt.show()