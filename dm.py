import random
import queue
import copy
import time
import matplotlib.pyplot as plt

def chargerGraphe(chemin):
	fichier = open(chemin)
	lignes = fichier.readlines()

	nb_sommets = int(lignes[1])
	sommets = []
	for i in range(3, 3+nb_sommets):
		sommets.append(int(lignes[i]))

	nb_aretes = int(lignes[3+nb_sommets+1])
	liste_adjacence = [[] for _ in range(nb_sommets)]
	for i in range(3+nb_sommets+3, len(lignes)):
		u,v = int(lignes[i].split())
		liste_adjacence[u].append(v)
		liste_adjacence[v].append(u)
	return sommets, liste_adjacence

# --------------- Partie 2

# 2.1
def supprimer_sommets(sommets, liste_adjacence, sommets_a_supprimer):

	sommets = copy.copy(sommets) #on fait des copies pour ne pas faire de modification en place
	liste_adjacence = copy.deepcopy(liste_adjacence)

	for sommet_a_supprimer in sommets_a_supprimer:
		if sommet_a_supprimer in sommets:
			liste_adjacence.pop(sommets.index(sommet_a_supprimer))
			sommets.pop(sommets.index(sommet_a_supprimer))
			
		for voisins in liste_adjacence:
			if sommet_a_supprimer in voisins:
				voisins.pop(voisins.index(sommet_a_supprimer))
			
	return sommets, liste_adjacence

def supprimer_sommet(sommets, liste_adjacence, sommet):
	return supprimer_sommets(sommets, liste_adjacence, [sommet])

def get_degres(sommets, liste_adjacence):

	degres = {}
	for s in sommets:
		degres[s]=0

	for voisins in liste_adjacence:
		for voisin in voisins:
			degres[voisin] += 1

	return [degres[s] for s in sommets]

def get_sommet_degre_maximum(sommets, liste_adjacence):
	degres = get_degres(sommets, liste_adjacence)
	if len(degres)==0:
		return "pas de sommets !"
	else:
		return sommets[degres.index(max(degres))]

#2.2
def graphe_aleatoire(n, p):
	sommets = [i for i in range(n)]
	liste_adjacence = [[] for _ in range(n)]
	for i in range(n):
		for j in range(i):
			t = random.random()
			if t < p:
				liste_adjacence[i].append(j)
				liste_adjacence[j].append(i)
	return sommets, liste_adjacence



# --------------- Partie 3

def test(f, n, p): #fonction pour afficher les temps de calcul pour des graphes de taille i pour i entre 1 et n en utilisant la fonction f pour trouver une couverture minimale
	temps = []
	for i in range(1, n):
	
		print(i)
		
		somme_temps=0
		for _ in range(10):
			sommets, liste_adjacence = graphe_aleatoire(i,p)
			debut = time.time()
			f(sommets, liste_adjacence)
			somme_temps += time.time()-debut
			
		temps.append(somme_temps/10)
		
	plt.plot([i for i in range(1,n)], temps, label=f.__name__ + " avec p=" + str(p))

def algo_couplage(sommets, liste_adjacence):
	C = []
	for i in range(len(liste_adjacence)):
		for v in liste_adjacence[i]:
			if sommets[i] not in C and v not in C:
				C += [sommets[i], v]
	return C

def algo_glouton(sommets, liste_adjacence):
	C = []
	
	liste_adjacence = copy.deepcopy(liste_adjacence)
	
	def nb_aretes(liste_adjacence):
		n = 0
		for l in liste_adjacence:
			n += len(l)
		return n//2
		
	def supprimer_aretes(v):
		for i in range(len(liste_adjacence)):
			if i == sommets.index(v):
				liste_adjacence[i] = []
			else:
				new_voisins = [e for e in liste_adjacence[i] if e != v]
				liste_adjacence[i] = new_voisins
		
	while nb_aretes(liste_adjacence) > 0:
		v = get_sommet_degre_maximum(sommets, liste_adjacence)
		C.append(v)
		supprimer_aretes(v)
		
	return C

test(algo_glouton, 100, 0.2)
test(algo_glouton, 100, 0.5)
test(algo_glouton, 100, 0.7)

plt.xlabel("n")
plt.ylabel("temps")
plt.legend()
plt.show()

# --------------- Partie 4
	
def test_nb_noeuds(f, n, p): #fonction pour afficher le nombre de noeuds visites pour des graphes de taille i pour i entre 1 et n en utilisant la fonction f pour trouver une couverture minimale
	nb_noeuds = []
	for i in range(1, n):
	
		print(i)
	
		sommets, liste_adjacence = graphe_aleatoire(i,p)
		
		nb_noeuds.append(f(sommets, liste_adjacence, True)[1])
		
	plt.plot([i for i in range(1,n)], nb_noeuds, label=f.__name__ + " avec p=" + str(p))

#4.1
def branchement(sommets, liste_adjacence, retourner_nombre_noeuds=False):

	nombre_noeuds=0

	couverture_minimum = sommets
	
	q = queue.LifoQueue() #on utilise une pile dont chaque element est un triplé sommets, liste_adjacence, couverture decrivant l'etat du noeud en cours
	q.put((sommets, liste_adjacence, []))

	while not q.empty():
	
		nombre_noeuds += 1
	
		sommets, liste_adjacence, couverture = q.get()
		
		if sum(get_degres(sommets, liste_adjacence)) != 0: # si jamais on a pas encore trouvé de couverture
			i = get_sommet_degre_maximum(sommets, liste_adjacence) #on prend un sommet de degre max != 0
			j = liste_adjacence[sommets.index(i)][0] #on prend un de ses voisins
			
			#on ajoute dans le tas le cas ou on prend le sommet i
			new_sommets, new_liste_adjacence = supprimer_sommet(sommets, liste_adjacence, i)
			q.put((new_sommets, new_liste_adjacence, couverture + [i]))
			
			#puis celui ou on prend le sommet j
			new_sommets, new_liste_adjacence = supprimer_sommet(sommets, liste_adjacence, j)
			q.put((new_sommets, new_liste_adjacence, couverture + [j]))
			
		else: #si on a une couverture et qu'elle est de taille inferieure à la meilleure couverture rencontrée jusque maintenant on met à jour la couverture minimale
			if len(couverture) < len(couverture_minimum):
				couverture_minimum = couverture
	
	if not retourner_nombre_noeuds:
		return couverture
	else:
		return couverture, nombre_noeuds
	

#4.2
def branchement_ameliore(sommets, liste_adjacence, retourner_nombre_noeuds=False):

	nombre_noeuds = 0

	couverture_minimum = sommets
	
	q = queue.LifoQueue()
	q.put((sommets, liste_adjacence, []))

	while not q.empty():
	
		nombre_noeuds += 1
	
		sommets, liste_adjacence, couverture = q.get()
		
		if sum(get_degres(sommets, liste_adjacence)) != 0:
			i = get_sommet_degre_maximum(sommets, liste_adjacence)
			
			j = liste_adjacence[sommets.index(i)][0]
			
			new_sommets, new_liste_adjacence = supprimer_sommet(sommets, liste_adjacence, i)
			q.put((new_sommets, new_liste_adjacence, couverture + [i]))
			
			new_sommets, new_liste_adjacence = supprimer_sommets(sommets, liste_adjacence, liste_adjacence[sommets.index(i)]) #c'est le même programme que precedemment mais on supprime tout les voisins de i à cette etape et on les ajoute tous dans la couverture en cours
			q.put((new_sommets, new_liste_adjacence, couverture + liste_adjacence[sommets.index(i)]))
			
		else:
			if len(couverture) < len(couverture_minimum):
				couverture_minimum = couverture
	
	if not retourner_nombre_noeuds:
		return couverture
	else:
		return couverture, nombre_noeuds
	
def branchement_ameliore2(sommets, liste_adjacence, retourner_nombre_noeuds=False):

	nombre_noeuds=0

	couverture_minimum = sommets
	
	q = queue.LifoQueue()
	q.put((sommets, liste_adjacence, []))

	while not q.empty():
	
		nombre_noeuds+=1
		
		sommets, liste_adjacence, couverture = q.get()
		degres = get_degres(sommets, liste_adjacence)
		
		# on rajoute une partie permettant de prendre touts les sommets voisins d'un sommets de degre 1 sans branchements tant qu'il en existe
		sommets_voisin_deg_1=[]
		for i in range(len(degres)):
			if degres[i] == 1:
				sommets_voisin_deg_1.append(liste_adjacence[i][0])
				
		
		while(len(sommets_voisin_deg_1)>0):
			
		
			couverture += sommets_voisin_deg_1
		
			sommets, liste_adjacence = supprimer_sommets(sommets, liste_adjacence, sommets_voisin_deg_1)
			degres = get_degres(sommets, liste_adjacence)
			
			sommets_voisin_deg_1=[]
			for i in range(len(degres)):
				if degres[i] == 1:
					sommets_voisin_deg_1.append(liste_adjacence[i][0])
		
		#ensuite on repart sur la methode de base
		
		if sum(degres) != 0:
			i = get_sommet_degre_maximum(sommets, liste_adjacence)
			j = liste_adjacence[sommets.index(i)][0]
			
			new_sommets, new_liste_adjacence = supprimer_sommet(sommets, liste_adjacence, i)
			q.put((new_sommets, new_liste_adjacence, couverture + [i]))
			
			new_sommets, new_liste_adjacence = supprimer_sommets(sommets, liste_adjacence, liste_adjacence[sommets.index(i)])
			q.put((new_sommets, new_liste_adjacence, couverture + liste_adjacence[sommets.index(i)]))
			
		else:
			if len(couverture) < len(couverture_minimum):
				couverture_minimum = couverture
				
	if retourner_nombre_noeuds:
		return couverture, nombre_noeuds
	else:
		return couverture

#4.4
