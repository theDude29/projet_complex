import random
import queue
import copy
import math

def chargerGraphe(chemin):
	"""
	Charge un graphe depuis un fichier texte.

	Format attendu :
	- la ligne 2 contient le nombre de sommets
	- ensuite une section listant les labels des sommets
	- puis le nombre d'arêtes et les arêtes sous la forme "u v"

	Paramètres:
		chemin (str): chemin vers le fichier du graphe

	Retourne:
		(sommets, liste_adjacence)
		sommets : liste d'entiers correspondant aux labels
		liste_adjacence : liste de listes, où liste_adjacence[i] contient les voisins du sommet sommets[i]

	Remarques:
		La fonction ne valide pas exhaustivement le format ; elle suit le format attendu
		par le code original. En cas de format différent, des erreurs peuvent survenir.
	"""

	fichier = open(chemin)
	lignes = fichier.readlines()

	nb_sommets = int(lignes[1])
	sommets = []
	for i in range(3, 3+nb_sommets):
		sommets.append(int(lignes[i]))

	nb_aretes = int(lignes[3+nb_sommets+1])
	liste_adjacence = [[] for _ in range(nb_sommets)]
	for i in range(3+nb_sommets+3, len(lignes)):
		parts = lignes[i].split()
		if len(parts) < 2:
			continue
		u, v = map(int, parts[:2])
		liste_adjacence[u].append(v)
		liste_adjacence[v].append(u)

	return sommets, liste_adjacence


# --------------- Partie 2

# 2.1
def supprimer_sommets(sommets, liste_adjacence, sommets_a_supprimer):

	"""
	Supprime un ensemble de sommets (labels) du graphe et reconstruit
	les structures `sommets` et `liste_adjacence` correspondantes.

	Paramètres:
		sommets : liste des labels actuels
		liste_adjacence : liste de listes des voisins (alignée avec `sommets`)
		sommets_a_supprimer : iterable des labels à supprimer

	Retourne:
		(new_sommets, new_liste_adjacence)

	La reconstruction évite les suppressions in-place qui décaleraient
	les indices et provoqueraient des incohérences.
	"""

	# copie superficielle des sommets et copie profonde des listes d'adjacence
	sommets = copy.copy(sommets)
	liste_adjacence = copy.deepcopy(liste_adjacence)

	to_remove = set(sommets_a_supprimer)
	label_to_index = {label: idx for idx, label in enumerate(sommets)}

	new_sommets = [s for s in sommets if s not in to_remove]
	new_liste_adjacence = []

	for s in new_sommets:
		old_idx = label_to_index[s]
		old_neighbors = liste_adjacence[old_idx]
		
		new_neighbors = [v for v in old_neighbors if v not in to_remove]
		new_liste_adjacence.append(new_neighbors)

	return new_sommets, new_liste_adjacence

def supprimer_sommet(sommets, liste_adjacence, sommet):
	"""Supprime un seul sommet en réutilisant supprimer_sommets.

	Renvoie le couple (new_sommets, new_liste_adjacence).
	"""
	return supprimer_sommets(sommets, liste_adjacence, [sommet])

def get_degres(sommets, liste_adjacence):
	"""
	Calcule le degré de chaque sommet.

	Retourne une liste de degrés dans le même ordre que `sommets`.
	"""

	degres = {}
	for s in sommets:
		degres[s]=0

	for voisins in liste_adjacence:
		for voisin in voisins:
			degres[voisin] += 1

	return [degres[s] for s in sommets]

def get_sommet_degre_maximum(sommets, liste_adjacence):
	"""
	Retourne le label du sommet ayant le degré maximum.

	Renvoie None si la liste des degrés est vide.
	"""

	degres = get_degres(sommets, liste_adjacence)
	if len(degres) == 0:
		return None

	return sommets[degres.index(max(degres))]

#2.2
def graphe_aleatoire(n, p):
	"""
	Génère un graphe aléatoire G(n, p) non orienté simple.

	Paramètres:
		n (int) : nombre de sommets (labels 0..n-1)
		p (float) : probabilité d'existence d'une arête entre deux sommets

	Retourne:
		sommets, liste_adjacence
	"""

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

def algo_couplage(sommets, liste_adjacence):
	"""
	Construit un couplage
	et renvoie la liste des sommets des arêtes du couplage.

	Paramètres:
		sommets, liste_adjacence : représentation du graphe

	Retourne:
		C : liste de paires (u, v) représentant les arêtes du couplage
	"""

	S = set()  # on garde en mémoire les sommets ajoutés
	for i in range(len(liste_adjacence)):
		for v in liste_adjacence[i]:
			if sommets[i] not in S and v not in S:
				S.add(sommets[i])
				S.add(v)
	return S

def nb_aretes(liste_adjacence):
	"""
	Compte le nombre d'arêtes dans un graphe non orienté donné par liste_adjacence.

	Retourne un entier correspondant au nombre d'arêtes unique (chaque arête est comptée deux fois
	dans les listes de voisins, d'où la division par 2).
	"""

	n = 0
	for l in liste_adjacence:
		n += len(l)
	return n//2

def algo_glouton(sommets, liste_adjacence):
	"""
	Algorithme glouton simple pour la couverture de sommets.

	À chaque itération, on sélectionne un sommet de degré maximum et on l'ajoute
	à la couverture, puis on retire toutes les arêtes incidentes.

	Paramètres:
		sommets, liste_adjacence : représentation du graphe

	Retourne:
		C : liste des labels des sommets choisis dans la couverture
	"""

	C = []

	# travailler sur une copie pour ne pas modifier l'original
	liste_adjacence = copy.deepcopy(liste_adjacence)

	# mapping pour accéder rapidement à l'indice d'un label
	label_to_index = {label: idx for idx, label in enumerate(sommets)}

	def supprimer_aretes(v):
		# si v n'est pas dans le mapping, rien à faire
		if v in label_to_index:
			v_idx = label_to_index[v]
			# supprimer toutes les arêtes qui partent de v
			liste_adjacence[v_idx] = []
		# supprimer v des listes de voisins des autres sommets
		for i in range(len(liste_adjacence)):
			if i == v_idx:
				continue
			liste_adjacence[i] = [e for e in liste_adjacence[i] if e != v]

	while nb_aretes(liste_adjacence) > 0:
		v = get_sommet_degre_maximum(sommets, liste_adjacence)
		C.append(v)
		supprimer_aretes(v)

	return C


# --------------- Partie 4

#4.1
def branchement(sommets, liste_adjacence, retourner_nombre_noeuds=False):
	"""
	Recherche exhaustive (branchement) pour la couverture de sommets minimale.

	Algorithme DFS itératif : à chaque étape, on choisit un sommet de degré
	positif et on branche sur le cas où on le prend ou sur le cas où on prend
	un de ses voisins. On conserve la meilleure couverture trouvée.

	Paramètres:
		sommets, liste_adjacence : représentation du graphe
		retourner_nombre_noeuds (bool) : si True, la fonction renvoie aussi
			le nombre de nœuds explorés

	Retourne:
		couverture_minimum (ou (couverture_minimum, nombre_noeuds) si demandé)
	"""

	nombre_noeuds = 0

	couverture_minimum = sommets

	q = queue.LifoQueue()
	q.put((sommets, liste_adjacence, []))

	while not q.empty():

		nombre_noeuds += 1

		sommets, liste_adjacence, couverture = q.get()
		# mapping label -> index pour accès O(1)
		label_to_index = {label: idx for idx, label in enumerate(sommets)}

		if sum(get_degres(sommets, liste_adjacence)) != 0:
			i = get_sommet_degre_maximum(sommets, liste_adjacence)
			# prendre un voisin quelconque
			neighs = liste_adjacence[label_to_index[i]]
			j = neighs[0]

			# on ajoute dans la pile le cas où on prend le sommet i
			new_sommets, new_liste_adjacence = supprimer_sommet(sommets, liste_adjacence, i)
			q.put((new_sommets, new_liste_adjacence, couverture + [i]))

			# puis celui où on prend le sommet j
			new_sommets, new_liste_adjacence = supprimer_sommet(sommets, liste_adjacence, j)
			q.put((new_sommets, new_liste_adjacence, couverture + [j]))

		else:
			if len(couverture) < len(couverture_minimum):
				couverture_minimum = couverture

	if not retourner_nombre_noeuds:
		return couverture_minimum
	else:
		return couverture_minimum, nombre_noeuds

#4.2
def branchement_borne(sommets, liste_adjacence, retourner_nombre_noeuds=False):
	"""
	Branchement avec bornes inférieures pour couper des sous-arbres.

	La fonction calcule plusieurs bornes inférieures (b1, b2, b3) et n'explore
	que les états dont la borne est strictement inférieure à la taille de la meilleure
	couverture trouvée.

	Paramètres:
		sommets, liste_adjacence : graphe
		retourner_nombre_noeuds : si True, renvoyer aussi le nombre de nœuds explorés

	Retourne:
		couverture_minimum (ou (couverture_minimum, nombre_noeuds))
	"""

	nombre_noeuds=0

	couverture_minimum = sommets

	q = queue.LifoQueue() # on utilise une pile dont chaque élément est un triplet (sommets, liste_adjacence, couverture)
	q.put((sommets, liste_adjacence, []))

	while not q.empty():

		nombre_noeuds += 1

		sommets, liste_adjacence, couverture = q.get()
		# mapping label -> index
		label_to_index = {label: idx for idx, label in enumerate(sommets)}

		degres = get_degres(sommets, liste_adjacence)
		m = nb_aretes(liste_adjacence)

		if max(degres) > 0:
			b1 = math.ceil(m / max(degres)) + len(couverture)
		else:
			b1 = 0
		b2 = len(algo_couplage(sommets, liste_adjacence))/2 + len(couverture) #on divise par 2 car on considère ici le nombre d'aretes du couplage
		b3 = (2*len(sommets) - 1 - math.sqrt((2*len(sommets) - 1)**2 - 8*nb_aretes(liste_adjacence))) / 2

		borne_inf = max([b1, b2, b3])

		if borne_inf < len(couverture_minimum):
			if sum(degres) != 0:
				i = get_sommet_degre_maximum(sommets, liste_adjacence)
				j = liste_adjacence[label_to_index[i]][0]

				new_sommets, new_liste_adjacence = supprimer_sommet(sommets, liste_adjacence, i)
				q.put((new_sommets, new_liste_adjacence, couverture + [i]))

				new_sommets, new_liste_adjacence = supprimer_sommet(sommets, liste_adjacence, j)
				q.put((new_sommets, new_liste_adjacence, couverture + [j]))

			else:
				if len(couverture) < len(couverture_minimum):
					couverture_minimum = couverture

	if not retourner_nombre_noeuds:
		return couverture_minimum
	else:
		return couverture_minimum, nombre_noeuds
		

def branchement_borne2(sommets, liste_adjacence, retourner_nombre_noeuds=False):
	"""
	Branchement avec heuristique: on compare l'effet local d'explorer chaque branche
	en utilisant l'algorithme glouton pour ordonner les branches.

	La fonction évalue pour i et j la taille de la couverture gloutonne après la suppression
	(pour estimer quelle branche est la plus prometteuse) et pousse d'abord la moins
	prometteuse pour gagner en efficacité de parcours.

	Remarque: la fonction contient plusieurs bornes et heuristiques combinées.
	"""

	nombre_noeuds = 0

	couverture_minimum = sommets

	q = queue.LifoQueue()
	q.put((sommets, liste_adjacence, []))

	while not q.empty():

		nombre_noeuds += 1

		sommets, liste_adjacence, couverture = q.get()
		# mapping label -> index
		label_to_index = {label: idx for idx, label in enumerate(sommets)}

		degres = get_degres(sommets, liste_adjacence)
		m = nb_aretes(liste_adjacence)

		if max(degres) > 0:
			b1 = math.ceil(m / max(degres)) + len(couverture)
		else:
			b1 = 0
		b2 = len(algo_couplage(sommets, liste_adjacence))/2 + len(couverture) #on divise par 2 car on considère ici le nombre d'aretes du couplage
		b3 = (2*len(sommets) - 1 - math.sqrt((2*len(sommets) - 1)**2 - 8*nb_aretes(liste_adjacence))) / 2

		borne_inf = max([b1, b2, b3])

		if borne_inf < len(couverture_minimum):
			if sum(degres) != 0:
				i = get_sommet_degre_maximum(sommets, liste_adjacence)
				j = liste_adjacence[label_to_index[i]][0]

				new_sommets1, new_liste_adjacence1 = supprimer_sommet(sommets, liste_adjacence, i)
				c1 = len(algo_glouton(new_sommets1, new_liste_adjacence1))

				new_sommets2, new_liste_adjacence2 = supprimer_sommet(sommets, liste_adjacence, j)
				c2 = len(algo_glouton(new_sommets2, new_liste_adjacence2))

				# pousser d'abord la branche avec la gloutonne la plus grande (moins prometteuse),
				# de sorte que la plus prometteuse soit traitée en premier (pile LIFO)
				if c1 < c2:
					q.put((new_sommets2, new_liste_adjacence2, couverture + [j]))
					q.put((new_sommets1, new_liste_adjacence1, couverture + [i]))
				else:
					q.put((new_sommets1, new_liste_adjacence1, couverture + [i]))
					q.put((new_sommets2, new_liste_adjacence2, couverture + [j]))

			else:
				if len(couverture) < len(couverture_minimum):
					couverture_minimum = couverture

	if not retourner_nombre_noeuds:
		return couverture_minimum
	else:
		return couverture_minimum, nombre_noeuds
	
def branchement_borne3(sommets, liste_adjacence, retourner_nombre_noeuds=False):
	"""
	Branchement avec heuristique: on compare l'effet local d'explorer chaque branche
	en utilisant l'algorithme glouton pour ordonner les branches.

	La fonction évalue pour i et j la taille de la couverture gloutonne après la suppression
	(pour estimer quelle branche est la plus prometteuse) et pousse d'abord la moins
	prometteuse pour gagner en efficacité de parcours.

	Remarque : on n'utilise pas les bornes min ici
	"""

	nombre_noeuds = 0

	couverture_minimum = sommets

	q = queue.LifoQueue()
	q.put((sommets, liste_adjacence, []))

	while not q.empty():

		nombre_noeuds += 1

		sommets, liste_adjacence, couverture = q.get()
		# mapping label -> index
		label_to_index = {label: idx for idx, label in enumerate(sommets)}

		degres = get_degres(sommets, liste_adjacence)

		if len(couverture) < len(couverture_minimum):
			if sum(degres) != 0:
				i = get_sommet_degre_maximum(sommets, liste_adjacence)
				j = liste_adjacence[label_to_index[i]][0]

				new_sommets1, new_liste_adjacence1 = supprimer_sommet(sommets, liste_adjacence, i)
				c1 = len(algo_glouton(new_sommets1, new_liste_adjacence1))

				new_sommets2, new_liste_adjacence2 = supprimer_sommet(sommets, liste_adjacence, j)
				c2 = len(algo_glouton(new_sommets2, new_liste_adjacence2))

				# pousser d'abord la branche avec la gloutonne la plus grande (moins prometteuse),
				# de sorte que la plus prometteuse soit traitée en premier (pile LIFO)
				if c1 < c2:
					q.put((new_sommets2, new_liste_adjacence2, couverture + [j]))
					q.put((new_sommets1, new_liste_adjacence1, couverture + [i]))
				else:
					q.put((new_sommets1, new_liste_adjacence1, couverture + [i]))
					q.put((new_sommets2, new_liste_adjacence2, couverture + [j]))

			else:
				if len(couverture) < len(couverture_minimum):
					couverture_minimum = couverture

	if not retourner_nombre_noeuds:
		return couverture_minimum
	else:
		return couverture_minimum, nombre_noeuds


#4.3
def branchement_ameliore(sommets, liste_adjacence, retourner_nombre_noeuds=False):
	"""
	Amélioration du branchement : lorsqu'on décide de prendre un sommet i,
	une autre branche envisage de prendre tous les voisins de i en une seule fois.

	Cette stratégie exploite des réductions structurelles simples pour couper
	le nombre d'états explorés.

	Paramètres et retour : identiques à `branchement`.
	"""

	nombre_noeuds = 0

	couverture_minimum = sommets

	q = queue.LifoQueue()
	q.put((sommets, liste_adjacence, []))

	while not q.empty():

		nombre_noeuds += 1

		sommets, liste_adjacence, couverture = q.get()
		# mapping label -> index pour accès O(1)
		label_to_index = {label: idx for idx, label in enumerate(sommets)}
		
		if sum(get_degres(sommets, liste_adjacence)) != 0:
			i = get_sommet_degre_maximum(sommets, liste_adjacence)
			j = liste_adjacence[label_to_index[i]][0]
			
			# brancher sur prendre i
			new_sommets, new_liste_adjacence = supprimer_sommet(sommets, liste_adjacence, i)
			q.put((new_sommets, new_liste_adjacence, couverture + [i]))
			
			# brancher sur prendre tous les voisins de i (réduction)
			neighs = liste_adjacence[label_to_index[i]]
			new_sommets, new_liste_adjacence = supprimer_sommets(sommets, liste_adjacence, neighs) # supprimer tous les voisins de i
			q.put((new_sommets, new_liste_adjacence, couverture + neighs))
			
		else:
			if len(couverture) < len(couverture_minimum):
				couverture_minimum = couverture

	if not retourner_nombre_noeuds:
		return couverture_minimum
	else:
		return couverture_minimum, nombre_noeuds
	
def branchement_ameliore2(sommets, liste_adjacence, retourner_nombre_noeuds=False):
	"""
	Amélioration du branchement : lorsqu'on décide de prendre un sommet i,
	une autre branche envisage de prendre tous les voisins de i en une seule fois.

	Cette stratégie exploite des réductions structurelles simples pour couper
	le nombre d'états explorés.

	Paramètres et retour : identiques à `branchement`.
	"""

	nombre_noeuds = 0

	couverture_minimum = sommets

	q = queue.LifoQueue()
	q.put((sommets, liste_adjacence, []))

	while not q.empty():

		nombre_noeuds += 1

		sommets, liste_adjacence, couverture = q.get()
		degres = get_degres(sommets, liste_adjacence)

		# on prend systématiquement les voisins de sommets de degré 1 sans brancher
		sommet_voisin_deg_1=-1
		for i in range(len(degres)):
			if degres[i] == 1:
				sommet_voisin_deg_1 = liste_adjacence[i][0]
				break
		
		while(sommet_voisin_deg_1 != -1):
			# ajouter le voisin à la couverture puis réduire le graphe
			couverture += [sommet_voisin_deg_1]
			sommets, liste_adjacence = supprimer_sommet(sommets, liste_adjacence, sommet_voisin_deg_1)
			degres = get_degres(sommets, liste_adjacence)
			
			sommet_voisin_deg_1=-1
			for i in range(len(degres)):
				if degres[i] == 1:
					sommet_voisin_deg_1 = liste_adjacence[i][0]
					break

		label_to_index = {label: idx for idx, label in enumerate(sommets)}

		if sum(degres) != 0:
			i = get_sommet_degre_maximum(sommets, liste_adjacence)

			j = liste_adjacence[label_to_index[i]][0]
			
			# brancher sur prendre i
			new_sommets, new_liste_adjacence = supprimer_sommet(sommets, liste_adjacence, i)
			q.put((new_sommets, new_liste_adjacence, couverture + [i]))
			
			# brancher sur prendre tous les voisins de i (réduction)
			neighs = liste_adjacence[label_to_index[i]]
			new_sommets, new_liste_adjacence = supprimer_sommets(sommets, liste_adjacence, neighs) # supprimer tous les voisins de i
			q.put((new_sommets, new_liste_adjacence, couverture + neighs))
			
		else:
			if len(couverture) < len(couverture_minimum):
				couverture_minimum = couverture

	if not retourner_nombre_noeuds:
		return couverture_minimum
	else:
		return couverture_minimum, nombre_noeuds


#4.4.2
# Couplage + petit nettoyage pour enlever les sommets inutiles

def est_couverture(sommets, liste_adjacence, C):
	""" Vérifie si C couvre toutes les arêtes """
	C = set(C)
	label_to_index = {label: idx for idx, label in enumerate(sommets)}
	for i in range(len(sommets)):
		for v in liste_adjacence[i]:
			if (sommets[i] not in C) and (v not in C):
				return False
	return True

def post_traitement_1opt(sommets, liste_adjacence, C):
	"""
	Tant qu'on peut enlever un sommet sans casser la couverture, on le fait.
	C'est un petit 1-opt de suppression.
	"""
	C = set(C)
	label_to_index = {label: idx for idx, label in enumerate(sommets)}
	change = True
	while change:
		change = False
		for v in list(C):
			i = label_to_index.get(v)
			if i is None:
				continue
			voisins = liste_adjacence[i]
			ok = True
			for u in voisins:
				if u not in C:
					ok = False
					break
			if ok:
				C.remove(v)
				change = True
	return list(C)

def algo_couplage_plus(sommets, liste_adjacence):
	"""
	Heuristique 4.4.2 :
	on fait le couplage et ensuite on essaie d'enlever les sommets redondants.
	"""
	C_init = list(algo_couplage(sommets, liste_adjacence))
	C_final = post_traitement_1opt(sommets, liste_adjacence, C_init)
	if not est_couverture(sommets, liste_adjacence, C_final):
		# au cas où on a retiré trop, on garde l'ancien
		return C_init
	return C_final
