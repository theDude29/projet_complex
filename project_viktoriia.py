import random
import time
import math
from collections import defaultdict
import matplotlib.pyplot as plt

# PARTIE 1 : OPÉRATIONS DE BASE SUR LES GRAPHES

class Graphe:
    def __init__(self):
        self.liste_adj = defaultdict(set)
        self.sommets = set()
       
    def ajouter_sommet(self, v):
        self.sommets.add(v)
        if v not in self.liste_adj:
            self.liste_adj[v] = set()
    
    def ajouter_arete(self, u, v):
        self.ajouter_sommet(u)
        self.ajouter_sommet(v)
        self.liste_adj[u].add(v)
        self.liste_adj[v].add(u)
    
    def supprimer_sommet(self, v):
        if v not in self.sommets:
            return
        
        for voisin in list(self.liste_adj[v]):
            self.liste_adj[voisin].discard(v)
        
        del self.liste_adj[v]
        self.sommets.discard(v)
    
    def supprimer_sommets(self, ensemble_sommets):
        for v in ensemble_sommets:
            self.supprimer_sommet(v)

    def degre(self, v):
        return len(self.liste_adj.get(v, set()))
    
    def degres(self):
        return {v: self.degre(v) for v in self.sommets}
    
    def sommet_degre_max(self):
        if not self.sommets:
            return None
        return max(self.sommets, key=lambda v: self.degre(v))
    
    def obtenir_aretes(self):
        aretes = []
        vus = set()
        for u in self.sommets:
            for v in self.liste_adj[u]:
                arete = tuple(sorted([u, v]))
                if arete not in vus:
                    aretes.append((u, v))
                    vus.add(arete)
        return aretes
    
    def copie(self):
        nouveau_graphe = Graphe()
        nouveau_graphe.sommets = self.sommets.copy()
        nouveau_graphe.liste_adj = defaultdict(set)
        for v in self.liste_adj:
            nouveau_graphe.liste_adj[v] = self.liste_adj[v].copy()
        return nouveau_graphe
    
    def nb_sommets(self):
        return len(self.sommets)
    
    def nb_aretes(self):
        return len(self.obtenir_aretes())

def lire_graphe_depuis_fichier(nom_fichier):
    graphe = Graphe()
    
    with open(nom_fichier, 'r') as f:
        lignes = [ligne.strip() for ligne in f.readlines() if ligne.strip()]
    
    indice = 0
    
    # Lire le nombre de sommets
    if lignes[indice] == "Nombre de sommets":
        indice += 1
        n = int(lignes[indice])
        indice += 1

    
# Lire les sommets
    if lignes[indice] == "Sommets":
        indice += 1
        for _ in range(n):
            sommet = int(lignes[indice])
            graphe.ajouter_sommet(sommet)
            indice += 1
    
    # Lire le nombre d’arêtes
    if lignes[indice] == "Nombre d’arêtes":
        indice += 1
        m = int(lignes[indice])
        indice += 1
    
    # Lire les arêtes
    if lignes[indice] == "Arêtes":
        indice += 1
        for _ in range(m):
            u, v = map(int, lignes[indice].split())
            graphe.ajouter_arete(u, v)
            indice += 1
    
    return graphe

# Fonction qui génère un graphe aléatoire G(n,p) selon le modèle d’Erdős–Rényi
# n : nombre de sommets
# p : probabilité d’existence de chaque arête
def generer_graphe_aleatoire(n, p):
    graphe = Graphe()
    
    # Ajouter tous les sommets
    for i in range(n):
        graphe.ajouter_sommet(i)
    
    # Ajouter des arêtes avec la probabilité p
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                graphe.ajouter_arete(i, j)
    
    return graphe


# PARTIE 2: MÉTHODES APPROCHÉES

# Algorithme via un couplage maximal, retourne une couverture de sommets
def algo_couplage(graphe):
    C = set()
    aretes = graphe.obtenir_aretes().copy()
    
    for u, v in aretes:
        if u not in C and v not in C:
            C.add(u)
            C.add(v)
    
    return C

# Algorithme glouton : à chaque étape choisir le sommet de degré maximal
def algo_glouton(graphe):
    C = set()
    G = graphe.copie()
    
    while G.nb_aretes() > 0:
        v = G.sommet_degre_max()
        if v is None:
            break
        
        C.add(v)
        G.supprimer_sommet(v)
    
    return C

# PARTIE 3: MÉTHODES EXACTES (BRANCH AND BOUND)

def calculer_bornes_inferieures(graphe):
    n = graphe.nb_sommets()
    m = graphe.nb_aretes()
    
    if n == 0 or m == 0:
        return 0
    
    # ceil(m / degré_max)
    degre_max = max(graphe.degre(v) for v in graphe.sommets) if graphe.sommets else 1
    b1 = math.ceil(m / degre_max) if degre_max > 0 else 0
    
    # taille du couplage
    couplage = []
    graphe_temp = graphe.copie()
    for u, v in graphe_temp.obtenir_aretes():
        if u in graphe_temp.sommets and v in graphe_temp.sommets:
            couplage.append((u, v))
            graphe_temp.supprimer_sommet(u)
            graphe_temp.supprimer_sommet(v)
    b2 = len(couplage)
    
    # formule quadratique
    discriminant = (2*n - 1)**2 - 8*m
    if discriminant >= 0:
        b3 = (2*n - 1 - math.sqrt(discriminant)) / 2
    else:
        b3 = 0
    
    return max(b1, b2, int(b3))

# Utilise une pile pour gérer les nœuds de l’arbre de recherche (algorithme de base)
def separation_et_evaluation_simple(graphe):
    meilleure_solution = set(graphe.sommets)
    meilleure_taille = len(meilleure_solution)
    noeuds_explores = 0
    
    # Pile : (graphe courant, couverture courante)
    pile = [(graphe.copie(), set())]
    
    while pile:
        graphe_courant, couverture_courante = pile.pop()
        noeuds_explores += 1

        if graphe_courant.nb_aretes() == 0:
            if len(couverture_courante) < meilleure_taille:
                meilleure_solution = couverture_courante.copy()
                meilleure_taille = len(couverture_courante)
            continue
        
        if len(couverture_courante) >= meilleure_taille:
            continue

        aretes = graphe_courant.obtenir_aretes()
        if not aretes:
            continue
        u, v = aretes[0]
        
        # Branche 1 : ajouter u à la couverture
        graphe1 = graphe_courant.copie()
        couverture1 = couverture_courante.copy()
        couverture1.add(u)
        graphe1.supprimer_sommet(u)
        pile.append((graphe1, couverture1))
        
        # Branche 2 : ajouter v à la couverture
        graphe2 = graphe_courant.copie()
        couverture2 = couverture_courante.copy()
        couverture2.add(v)
        graphe2.supprimer_sommet(v)
        pile.append((graphe2, couverture2))
    
    return meilleure_solution, noeuds_explores

# Algorithme amélioré avec bornes inférieures, solution heuristique et branchement optimisé
def separation_et_evaluation_amelioree(graphe):
    solution_initiale = algo_couplage(graphe)
    meilleure_solution = solution_initiale
    meilleure_taille = len(meilleure_solution)
    noeuds_explores = 0

    pile = [(graphe.copie(), set())]
    
    while pile:
        graphe_courant, couverture_courante = pile.pop()
        noeuds_explores += 1

        if graphe_courant.nb_aretes() == 0:
            if len(couverture_courante) < meilleure_taille:
                meilleure_solution = couverture_courante.copy()
                meilleure_taille = len(couverture_courante)
            continue

        borne_inferieure = len(couverture_courante) + calculer_bornes_inferieures(graphe_courant)
        
        # Élagage par borne inférieure
        if borne_inferieure >= meilleure_taille:
            continue
        
        # Élagage si la couverture courante est déjà trop grande
        if len(couverture_courante) >= meilleure_taille:
            continue
        
        # Obtenir une solution heuristique pour le graphe courant
        solution_heuristique = algo_couplage(graphe_courant)
        total_heuristique = couverture_courante.union(solution_heuristique)
        if len(total_heuristique) < meilleure_taille:
            meilleure_solution = total_heuristique
            meilleure_taille = len(total_heuristique)
            
        aretes = graphe_courant.obtenir_aretes()
        if not aretes:
            continue
        
        u, v = max(aretes, key=lambda e: max(graphe_courant.degre(e[0]), 
                                             graphe_courant.degre(e[1])))

        graphe1 = graphe_courant.copie()
        couverture1 = couverture_courante.copy()
        couverture1.add(u)
        graphe1.supprimer_sommet(u)
        pile.append((graphe1, couverture1))


        graphe2 = graphe_courant.copie()
        couverture2 = couverture_courante.copy()
        couverture2.add(v)
        voisins_u = list(graphe_courant.liste_adj[u])
        for voisin in voisins_u:
            couverture2.add(voisin)
            if voisin in graphe2.sommets:
                graphe2.supprimer_sommet(voisin)
        pile.append((graphe2, couverture2))
    
    return meilleure_solution, noeuds_explores


# PARTIE 4 : TESTS ET ANALYSE AVEC GRAPHIQUES

def collecter_donnees_vs_n(n_valeurs, p, nb_essais=10):
    donnees = {
        'n': [],
        'coupling_time': [],
        'greedy_time': [],
        'optimal_time': [],
        'coupling_size': [],
        'greedy_size': [],
        'optimal_size': [],
        'coupling_ratio': [],
        'greedy_ratio': []
    }
    
    print(f"\nCollecte des données pour p = {p}...")
    
    for n in n_valeurs:
        print(f"  Test de n = {n}...", end=" ")
        
        temps_couplage = []
        temps_glouton = []
        temps_optimal = []
        tailles_couplage = []
        tailles_glouton = []
        tailles_optimal = []
        rapports_couplage = []
        rapports_glouton = []
        
        for _ in range(nb_essais):
            graphe = generer_graphe_aleatoire(n, p)
            
            # Couplage
            debut = time.time()
            couverture_couplage = algo_couplage(graphe)
            temps_couplage.append(time.time() - debut)
            tailles_couplage.append(len(couverture_couplage))
            
            # Glouton
            debut = time.time()
            couverture_glouton = algo_glouton(graphe)
            temps_glouton.append(time.time() - debut)
            tailles_glouton.append(len(couverture_glouton))
            
            # Optimal (pour les petits graphes)
            if n <= 12:
                debut = time.time()
                couverture_opt, _ = separation_et_evaluation_amelioree(graphe)
                temps_optimal.append(time.time() - debut)
                taille_opt = len(couverture_opt)
                tailles_optimal.append(taille_opt)
                
                if taille_opt > 0:
                    rapports_couplage.append(len(couverture_couplage) / taille_opt)
                    rapports_glouton.append(len(couverture_glouton) / taille_opt)
        
        donnees['n'].append(n)
        donnees['coupling_time'].append(sum(temps_couplage) / len(temps_couplage))
        donnees['greedy_time'].append(sum(temps_glouton) / len(temps_glouton))
        donnees['coupling_size'].append(sum(tailles_couplage) / len(tailles_couplage))
        donnees['greedy_size'].append(sum(tailles_glouton) / len(tailles_glouton))
        
        if temps_optimal:
            donnees['optimal_time'].append(sum(temps_optimal) / len(temps_optimal))
            donnees['optimal_size'].append(sum(tailles_optimal) / len(tailles_optimal))
            donnees['coupling_ratio'].append(sum(rapports_couplage) / len(rapports_couplage))
            donnees['greedy_ratio'].append(sum(rapports_glouton) / len(rapports_glouton))
        
        print("✓")
    
    return donnees

# Collecte des données pour les graphiques en fonction de p
def collecter_donnees_vs_p(n, p_valeurs, nb_essais=10):
    donnees = {
        'p': [],
        'coupling_time': [],
        'greedy_time': [],
        'optimal_time': [],
        'coupling_size': [],
        'greedy_size': [],
        'optimal_size': [],
        'coupling_ratio': [],
        'greedy_ratio': []
    }
    
    print(f"\nCollecte des données pour n = {n}...")
    
    for p in p_valeurs:
        print(f"  Test de p = {p}...", end=" ")
        
        temps_couplage = []
        temps_glouton = []
        temps_optimal = []
        tailles_couplage = []
        tailles_glouton = []
        tailles_optimal = []
        rapports_couplage = []
        rapports_glouton = []
        
        for _ in range(nb_essais):
            graphe = generer_graphe_aleatoire(n, p)
            
            # Couplage
            debut = time.time()
            couverture_couplage = algo_couplage(graphe)
            temps_couplage.append(time.time() - debut)
            tailles_couplage.append(len(couverture_couplage))
            
            # Glouton
            debut = time.time()
            couverture_glouton = algo_glouton(graphe)
            temps_glouton.append(time.time() - debut)
            tailles_glouton.append(len(couverture_glouton))
            
            # Optimal
            debut = time.time()
            couverture_opt, _ = separation_et_evaluation_amelioree(graphe)
            temps_optimal.append(time.time() - debut)
            taille_opt = len(couverture_opt)
            tailles_optimal.append(taille_opt)
            
            if taille_opt > 0:
                rapports_couplage.append(len(couverture_couplage) / taille_opt)
                rapports_glouton.append(len(couverture_glouton) / taille_opt)
        
        donnees['p'].append(p)
        donnees['coupling_time'].append(sum(temps_couplage) / len(temps_couplage))
        donnees['greedy_time'].append(sum(temps_glouton) / len(temps_glouton))
        donnees['optimal_time'].append(sum(temps_optimal) / len(temps_optimal))
        donnees['coupling_size'].append(sum(tailles_couplage) / len(tailles_couplage))
        donnees['greedy_size'].append(sum(tailles_glouton) / len(tailles_glouton))
        donnees['optimal_size'].append(sum(tailles_optimal) / len(tailles_optimal))
        donnees['coupling_ratio'].append(sum(rapports_couplage) / len(rapports_couplage))
        donnees['greedy_ratio'].append(sum(rapports_glouton) / len(rapports_glouton))
        
        print("✓")
    
    return donnees

def tracer_tous_les_graphes(donnees_vs_n, donnees_vs_p):
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Analyse des algorithmes de couverture de sommets', fontsize=16, fontweight='bold')
    
    # GRAPHIQUES EN FONCTION DE N
    
    # Temps d’exécution en fonction de n
    ax1 = axes[0, 0]
    ax1.plot(donnees_vs_n['n'], donnees_vs_n['coupling_time'], 'o-', label='Couplage', linewidth=2)
    ax1.plot(donnees_vs_n['n'], donnees_vs_n['greedy_time'], 's-', label='Glouton', linewidth=2)
    if donnees_vs_n['optimal_time']:
        ax1.plot(donnees_vs_n['n'][:len(donnees_vs_n['optimal_time'])], 
                donnees_vs_n['optimal_time'], '^-', label='Optimal', linewidth=2)
    ax1.set_xlabel('Nombre de sommets (n)', fontsize=11)
    ax1.set_ylabel('Temps d’exécution (sec)', fontsize=11)
    ax1.set_title('Temps d’exécution en fonction de n', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Taille de la couverture en fonction de n
    ax2 = axes[0, 1]
    ax2.plot(donnees_vs_n['n'], donnees_vs_n['coupling_size'], 'o-', label='Couplage', linewidth=2)
    ax2.plot(donnees_vs_n['n'], donnees_vs_n['greedy_size'], 's-', label='Glouton', linewidth=2)
    if donnees_vs_n['optimal_size']:
        ax2.plot(donnees_vs_n['n'][:len(donnees_vs_n['optimal_size'])], 
                donnees_vs_n['optimal_size'], '^-', label='Optimal', linewidth=2)
    ax2.set_xlabel('Nombre de sommets (n)', fontsize=11)
    ax2.set_ylabel('Taille de la couverture', fontsize=11)
    ax2.set_title('Taille de la couverture en fonction de n', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Facteur d’approximation en fonction de n
    ax3 = axes[0, 2]
    if donnees_vs_n['coupling_ratio']:
        ax3.plot(donnees_vs_n['n'][:len(donnees_vs_n['coupling_ratio'])], 
                donnees_vs_n['coupling_ratio'], 'o-', label='Couplage', linewidth=2)
        ax3.plot(donnees_vs_n['n'][:len(donnees_vs_n['greedy_ratio'])], 
                donnees_vs_n['greedy_ratio'], 's-', label='Glouton', linewidth=2)
        ax3.axhline(y=1.0, color='g', linestyle='--', label='Optimal (1.0)', linewidth=1.5)
        ax3.axhline(y=2.0, color='r', linestyle='--', label='2-approximation', linewidth=1.5, alpha=0.5)
    ax3.set_xlabel('Nombre de sommets (n)', fontsize=11)
    ax3.set_ylabel('Rapport à l’optimal', fontsize=11)
    ax3.set_title('Facteur d’approximation en fonction de n', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    
# GRAPHIQUES EN FONCTION DE P
    
    # Temps d’exécution en fonction de p
    ax4 = axes[1, 0]
    ax4.plot(donnees_vs_p['p'], donnees_vs_p['coupling_time'], 'o-', label='Couplage', linewidth=2)
    ax4.plot(donnees_vs_p['p'], donnees_vs_p['greedy_time'], 's-', label='Glouton', linewidth=2)
    ax4.plot(donnees_vs_p['p'], donnees_vs_p['optimal_time'], '^-', label='Optimal', linewidth=2)
    ax4.set_xlabel('Probabilité d’arête (p)', fontsize=11)
    ax4.set_ylabel('Temps d’exécution (s)', fontsize=11)
    ax4.set_title('Temps d’exécution en fonction de p', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Taille de la couverture en fonction de p
    ax5 = axes[1, 1]
    ax5.plot(donnees_vs_p['p'], donnees_vs_p['coupling_size'], 'o-', label='Couplage', linewidth=2)
    ax5.plot(donnees_vs_p['p'], donnees_vs_p['greedy_size'], 's-', label='Glouton', linewidth=2)
    ax5.plot(donnees_vs_p['p'], donnees_vs_p['optimal_size'], '^-', label='Optimal', linewidth=2)
    ax5.set_xlabel('Probabilité d’arête (p)', fontsize=11)
    ax5.set_ylabel('Taille de la couverture de sommets', fontsize=11)
    ax5.set_title('Taille de la couverture en fonction de p', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Facteur d’approximation en fonction de p
    ax6 = axes[1, 2]
    ax6.plot(donnees_vs_p['p'], donnees_vs_p['coupling_ratio'], 'o-', label='Couplage', linewidth=2)
    ax6.plot(donnees_vs_p['p'], donnees_vs_p['greedy_ratio'], 's-', label='Glouton', linewidth=2)
    ax6.axhline(y=1.0, color='g', linestyle='--', label='Optimal (1.0)', linewidth=1.5)
    ax6.axhline(y=2.0, color='r', linestyle='--', label='2-approximation', linewidth=1.5, alpha=0.5)
    ax6.set_xlabel('Probabilité d’arête (p)', fontsize=11)
    ax6.set_ylabel('Rapport à l’optimal', fontsize=11)
    ax6.set_title('Facteur d’approximation en fonction de p', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vertex_cover_analysis.png', dpi=300, bbox_inches='tight')
    print("\n Les graphiques ont été enregistrés dans le fichier 'vertex_cover_analysis.png'")
    plt.show()


# Test des algorithmes approchés sur des graphes aléatoires
def tester_algorithmes_approximation(n_valeurs, p_valeurs, nb_essais=10):
    print("\nTEST DES ALGORITHMES APPROCHÉS")
    
    for p in p_valeurs:
        print(f"\n--- Probabilité p = {p} ---\n")
        print(f"{'n':>5} {'Taille Couplage':>15} {'Temps Couplage':>15} "
              f"{'Taille Glouton':>15} {'Temps Glouton':>15}")
        print("-" * 80)
        
        for n in n_valeurs:
            tailles_couplage = []
            temps_couplage = []
            tailles_glouton = []
            temps_glouton = []
            
            for _ in range(nb_essais):
                graphe = generer_graphe_aleatoire(n, p)
                
                # Test du couplage
                debut = time.time()
                couverture_couplage = algo_couplage(graphe)
                temps_couplage_single = time.time() - debut
                tailles_couplage.append(len(couverture_couplage))
                temps_couplage.append(temps_couplage_single)
                
                # Test du glouton
                debut = time.time()
                couverture_glouton = algo_glouton(graphe)
                temps_glouton_single = time.time() - debut
                tailles_glouton.append(len(couverture_glouton))
                temps_glouton.append(temps_glouton_single)
            
            avg_coupling_size = sum(tailles_couplage) / len(tailles_couplage)
            avg_coupling_time = sum(temps_couplage) / len(temps_couplage)
            avg_greedy_size = sum(tailles_glouton) / len(tailles_glouton)
            avg_greedy_time = sum(temps_glouton) / len(temps_glouton)
            
            print(f"{n:>5} {avg_coupling_size:>15.2f} {avg_coupling_time:>15.6f} "
                  f"{avg_greedy_size:>15.2f} {avg_greedy_time:>15.6f}")

# Test des algorithmes exacts (Branch and Bound)
def tester_algorithmes_exacts(n_valeurs, p, nb_essais=5):
    print("\n")
    print("TEST DES ALGORITHMES EXACTS (BRANCH AND BOUND)")

    print(f"\nProbabilité p = {p}\n")
    print(f"{'n':>5} {'Taille Basique':>12} {'Temps Basique':>12} {'Nœuds Basiques':>12} "
          f"{'Taille Améliorée':>15} {'Temps Amélioré':>15} {'Nœuds Améliorés':>15}")
    print("-" * 100)
    
    for n in n_valeurs:
        tailles_simple = []
        temps_simple = []
        noeuds_simple = []
        tailles_amelioree = []
        temps_amelioree = []
        noeuds_amelioree = []
        
        for _ in range(nb_essais):
            graphe = generer_graphe_aleatoire(n, p)
            
            # Algorithme basique
            debut = time.time()
            couverture_simple, explores_simple = separation_et_evaluation_simple(graphe)
            temps_s = time.time() - debut
            tailles_simple.append(len(couverture_simple))
            temps_simple.append(temps_s)
            noeuds_simple.append(explores_simple)
            
            # Algorithme amélioré
            debut = time.time()
            couverture_amel, explores_amel = separation_et_evaluation_amelioree(graphe)
            temps_a = time.time() - debut
            tailles_amelioree.append(len(couverture_amel))
            temps_amelioree.append(temps_a)
            noeuds_amelioree.append(explores_amel)
        
        print(f"{n:>5} {sum(tailles_simple)/len(tailles_simple):>12.2f} "
              f"{sum(temps_simple)/len(temps_simple):>12.4f} "
              f"{sum(noeuds_simple)/len(noeuds_simple):>12.0f} "
              f"{sum(tailles_amelioree)/len(tailles_amelioree):>15.2f} "
              f"{sum(temps_amelioree)/len(temps_amelioree):>15.4f} "
              f"{sum(noeuds_amelioree)/len(noeuds_amelioree):>15.0f}")

def evaluer_rapport_approximation(n_valeurs, p, nb_essais=10):
    print("\n")
    print("ÉVALUATION DU RAPPORT D’APPROXIMATION")

    print(f"\nProbabilité p = {p}\n")
    print(f"{'n':>5} {'Optimal':>10} {'Couplage':>10} {'Rapport':>10} "
          f"{'Glouton':>10} {'Rapport':>10}")
    print("-" * 65)
    
    for n in n_valeurs:
        tailles_opt = []
        rapports_c = []
        rapports_g = []
        
        for _ in range(nb_essais):
            graphe = generer_graphe_aleatoire(n, p)
            
            # Solution optimale
            couverture_opt, _ = separation_et_evaluation_amelioree(graphe)
            taille_opt = len(couverture_opt)
            
            # Solutions approchées
            couverture_c = algo_couplage(graphe)
            couverture_g = algo_glouton(graphe)
            
            if taille_opt > 0:
                tailles_opt.append(taille_opt)
                rapports_c.append(len(couverture_c) / taille_opt)
                rapports_g.append(len(couverture_g) / taille_opt)
        
        if tailles_opt:
            moy_opt = sum(tailles_opt) / len(tailles_opt)
            moy_rapport_c = sum(rapports_c) / len(rapports_c)
            moy_rapport_g = sum(rapports_g) / len(rapports_g)
            
            print(f"{n:>5} {moy_opt:>10.2f} {moy_opt*moy_rapport_c:>10.2f} "
                  f"{moy_rapport_c:>10.2f} {moy_opt*moy_rapport_g:>10.2f} "
                  f"{moy_rapport_g:>10.2f}")


# PARTIE 5 : FONCTION PRINCIPALE

def principal():
    print("\nEXEMPLE DE FONCTIONNEMENT DU PROGRAMME")
    print("-" * 40)
    try:

        graphe = Graphe()
        graphe.ajouter_arete(0, 1)
        graphe.ajouter_arete(3, 2)
        graphe.ajouter_arete(4, 1)
        graphe.ajouter_arete(0, 3)
        graphe.ajouter_arete(4, 2)
        graphe.ajouter_arete(1, 2)
        
        print(f"Graphe : {graphe.nb_sommets()} sommets, {graphe.nb_aretes()} arêtes")
        print(f"Arêtes : {graphe.obtenir_aretes()}")
        print(f"Degrés des sommets : {graphe.degres()}")
        print(f"Sommet avec degré max : {graphe.sommet_degre_max()}")
        
        couverture_couplage = algo_couplage(graphe)
        couverture_glouton = algo_glouton(graphe)
        couverture_opt, noeuds = separation_et_evaluation_amelioree(graphe)
        
        print(f"\nRésultats :")
        print(f"  Algorithme Couplage : {sorted(couverture_couplage)} (taille : {len(couverture_couplage)})")
        print(f"  Algorithme Glouton :   {sorted(couverture_glouton)} (taille : {len(couverture_glouton)})")
        print(f"  Optimal :              {sorted(couverture_opt)} (taille : {len(couverture_opt)})")
        print(f"  Nœuds explorés :       {noeuds}")
        
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
    
    print("\n")

    n_valeurs_approx = [10, 20, 30, 40, 50]
    p_valeurs = [0.3, 0.5]
    tester_algorithmes_approximation(n_valeurs_approx, p_valeurs, nb_essais=10)

    n_valeurs_exacts = [5, 7, 9, 11, 13]
    tester_algorithmes_exacts(n_valeurs_exacts, p=0.5, nb_essais=5)
    
    n_valeurs_rapport = [5, 7, 9, 11]
    evaluer_rapport_approximation(n_valeurs_rapport, p=0.5, nb_essais=8)
    
    print("\n")
    print("TRACÉ DES GRAPHIQUES")
    
    n_valeurs_pour_plot = [5, 7, 9, 11, 13, 15, 17, 19, 21]
    donnees_vs_n = collecter_donnees_vs_n(n_valeurs_pour_plot, p=0.5, nb_essais=8)
    
    p_valeurs_pour_plot = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    donnees_vs_p = collecter_donnees_vs_p(n=10, p_valeurs=p_valeurs_pour_plot, nb_essais=10)
    
    tracer_tous_les_graphes(donnees_vs_n, donnees_vs_p)
    
    print("\n")
    print("TESTS TERMINÉS")


if __name__ == "__main__":
    principal()
