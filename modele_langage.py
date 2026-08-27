# -*- coding: utf-8 -*-
"""
Modèle de langage basé sur les N-grammes.
TP NLP — Master IA / Data Engineering — ISI

Ce module regroupe toutes les fonctions du TP.
Il est importé par le notebook TP_Ngram.ipynb et par mini_modele_langage.py.
"""
import math
import random
import re
from collections import Counter

# Marqueurs de début et de fin de phrase (section 3.1 du TP)
DEBUT = "<s>"
FIN = "</s>"


# =====================================================================
# PARTIE 1 — PRÉTRAITEMENT
# =====================================================================

def tokeniser(phrase, ajouter_marqueurs=True):
    """Transforme une phrase brute en liste de tokens.

    Étapes : minuscules -> suppression ponctuation -> découpage -> marqueurs.

    >>> tokeniser("Le chat mange du poisson.")
    ['<s>', 'le', 'chat', 'mange', 'du', 'poisson', '</s>']
    """
    phrase = phrase.lower()
    phrase = re.sub(r"[^a-zàâäéèêëîïôöùûüç'\s]", " ", phrase)
    tokens = phrase.split()
    if ajouter_marqueurs:
        tokens = [DEBUT] + tokens + [FIN]
    return tokens


def charger_corpus(chemin):
    """Lit un fichier texte (une phrase par ligne) et retourne une liste
    de phrases tokenisées (liste de listes)."""
    with open(chemin, "r", encoding="utf-8") as f:
        lignes = f.read().strip().split("\n")
    return [tokeniser(ligne) for ligne in lignes if ligne.strip()]


def construire_vocabulaire(corpus):
    """Retourne la liste triée des mots distincts (types) du corpus."""
    return sorted({token for phrase in corpus for token in phrase})


def aplatir(corpus):
    """Retourne la liste de toutes les occurrences de tokens du corpus."""
    return [token for phrase in corpus for token in phrase]


def statistiques_corpus(corpus):
    """Retourne un dictionnaire de statistiques descriptives du corpus."""
    tokens = aplatir(corpus)
    vocabulaire = construire_vocabulaire(corpus)
    return {
        "nb_phrases": len(corpus),
        "nb_tokens": len(tokens),
        "taille_vocabulaire": len(vocabulaire),
        "frequences": Counter(tokens),
    }

# =====================================================================
# PARTIE 2 — CONSTRUCTION DES N-GRAMMES
# =====================================================================

def construire_ngrammes(corpus, n):
    """Construit les N-grammes d'ordre n et retourne leurs fréquences.

    Les N-grammes ne traversent JAMAIS une frontière de phrase : ils sont
    construits phrase par phrase. Le dernier token d'une phrase et le premier
    de la suivante n'ont aucun lien linguistique.

    Retourne un Counter dont les clés sont des tuples de n tokens.
    Pour n = 1, les clés sont des tuples à un élément : ('le',).
    """
    compteur = Counter()
    for phrase in corpus:
        for i in range(len(phrase) - n + 1):
            compteur[tuple(phrase[i:i + n])] += 1
    return compteur


def construire_unigrammes(corpus):
    """Fréquences des unigrammes."""
    return construire_ngrammes(corpus, 1)


def construire_bigrammes(corpus):
    """Fréquences des bigrammes."""
    return construire_ngrammes(corpus, 2)


def construire_trigrammes(corpus):
    """Fréquences des trigrammes."""
    return construire_ngrammes(corpus, 3)


def afficher_ngrammes(compteur, titre="N-grammes", limite=None):
    """Affiche un tableau lisible des N-grammes triés par fréquence
    décroissante, puis par ordre alphabétique."""
    items = sorted(compteur.items(), key=lambda kv: (-kv[1], kv[0]))
    if limite:
        items = items[:limite]
    largeur = max(len(" ".join(ng)) for ng in compteur) + 2
    print(f"{titre} — {len(compteur)} distincts, {sum(compteur.values())} occurrences\n")
    print(f"{'N-gramme':{largeur}s} {'freq':>4s}")
    print("-" * (largeur + 5))
    for ngramme, freq in items:
        print(f"{' '.join(ngramme):{largeur}s} {freq:>4d}")

# =====================================================================
# PARTIE 3 — MODÈLE BIGRAMME
# =====================================================================

class ModeleNgramme:
    """Modèle de langage N-gramme entraîné sur un corpus tokenisé.

    Regroupe le corpus, le vocabulaire et les comptages d'unigrammes,
    bigrammes et trigrammes dans un seul objet. Évite les variables
    globales et permet d'entraîner plusieurs modèles en parallèle
    (utile en Partie 8, avec le corpus de correction).
    """

    def __init__(self, corpus):
        self.corpus = corpus
        self.vocabulaire = construire_vocabulaire(corpus)
        self.V = len(self.vocabulaire)
        self.unigrammes = construire_unigrammes(corpus)
        self.bigrammes = construire_bigrammes(corpus)
        self.trigrammes = construire_trigrammes(corpus)
        self.N = sum(self.unigrammes.values())

    # --- comptages -----------------------------------------------------

    def compte_unigramme(self, mot):
        """C(mot) : nombre d'occurrences du mot dans le corpus."""
        return self.unigrammes[(mot,)]

    def compte_bigramme(self, mot_precedent, mot):
        """C(mot_precedent, mot) : nombre d'occurrences du bigramme."""
        return self.bigrammes[(mot_precedent, mot)]

    # --- probabilités --------------------------------------------------

    def probabilite_bigramme(self, mot_precedent, mot):
        """P(mot | mot_precedent) = C(mot_precedent, mot) / C(mot_precedent)

        Estimation par maximum de vraisemblance (MLE).
        Retourne 0.0 si le mot précédent est absent du corpus, afin
        d'éviter une division par zéro.
        """
        denominateur = self.compte_unigramme(mot_precedent)
        if denominateur == 0:
            return 0.0
        return self.compte_bigramme(mot_precedent, mot) / denominateur

    def successeurs(self, mot_precedent):
        """Retourne {mot: probabilité} pour tous les mots observés après
        mot_precedent, triés par probabilité décroissante."""
        suivants = {
            bigramme[1]: freq
            for bigramme, freq in self.bigrammes.items()
            if bigramme[0] == mot_precedent
        }
        total = self.compte_unigramme(mot_precedent)
        if total == 0:
            return {}
        return dict(sorted(
            ((mot, freq / total) for mot, freq in suivants.items()),
            key=lambda kv: (-kv[1], kv[0])
        ))

    def detail_probabilite(self, mot_precedent, mot):
        """Retourne une chaîne explicitant le calcul, pour l'affichage."""
        c_bi = self.compte_bigramme(mot_precedent, mot)
        c_uni = self.compte_unigramme(mot_precedent)
        p = self.probabilite_bigramme(mot_precedent, mot)
        return (f"P({mot} | {mot_precedent}) = C({mot_precedent}, {mot}) / C({mot_precedent})"
                f" = {c_bi}/{c_uni} = {p:.4f}")
    

    # --- PARTIE 4 : prédiction du mot suivant --------------------------

    def predire_mot_suivant(self, contexte, k=None):
        """Retourne les candidats possibles après un contexte, triés par
        probabilité décroissante.

        Le contexte peut être une chaîne ou une liste de tokens. Comme le
        modèle est bigramme, SEUL LE DERNIER MOT du contexte est utilisé :
        c'est exactement l'hypothèse de Markov d'ordre 1.

        Retourne une liste de couples (mot, probabilité), éventuellement
        tronquée aux k meilleurs. Liste vide si aucun successeur observé.
        """
        tokens = contexte.split() if isinstance(contexte, str) else list(contexte)
        if not tokens:
            tokens = [DEBUT]
        dernier = tokens[-1].lower()

        candidats = list(self.successeurs(dernier).items())
        return candidats[:k] if k else candidats

    def meilleur_mot_suivant(self, contexte):
        """Retourne le mot le plus probable après le contexte, ou None.

        En cas d'égalité, le tri de successeurs() départage par ordre
        alphabétique : la prédiction est donc DÉTERMINISTE.
        """
        candidats = self.predire_mot_suivant(contexte)
        return candidats[0][0] if candidats else None

    def afficher_prediction(self, contexte):
        """Affiche joliment les candidats après un contexte donné."""
        tokens = contexte.split() if isinstance(contexte, str) else list(contexte)
        dernier = tokens[-1].lower() if tokens else DEBUT
        candidats = self.predire_mot_suivant(contexte)

        print(f"Contexte : « {contexte} »   -> mot conditionnant : « {dernier} »"
              f"  (C = {self.compte_unigramme(dernier)})")

        if not candidats:
            print("    aucun successeur observé : le modèle ne peut rien prédire\n")
            return

        for mot, p in candidats:
            barre = "#" * int(p * 30)
            print(f"    P({mot:8s}| {dernier:6s}) = {p:.4f}  {barre}")

        meilleur = candidats[0]
        exaequo = [m for m, p in candidats if p == meilleur[1]]
        if len(exaequo) > 1:
            print(f"    => ÉGALITÉ entre {', '.join(exaequo)} "
                  f"-> choix alphabétique : « {meilleur[0]} »\n")
        else:
            print(f"    => mot le plus probable : « {meilleur[0]} »\n")

    # --- PARTIE 5 : génération de texte --------------------------------

    def generer_phrase(self, mode="argmax", longueur_max=20, graine=None,
                       tracer=False):
        """Génère une phrase en partant de <s> jusqu'à </s>.

        mode = "argmax"      : choisit toujours le mot le plus probable.
                               Déterministe -> génère toujours la MÊME phrase.
        mode = "echantillon" : tire le mot au hasard selon la distribution
                               de probabilité (random.choices). Non
                               déterministe -> phrases variées.

        longueur_max évite une boucle infinie si </s> n'est jamais atteint.
        graine fixe le tirage aléatoire pour rendre l'exécution reproductible.
        tracer affiche le détail de chaque étape.
        """
        if graine is not None:
            random.seed(graine)

        phrase = [DEBUT]
        while len(phrase) < longueur_max:
            candidats = self.predire_mot_suivant(phrase[-1])
            if not candidats:
                break

            if mode == "argmax":
                mot = candidats[0][0]
            elif mode == "echantillon":
                mots = [m for m, _ in candidats]
                poids = [p for _, p in candidats]
                mot = random.choices(mots, weights=poids, k=1)[0]
            else:
                raise ValueError("mode doit être 'argmax' ou 'echantillon'")

            if tracer:
                proba = dict(candidats)[mot]
                print(f"    {phrase[-1]:8s} -> {mot:8s} (p = {proba:.3f}, "
                      f"{len(candidats)} candidat(s))")

            phrase.append(mot)
            if mot == FIN:
                break

        return phrase

    def phrase_lisible(self, tokens):
        """Retire les marqueurs et recompose une phrase lisible."""
        mots = [t for t in tokens if t not in (DEBUT, FIN)]
        return " ".join(mots)

    # --- PARTIE 6 : probabilité d'une phrase ---------------------------

    def probabilite_phrase(self, phrase, tracer=False, lissage=False):
        """Calcule P(S) par la règle de la chaîne sous hypothèse bigramme :

            P(S) = P(w1|<s>) * P(w2|w1) * ... * P(</s>|wn)

        phrase peut être une chaîne brute ou une liste de tokens.
        lissage=True utilise Laplace au lieu du maximum de vraisemblance
        (méthode définie en Partie 10).
        """
        tokens = tokeniser(phrase) if isinstance(phrase, str) else list(phrase)
        proba = 1.0

        if tracer:
            print(f"P({self.phrase_lisible(tokens)}) =")

        for precedent, mot in zip(tokens, tokens[1:]):
            p = (self.probabilite_laplace(precedent, mot) if lissage
                 else self.probabilite_bigramme(precedent, mot))
            proba *= p
            if tracer:
                marque = "  <-- ZERO" if p == 0 else ""
                print(f"    P({mot:8s} | {precedent:8s}) = {p:.6f}{marque}")

        if tracer:
            print(f"    {'':>8s}  ---------------------------")
            print(f"    produit = {proba:.8f}\n")
        return proba

    def log_probabilite_phrase(self, phrase, lissage=False):
        """Retourne log2 P(S). Vaut -inf si une probabilité est nulle.

        Travailler en logarithmes transforme le produit en somme et évite
        le soupassement numérique (underflow) sur les phrases longues.
        """
        tokens = tokeniser(phrase) if isinstance(phrase, str) else list(phrase)
        total = 0.0
        for precedent, mot in zip(tokens, tokens[1:]):
            p = (self.probabilite_laplace(precedent, mot) if lissage
                 else self.probabilite_bigramme(precedent, mot))
            if p == 0:
                return float("-inf")
            total += math.log2(p)
        return total

    def perplexite(self, phrase, lissage=True):
        """Perplexité = 2^(-log2 P(S) / nombre de transitions).

        Interprétation : nombre moyen de choix équiprobables auxquels le
        modèle fait face à chaque mot. Plus c'est bas, mieux le modèle
        prédit la phrase.
        """
        tokens = tokeniser(phrase) if isinstance(phrase, str) else list(phrase)
        n = len(tokens) - 1
        logp = self.log_probabilite_phrase(tokens, lissage=lissage)
        if logp == float("-inf"):
            return float("inf")
        return 2 ** (-logp / n)
    
    # --- PARTIE 7 : comparaison de phrases -----------------------------

    def comparer_phrases(self, phrase1, phrase2, lissage=False, tracer=True):
        """Compare deux phrases et désigne la plus probable.

        Retourne un tuple (P(S1), P(S2)).
        Avec lissage=False, deux phrases peuvent être à égalité à 0.0 :
        le modèle est alors incapable de trancher.
        """
        resultats = []
        for etiquette, phrase in [("S1", phrase1), ("S2", phrase2)]:
            p = self.probabilite_phrase(phrase, lissage=lissage)
            resultats.append(p)
            if tracer:
                print(f"{etiquette} = « {phrase} »")
                self.probabilite_phrase(phrase, tracer=True, lissage=lissage)

        p1, p2 = resultats
        if tracer:
            if p1 == p2 == 0:
                print("VERDICT : égalité à zéro -> le modèle ne peut pas trancher.")
            elif p1 > p2:
                rapport = "infini" if p2 == 0 else f"{p1/p2:.1f}x"
                print(f"VERDICT : S1 est plus probable que S2 ({rapport}).")
            elif p2 > p1:
                rapport = "infini" if p1 == 0 else f"{p2/p1:.1f}x"
                print(f"VERDICT : S2 est plus probable que S1 ({rapport}).")
            else:
                print("VERDICT : les deux phrases sont équiprobables.")
        return p1, p2
    
# =====================================================================
# PARTIE 8 — CORRECTION CONTEXTUELLE
# =====================================================================

# Groupes de mots confondables : homophones ou quasi-homophones qui
# existent TOUS dans le dictionnaire. Un correcteur lexical ne peut donc
# pas les détecter ; seul le contexte permet de trancher.
CONFUSIONS = [
    {"cet", "sept"},
    {"a", "à"},
]


def candidats_confusion(mot, confusions=CONFUSIONS):
    """Retourne l'ensemble des mots confondables avec `mot` (lui inclus)."""
    for groupe in confusions:
        if mot in groupe:
            return sorted(groupe)
    return [mot]


def score_contextuel(modele, precedent, candidat, suivant):
    """Score d'un candidat dans son contexte immédiat :

        score = P(candidat | precedent) x P(suivant | candidat)

    On utilise les DEUX côtés du mot. Le contexte gauche seul suffit
    rarement : c'est souvent le mot qui SUIT qui est décisif.
    """
    gauche = modele.probabilite_bigramme(precedent, candidat)
    droite = modele.probabilite_bigramme(candidat, suivant) if suivant else 1.0
    return gauche * droite


def corriger_phrase(modele, phrase, confusions=CONFUSIONS, tracer=True):
    """Détecte et corrige les confusions contextuelles d'une phrase.

    Pour chaque mot appartenant à un groupe de confusion, évalue tous les
    candidats dans leur contexte et retient le mieux scoré.
    """
    tokens = tokeniser(phrase) if isinstance(phrase, str) else list(phrase)
    corrige = list(tokens)

    for i in range(1, len(tokens) - 1):
        mot = tokens[i]
        candidats = candidats_confusion(mot, confusions)
        if len(candidats) <= 1:
            continue

        precedent = corrige[i - 1]
        suivant = tokens[i + 1] if i + 1 < len(tokens) else None

        scores = {c: score_contextuel(modele, precedent, c, suivant)
                  for c in candidats}
        meilleur = max(scores, key=lambda c: (scores[c], c == mot))

        if tracer:
            print(f"Position {i} : « {mot} »  "
                  f"(contexte : {precedent} ___ {suivant})")
            for c in candidats:
                pg = modele.probabilite_bigramme(precedent, c)
                pd = modele.probabilite_bigramme(c, suivant) if suivant else 1.0
                marque = " <-- retenu" if c == meilleur else ""
                print(f"    {c:6s} : P({c}|{precedent}) = {pg:.4f}"
                      f"  x  P({suivant}|{c}) = {pd:.4f}"
                      f"  =  {scores[c]:.4f}{marque}")

        if meilleur != mot:
            corrige[i] = meilleur
            if tracer:
                print(f"    => CORRECTION : « {mot} » remplacé par « {meilleur} »\n")
        elif tracer:
            print(f"    => aucun changement\n")

    return corrige

# =====================================================================
# PARTIE 9 — LE PROBLÈME DES COMPTES NULS
# =====================================================================

def bigrammes_nuls(modele, limite=None, exclure_marqueurs=False):
    """Retourne la liste des bigrammes (w1, w2) jamais observés.

    On parcourt le produit cartésien V x V et on retient les couples de
    comptage nul. exclure_marqueurs ignore les couples impliquant <s>/</s>.
    """
    mots = [m for m in modele.vocabulaire
            if not (exclure_marqueurs and m in (DEBUT, FIN))]
    nuls = [(w1, w2) for w1 in mots for w2 in mots
            if modele.compte_bigramme(w1, w2) == 0]
    return nuls[:limite] if limite else nuls


def couverture_bigrammes(modele):
    """Statistiques de couverture de la matrice des bigrammes."""
    V = modele.V
    possibles = V * V
    observes = len(modele.bigrammes)
    return {
        "V": V,
        "possibles": possibles,
        "observes": observes,
        "nuls": possibles - observes,
        "taux_couverture": observes / possibles,
        "hapax": sum(1 for f in modele.bigrammes.values() if f == 1),
    }


def matrice_bigrammes(modele, mots=None):
    """Affiche la matrice des comptages C(w1, w2) sous forme de tableau."""
    mots = mots or modele.vocabulaire
    largeur = max(len(m) for m in mots) + 1

    print(" " * largeur + "".join(f"{m[:6]:>7s}" for m in mots))
    for w1 in mots:
        ligne = f"{w1:{largeur}s}"
        for w2 in mots:
            c = modele.compte_bigramme(w1, w2)
            ligne += f"{c if c else '.':>7}"
        print(ligne)