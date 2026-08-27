# -*- coding: utf-8 -*-
"""
Modèle de langage basé sur les N-grammes.
TP NLP — Master IA / Data Engineering — ISI

Ce module regroupe toutes les fonctions du TP.
Il est importé par le notebook TP_Ngram.ipynb et par mini_modele_langage.py.
"""

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