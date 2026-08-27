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