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