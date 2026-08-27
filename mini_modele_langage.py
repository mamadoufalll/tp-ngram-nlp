# -*- coding: utf-8 -*-
"""
MINI MODELE DE LANGAGE
TP NLP - Modeles de langage bases sur les N-grammes
Master IA / Data Engineering - ISI

Programme interactif exploitant le module modele_langage.py.
Lancement :  python mini_modele_langage.py
"""

import sys

from modele_langage import (
    DEBUT, FIN, CONFUSIONS,
    ModeleNgramme, charger_corpus, tokeniser,
    afficher_ngrammes, corriger_phrase,
    bigrammes_nuls, couverture_bigrammes,
)

CORPUS_PRINCIPAL = "data/corpus.txt"
CORPUS_CORRECTION = "data/corpus_correction.txt"

MENU = """

        MINI MODELE DE LANGAGE

 1. Afficher le vocabulaire
 2. Afficher les unigrammes
 3. Afficher les bigrammes
 4. Afficher les trigrammes
 5. Calculer une probabilite
 6. Predire le mot suivant
 7. Generer une phrase
 8. Calculer la probabilite d'une phrase
 9. Corriger une phrase
10. Comparer deux phrases
11. Quitter

"""



# Utilitaires d'interface


def demander(invite, defaut=None):
    """Lit une saisie utilisateur, avec valeur par defaut optionnelle."""
    suffixe = f" [{defaut}]" if defaut else ""
    reponse = input(f"{invite}{suffixe} : ").strip()
    return reponse or (defaut or "")


def pause():
    input("\n(Entree pour revenir au menu)")


def separateur(titre):
    print(f"\n--- {titre} ---\n")



# Options du menu


def option_vocabulaire(modele):
    separateur("VOCABULAIRE")
    print(f"Taille : V = {modele.V}   |   Tokens : N = {modele.N}\n")
    for i, mot in enumerate(modele.vocabulaire, 1):
        marque = "  (marqueur)" if mot in (DEBUT, FIN) else ""
        print(f"  {i:2d}. {mot}{marque}")


def option_unigrammes(modele):
    separateur("UNIGRAMMES")
    afficher_ngrammes(modele.unigrammes, "Unigrammes")


def option_bigrammes(modele):
    separateur("BIGRAMMES")
    afficher_ngrammes(modele.bigrammes, "Bigrammes")
    stats = couverture_bigrammes(modele)
    print(f"\nCouverture : {stats['observes']}/{stats['possibles']} "
          f"({stats['taux_couverture']:.1%}) - {stats['nuls']} bigrammes nuls")


def option_trigrammes(modele):
    separateur("TRIGRAMMES")
    afficher_ngrammes(modele.trigrammes, "Trigrammes")


def option_probabilite(modele):
    separateur("CALCUL D'UNE PROBABILITE")
    precedent = demander("Mot precedent", "le").lower()
    mot = demander("Mot suivant", "chat").lower()

    print()
    print("  MLE      :", modele.detail_probabilite(precedent, mot))
    print("  Laplace  :", modele.detail_laplace(precedent, mot))

    if modele.probabilite_bigramme(precedent, mot) == 0:
        print("\n  -> bigramme jamais observe : le lissage evite la probabilite nulle.")


def option_prediction(modele):
    separateur("PREDICTION DU MOT SUIVANT")
    contexte = demander("Contexte", "le chat")

    print()
    modele.afficher_prediction(contexte)
    print("Comparaison des trois modeles :\n")
    modele.comparer_modeles(contexte)


def option_generation(modele):
    separateur("GENERATION DE PHRASES")
    mode = demander("Mode (argmax / echantillon)", "echantillon")
    if mode not in ("argmax", "echantillon"):
        print("Mode inconnu, utilisation de 'echantillon'.")
        mode = "echantillon"

    try:
        nombre = int(demander("Nombre de phrases", "5"))
    except ValueError:
        nombre = 5

    print()
    for i in range(nombre):
        tokens = modele.generer_phrase(mode=mode)
        print(f"  {i+1}. {modele.phrase_lisible(tokens)}")

    if mode == "argmax":
        print("\n  -> l'argmax est deterministe : les phrases sont identiques.")


def option_probabilite_phrase(modele):
    separateur("PROBABILITE D'UNE PHRASE")
    phrase = demander("Phrase", "le chat mange du poisson")

    print()
    modele.probabilite_phrase(phrase, tracer=True)
    lisse = modele.probabilite_phrase(phrase, lissage=True)
    print(f"  Avec lissage de Laplace : {lisse:.3e}")
    print(f"  Perplexite              : {modele.perplexite(phrase):.2f}")


def option_correction(modele_corr):
    separateur("CORRECTION CONTEXTUELLE")
    print("Modele entraine sur le corpus de correction.")
    print(f"Confusions gerees : {[sorted(g) for g in CONFUSIONS]}\n")

    phrase = demander("Phrase a corriger", "Il a cet ans.")
    print()
    corrige = corriger_phrase(modele_corr, phrase)
    print(f"Phrase corrigee : « {modele_corr.phrase_lisible(corrige)} »")


def option_comparaison(modele):
    separateur("COMPARAISON DE DEUX PHRASES")
    p1 = demander("Phrase 1", "le chat mange du poisson")
    p2 = demander("Phrase 2", "poisson le mange chat du")

    print("\n=== Sans lissage (MLE) ===\n")
    modele.comparer_phrases(p1, p2, lissage=False)

    print("\n=== Avec lissage de Laplace ===\n")
    a, b = modele.comparer_phrases(p1, p2, lissage=True, tracer=False)
    print(f"  P(S1) = {a:.3e}   (perplexite {modele.perplexite(p1):.2f})")
    print(f"  P(S2) = {b:.3e}   (perplexite {modele.perplexite(p2):.2f})")
    if b:
        print(f"  -> S1 est {a/b:.0f} fois plus probable que S2")



# Boucle principale


def main():
    try:
        modele = ModeleNgramme(charger_corpus(CORPUS_PRINCIPAL))
        modele_corr = ModeleNgramme(charger_corpus(CORPUS_CORRECTION))
    except FileNotFoundError as erreur:
        print(f"Corpus introuvable : {erreur.filename}")
        print("Lancez le programme depuis la racine du projet.")
        sys.exit(1)

    print(f"Corpus charge : {len(modele.corpus)} phrases, "
          f"N = {modele.N} tokens, V = {modele.V} types")

    actions = {
        "1": lambda: option_vocabulaire(modele),
        "2": lambda: option_unigrammes(modele),
        "3": lambda: option_bigrammes(modele),
        "4": lambda: option_trigrammes(modele),
        "5": lambda: option_probabilite(modele),
        "6": lambda: option_prediction(modele),
        "7": lambda: option_generation(modele),
        "8": lambda: option_probabilite_phrase(modele),
        "9": lambda: option_correction(modele_corr),
        "10": lambda: option_comparaison(modele),
    }

    while True:
        print(MENU)
        choix = input("Votre choix : ").strip()

        if choix in ("11", "q", "quitter"):
            print("\nAu revoir.")
            break

        action = actions.get(choix)
        if action is None:
            print("\nChoix invalide : entrez un nombre entre 1 et 11.")
            continue

        try:
            action()
        except (ValueError, KeyError, IndexError) as erreur:
            print(f"\nErreur : {erreur}")
        pause()


if __name__ == "__main__":
    main()