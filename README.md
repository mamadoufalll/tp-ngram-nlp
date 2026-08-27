# TP — Modèles de langage basés sur les N-grammes

TP de Traitement Automatique du Langage Naturel (NLP)
Master IA & Data Engineering — Institut Supérieur d'Informatique — 2026/2027

Construction progressive d'un modèle de langage statistique à partir d'un corpus,
puis application à cinq tâches NLP. Aucune bibliothèque NLP n'est utilisée : tout
est implémenté en Python pur, conformément à l'énoncé.

## Structure

| Fichier | Rôle |
|---|---|
| modele_langage.py | Toutes les fonctions (Parties 1 à 11) |
| mini_modele_langage.py | Partie 12 : programme interactif à menu |
| evaluer_modele.py | Perplexité train/test, choix de alpha |
| TP_Ngram.ipynb | Notebook : code, résultats, réponses rédigées |
| tests/ | 51 tests unitaires |
| data/ | corpus.txt et corpus_correction.txt |

La logique vit dans modele_langage.py ; le notebook et le mini-projet l'importent.
Aucun code n'est dupliqué.

## Installation

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Utilisation

    jupyter notebook TP_Ngram.ipynb
    python mini_modele_langage.py
    python evaluer_modele.py
    pytest tests/ -v

Le mini-projet doit être lancé depuis la racine : il lit data/corpus.txt en
chemin relatif.

## Résultats principaux

Corpus : 6 phrases, N = 45 tokens, V = 15 types.

Dispersion des données :

| Modèle | Observés | Possibles | Couverture | Hapax |
|---|---|---|---|---|
| Unigramme | 15 | 15 | 100 % | 2/15 |
| Bigramme | 23 | 225 | 10.2 % | 13/23 |
| Trigramme | 25 | 3375 | 0.74 % | 19/25 |

Apport du contexte long — sur le contexte « le chat mange », le bigramme hésite
(de 0.500 / du 0.500) tandis que le trigramme tranche (du 1.000).

Effet du lissage — 202 bigrammes sur 225 sont nuls. Sans lissage, trois phrases
sur quatre reçoivent P = 0 et deviennent indistinguables :

| Phrase | MLE | Laplace |
|---|---|---|
| le chat mange du poisson | 0.055556 | 1.60e-05 |
| le chat mange du pain | 0 | 3.03e-06 |
| le poisson mange du chat | 0 | 6.68e-07 |
| poisson le mange chat du | 0 | 2.38e-08 |

Avec lissage, S1 est 672 fois plus probable que S2 et l'ordre correspond à la
plausibilité linguistique.

Correction contextuelle — le système corrige dans les deux sens, ce qui montre
que seul le contexte décide :

- « Il a cet ans » devient « il a sept ans »
- « Il a sept objet » devient « il a cet objet »

Évaluation — perplexité 6.13 en entraînement et 8.36 en test. L'optimum du
paramètre de lissage est à alpha = 0.1 (PP = 5.57) et non à alpha = 1 : le
lissage add-one est trop agressif sur un corpus de cette taille.

## Limite documentée

Pour le contexte de fin de phrase, la distribution lissée somme à 0.714 et non
à 1 : ce marqueur termine une phrase et n'ouvre jamais de bigramme, donc la
somme des comptages vaut 0 alors que son comptage unigramme vaut 6. Sans
conséquence pratique, et couvert par un test dédié.
