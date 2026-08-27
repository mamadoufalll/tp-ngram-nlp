# -*- coding: utf-8 -*-
"""
Modele de langage base sur les N-grammes.
TP NLP - Master IA / Data Engineering - ISI

Ce module regroupe toutes les fonctions du TP.
Il est importe par le notebook TP_Ngram.ipynb et par mini_modele_langage.py.
"""

import math
import random
import re
from collections import Counter


DEBUT = "<s>"
FIN = "</s>"



# PARTIE 1 - PRETRAITEMENT


def tokeniser(phrase, ajouter_marqueurs=True):
    """Transforme une phrase brute en liste de tokens.

    Etapes : minuscules -> suppression ponctuation -> decoupage -> marqueurs.

    >>> tokeniser("Le chat mange du poisson.")
    ['<s>', 'le', 'chat', 'mange', 'du', 'poisson', '</s>']
    """
    phrase = phrase.lower()
    phrase = re.sub(r"[^a-z\u00e0\u00e2\u00e4\u00e9\u00e8\u00ea\u00eb\u00ee\u00ef\u00f4\u00f6\u00f9\u00fb\u00fc\u00e7'\s]", " ", phrase)
    tokens = phrase.split()
    if ajouter_marqueurs:
        tokens = [DEBUT] + tokens + [FIN]
    return tokens


def charger_corpus(chemin):
    """Lit un fichier texte (une phrase par ligne) et retourne une liste
    de phrases tokenisees (liste de listes)."""
    with open(chemin, "r", encoding="utf-8") as f:
        lignes = f.read().strip().split("\n")
    return [tokeniser(ligne) for ligne in lignes if ligne.strip()]


def construire_vocabulaire(corpus):
    """Retourne la liste triee des mots distincts (types) du corpus."""
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



# PARTIE 2 - CONSTRUCTION DES N-GRAMMES


def construire_ngrammes(corpus, n):
    """Construit les N-grammes d'ordre n et retourne leurs frequences.

    Les N-grammes ne traversent JAMAIS une frontiere de phrase : ils sont
    construits phrase par phrase. Le dernier token d'une phrase et le premier
    de la suivante n'ont aucun lien linguistique.

    Retourne un Counter dont les cles sont des tuples de n tokens.
    Pour n = 1, les cles sont des tuples a un element : ('le',).
    """
    compteur = Counter()
    for phrase in corpus:
        for i in range(len(phrase) - n + 1):
            compteur[tuple(phrase[i:i + n])] += 1
    return compteur


def construire_unigrammes(corpus):
    """Frequences des unigrammes."""
    return construire_ngrammes(corpus, 1)


def construire_bigrammes(corpus):
    """Frequences des bigrammes."""
    return construire_ngrammes(corpus, 2)


def construire_trigrammes(corpus):
    """Frequences des trigrammes."""
    return construire_ngrammes(corpus, 3)


def afficher_ngrammes(compteur, titre="N-grammes", limite=None):
    """Affiche un tableau lisible des N-grammes tries par frequence
    decroissante, puis par ordre alphabetique."""
    items = sorted(compteur.items(), key=lambda kv: (-kv[1], kv[0]))
    if limite:
        items = items[:limite]
    largeur = max(len(" ".join(ng)) for ng in compteur) + 2
    print(f"{titre} - {len(compteur)} distincts, {sum(compteur.values())} occurrences\n")
    print(f"{'N-gramme':{largeur}s} {'freq':>4s}")
    print("-" * (largeur + 5))
    for ngramme, freq in items:
        print(f"{' '.join(ngramme):{largeur}s} {freq:>4d}")



# PARTIE 3 - MODELE BIGRAMME


class ModeleNgramme:
    """Modele de langage N-gramme entraine sur un corpus tokenise.

    Regroupe le corpus, le vocabulaire et les comptages d'unigrammes,
    bigrammes et trigrammes dans un seul objet. Evite les variables
    globales et permet d'entrainer plusieurs modeles en parallele
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

    #  comptages 

    def compte_unigramme(self, mot):
        """C(mot) : nombre d'occurrences du mot dans le corpus."""
        return self.unigrammes[(mot,)]

    def compte_bigramme(self, mot_precedent, mot):
        """C(mot_precedent, mot) : nombre d'occurrences du bigramme."""
        return self.bigrammes[(mot_precedent, mot)]

    #  probabilites 

    def probabilite_bigramme(self, mot_precedent, mot):
        """P(mot | mot_precedent) = C(mot_precedent, mot) / C(mot_precedent)

        Estimation par maximum de vraisemblance (MLE).
        Retourne 0.0 si le mot precedent est absent du corpus, afin
        d'eviter une division par zero.
        """
        denominateur = self.compte_unigramme(mot_precedent)
        if denominateur == 0:
            return 0.0
        return self.compte_bigramme(mot_precedent, mot) / denominateur

    def successeurs(self, mot_precedent):
        """Retourne {mot: probabilite} pour tous les mots observes apres
        mot_precedent, tries par probabilite decroissante."""
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
        """Retourne une chaine explicitant le calcul, pour l'affichage."""
        c_bi = self.compte_bigramme(mot_precedent, mot)
        c_uni = self.compte_unigramme(mot_precedent)
        p = self.probabilite_bigramme(mot_precedent, mot)
        return (f"P({mot} | {mot_precedent}) = C({mot_precedent}, {mot}) / C({mot_precedent})"
                f" = {c_bi}/{c_uni} = {p:.4f}")

    #  PARTIE 4 : prediction du mot suivant
    def predire_mot_suivant(self, contexte, k=None):
        """Retourne les candidats possibles apres un contexte, tries par
        probabilite decroissante.

        Le contexte peut etre une chaine ou une liste de tokens. Comme le
        modele est bigramme, SEUL LE DERNIER MOT du contexte est utilise :
        c'est exactement l'hypothese de Markov d'ordre 1.

        Retourne une liste de couples (mot, probabilite), eventuellement
        tronquee aux k meilleurs. Liste vide si aucun successeur observe.
        """
        tokens = contexte.split() if isinstance(contexte, str) else list(contexte)
        if not tokens:
            tokens = [DEBUT]
        dernier = tokens[-1].lower()

        candidats = list(self.successeurs(dernier).items())
        return candidats[:k] if k else candidats

    def meilleur_mot_suivant(self, contexte):
        """Retourne le mot le plus probable apres le contexte, ou None.

        En cas d'egalite, le tri de successeurs() departage par ordre
        alphabetique : la prediction est donc DETERMINISTE.
        """
        candidats = self.predire_mot_suivant(contexte)
        return candidats[0][0] if candidats else None

    def afficher_prediction(self, contexte):
        """Affiche joliment les candidats apres un contexte donne."""
        tokens = contexte.split() if isinstance(contexte, str) else list(contexte)
        dernier = tokens[-1].lower() if tokens else DEBUT
        candidats = self.predire_mot_suivant(contexte)

        print(f"Contexte : « {contexte} »   -> mot conditionnant : « {dernier} »"
              f"  (C = {self.compte_unigramme(dernier)})")

        if not candidats:
            print("    aucun successeur observe : le modele ne peut rien predire\n")
            return

        for mot, p in candidats:
            barre = "#" * int(p * 30)
            print(f"    P({mot:8s}| {dernier:6s}) = {p:.4f}  {barre}")

        meilleur = candidats[0]
        exaequo = [m for m, p in candidats if p == meilleur[1]]
        if len(exaequo) > 1:
            print(f"    => EGALITE entre {', '.join(exaequo)} "
                  f"-> choix alphabetique : « {meilleur[0]} »\n")
        else:
            print(f"    => mot le plus probable : « {meilleur[0]} »\n")

    #  PARTIE 5 : generation de texte 
    def generer_phrase(self, mode="argmax", longueur_max=20, graine=None,
                       tracer=False):
        """Genere une phrase en partant de <s> jusqu'a </s>.

        mode = "argmax"      : choisit toujours le mot le plus probable.
                               Deterministe -> genere toujours la MEME phrase.
        mode = "echantillon" : tire le mot au hasard selon la distribution
                               de probabilite (random.choices). Non
                               deterministe -> phrases variees.

        longueur_max evite une boucle infinie si </s> n'est jamais atteint.
        graine fixe le tirage aleatoire pour rendre l'execution reproductible.
        tracer affiche le detail de chaque etape.
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
                raise ValueError("mode doit etre 'argmax' ou 'echantillon'")

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

    # PARTIE 6 : probabilite d'une phrase

    def probabilite_phrase(self, phrase, tracer=False, lissage=False):
        """Calcule P(S) par la regle de la chaine sous hypothese bigramme :

            P(S) = P(w1|<s>) * P(w2|w1) * ... * P(</s>|wn)

        phrase peut etre une chaine brute ou une liste de tokens.
        lissage=True utilise Laplace au lieu du maximum de vraisemblance
        (methode definie en Partie 10).
        Retourne la probabilite (float).
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
        """Retourne log2 P(S). Vaut -inf si une probabilite est nulle.

        Travailler en logarithmes transforme le produit en somme et evite
        le soupassement numerique (underflow) sur les phrases longues.
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
        """Perplexite = 2^(-log2 P(S) / nombre de transitions).

        Interpretation : nombre moyen de choix equiprobables auxquels le
        modele fait face a chaque mot. Plus c'est bas, mieux le modele
        predit la phrase.
        """
        tokens = tokeniser(phrase) if isinstance(phrase, str) else list(phrase)
        n = len(tokens) - 1
        logp = self.log_probabilite_phrase(tokens, lissage=lissage)
        if logp == float("-inf"):
            return float("inf")
        return 2 ** (-logp / n)

    # PARTIE 7 : comparaison de phrases

    def comparer_phrases(self, phrase1, phrase2, lissage=False, tracer=True):
        """Compare deux phrases et designe la plus probable.

        Retourne un tuple (P(S1), P(S2)).
        Avec lissage=False, deux phrases peuvent etre a egalite a 0.0 :
        le modele est alors incapable de trancher.
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
                print("VERDICT : egalite a zero -> le modele ne peut pas trancher.")
            elif p1 > p2:
                rapport = "infini" if p2 == 0 else f"{p1/p2:.1f}x"
                print(f"VERDICT : S1 est plus probable que S2 ({rapport}).")
            elif p2 > p1:
                rapport = "infini" if p1 == 0 else f"{p2/p1:.1f}x"
                print(f"VERDICT : S2 est plus probable que S1 ({rapport}).")
            else:
                print("VERDICT : les deux phrases sont equiprobables.")
        return p1, p2



# PARTIE 8 - CORRECTION CONTEXTUELLE

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
    """Score d'un candidat dans son contexte immediat :

        score = P(candidat | precedent) x P(suivant | candidat)

    On utilise les DEUX cotes du mot. Le contexte gauche seul suffit
    rarement : c'est souvent le mot qui SUIT qui est decisif.
    """
    gauche = modele.probabilite_bigramme(precedent, candidat)
    droite = modele.probabilite_bigramme(candidat, suivant) if suivant else 1.0
    return gauche * droite


def corriger_phrase(modele, phrase, confusions=CONFUSIONS, tracer=True):
    """Detecte et corrige les confusions contextuelles d'une phrase.

    Pour chaque mot appartenant a un groupe de confusion, evalue tous les
    candidats dans leur contexte et retient le mieux score.
    Retourne la liste des tokens corriges.
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
                print(f"    => CORRECTION : « {mot} » remplace par « {meilleur} »\n")
        elif tracer:
            print(f"    => aucun changement\n")

    return corrige



# PARTIE 9 - LE PROBLEME DES COMPTES NULS


def bigrammes_nuls(modele, limite=None, exclure_marqueurs=False):
    """Retourne la liste des bigrammes (w1, w2) jamais observes.

    On parcourt le produit cartesien V x V et on retient les couples de
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

    #  PARTIE 10 : lissage de Laplace

    def probabilite_laplace(self, mot_precedent, mot, alpha=1.0):
        """P_Laplace(mot | mot_precedent) = (C(w1,w2) + a) / (C(w1) + a*V)

        On ajoute a a chaque numerateur (add-one si a = 1) et a*V au
        denominateur pour que la distribution somme toujours a 1.
        Aucune probabilite n'est jamais nulle.
        """
        num = self.compte_bigramme(mot_precedent, mot) + alpha
        den = self.compte_unigramme(mot_precedent) + alpha * self.V
        return num / den

    def detail_laplace(self, mot_precedent, mot, alpha=1.0):
        """Chaine explicitant le calcul lisse, pour l'affichage."""
        c_bi = self.compte_bigramme(mot_precedent, mot)
        c_uni = self.compte_unigramme(mot_precedent)
        p = self.probabilite_laplace(mot_precedent, mot, alpha)
        return (f"P_Laplace({mot} | {mot_precedent}) = "
                f"({c_bi} + {alpha:g}) / ({c_uni} + {alpha:g}x{self.V}) = "
                f"{p:.6f}")

    def compte_reconstitue(self, mot_precedent, mot, alpha=1.0):
        """Comptage effectif apres lissage : C*(w1,w2) = P_Laplace x C(w1).

        Permet de visualiser combien d'occurrences le lissage a
        RETIREES aux bigrammes observes pour les donner aux autres.
        """
        return self.probabilite_laplace(mot_precedent, mot, alpha) * \
            self.compte_unigramme(mot_precedent)

    def distribution_laplace(self, mot_precedent, alpha=1.0):
        """Distribution lissee complete sur TOUT le vocabulaire."""
        return {mot: self.probabilite_laplace(mot_precedent, mot, alpha)
                for mot in self.vocabulaire}


    #  PARTIE 11 : modeles unigramme et trigramme 

    def probabilite_unigramme(self, mot, lissage=False, alpha=1.0):
        """P(mot) = C(mot) / N   -- aucun contexte utilise.

        Le modele unigramme ignore totalement les mots precedents :
        il ne modelise que la frequence brute de chaque mot.
        """
        if lissage:
            return (self.compte_unigramme(mot) + alpha) / (self.N + alpha * self.V)
        return self.compte_unigramme(mot) / self.N if self.N else 0.0

    def compte_trigramme(self, w1, w2, w3):
        """C(w1, w2, w3) : occurrences du trigramme."""
        return self.trigrammes[(w1, w2, w3)]

    def probabilite_trigramme(self, w1, w2, w3, lissage=False, alpha=1.0):
        """P(w3 | w1, w2) = C(w1,w2,w3) / C(w1,w2)

        Le contexte est ici constitue des DEUX mots precedents.
        Le denominateur est le comptage du BIGRAMME (w1, w2), pas celui
        d'un unigramme : c'est l'erreur classique sur cette formule.
        """
        num = self.compte_trigramme(w1, w2, w3)
        den = self.compte_bigramme(w1, w2)
        if lissage:
            return (num + alpha) / (den + alpha * self.V)
        return num / den if den else 0.0

    def predire_unigramme(self, k=5):
        """Les k mots les plus frequents, sans aucun contexte."""
        total = self.N
        candidats = sorted(
            ((mot[0], freq / total) for mot, freq in self.unigrammes.items()),
            key=lambda kv: (-kv[1], kv[0]))
        return candidats[:k]

    def predire_trigramme(self, contexte, k=None):
        """Predit le mot suivant a partir des DEUX derniers mots du contexte.

        Retourne une liste vide si le bigramme de contexte n'a jamais ete
        observe : le modele trigramme est alors muet.
        """
        tokens = contexte.split() if isinstance(contexte, str) else list(contexte)
        if len(tokens) < 2:
            tokens = [DEBUT] + tokens
        w1, w2 = tokens[-2].lower(), tokens[-1].lower()

        den = self.compte_bigramme(w1, w2)
        if den == 0:
            return []

        candidats = sorted(
            ((tri[2], freq / den) for tri, freq in self.trigrammes.items()
             if tri[0] == w1 and tri[1] == w2),
            key=lambda kv: (-kv[1], kv[0]))
        return candidats[:k] if k else candidats

    def comparer_modeles(self, contexte, k=3):
        """Compare les predictions des trois modeles pour un meme contexte."""
        tokens = contexte.split() if isinstance(contexte, str) else list(contexte)

        uni = self.predire_unigramme(k)
        bi = self.predire_mot_suivant(contexte, k)
        tri = self.predire_trigramme(contexte, k)

        print(f"Contexte : « {contexte} »")
        for nom, candidats, vu in [
                ("unigramme", uni, "aucun contexte"),
                ("bigramme ", bi, f"« {tokens[-1]} »"),
                ("trigramme", tri, f"« {' '.join(tokens[-2:])} »" if len(tokens) >= 2
                 else f"« {DEBUT} {tokens[-1]} »")]:
            if candidats:
                detail = "  ".join(f"{m} ({p:.3f})" for m, p in candidats)
            else:
                detail = "AUCUNE PREDICTION (contexte jamais observe)"
            print(f"    {nom} (voit {vu:22s}) : {detail}")
        print()
        return uni, bi, tri
