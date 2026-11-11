---
tags: [interpretabilita, attribuzioni, metodo, sanity-check]
---

# Attribuzioni e sanity check

Se l'attenzione non basta, la famiglia successiva sono le attribuzioni: metodi che
assegnano a ogni token un punteggio di importanza per una decisione specifica. Gradienti,
Gradient×Input, Integrated Gradients, SHAP, LRP, e per i transformer le varianti che
propagano la rilevanza attraverso i livelli.

## Il paper che va letto prima di usarne una qualsiasi

**[[sanity-check-salienza]]** (Adebayo et al.) propone due test, e il risultato e' che
alcuni metodi molto usati sono **indipendenti dal modello e dai dati**: la mappa non
cambia se il modello viene riaddestrato con etichette permutate, e diventa indistinguibile
da un **rilevatore di bordi**, uno strumento che non ha bisogno ne' di modello ne' di
addestramento. Il messaggio operativo e' semplice: prima di usare una spiegazione, verificare
che dipenda dal modello che dovrebbe spiegare. Una mappa bella che non passa il test spiega
l'immagine (o il testo), non la rete.

Il progetto ha gli stessi riflessi per altre vie: `tools/noise_floor.py` isola il rumore
prima di interpretare un delta piccolo, `tools/oer_audit.py` misura il raggio d'azione di
una metrica prima di citarla. Qui lo strumento di spiegazione stesso va messo alla prova.

## Le varianti per transformer

**[[interpretabilita-transformer]]** (Chefer et al.) e' la versione che serve a un
transformer: assegna una rilevanza locale con Deep Taylor Decomposition e la propaga
attraverso i livelli **conservando la somma totale**, attraverso l'attenzione e le
connessioni residue. Le proprieta' che interessano: e' **class-specific** (la stessa
immagine o frase ha spiegazioni diverse per classi diverse, mentre molte mappe di
attenzione no), e la conservazione della somma e' una verifica interna: se la
rilevanza totale non si conserva, la propagazione e' sbagliata. C'e' codice pubblico.

Per un modello come Laya, che calcola piu' decisioni per ogni forward (una per domanda)
e ha una testa che legge solo i `[MASK]`, la proprieta' class-specific e' esattamente
quella che ci serve: la domanda giusta non e' "qual e' il token importante", ma
"qual e' il token importante **per l'opzione che il modello ha scelto invece di quella
giusta**".

## La regola operativa

Nel programma [[E23-meccanismo-injection]] ogni attribuzione viene accompagnata da tre
controlli, e nessuno e' opzionale:

1. **Randomizzazione del modello**: la stessa attribuzione su un encoder con pesi
   mescolati deve dare una mappa diversa. Se la da' uguale, lo strumento misura il testo.
2. **Randomizzazione delle etichette**: qui e' piu' complicato perche' non possiamo
   riaddestrare Laya; l'equivalente pratico e' l'attribuzione su una decisione il cui
   esito non dipende dalla frase (frase neutra, E22) — se la mappa e' la stessa, non
   distingue la nominazione.
3. **Confronto con l'ablazione**: il token piu' rilevante per il modello e il token la cui
   rimozione cambia di piu' la decisione devono coincidere. Se non coincidono, uno dei due
   strumenti mente, e va detto.

Il terzo controllo e' il piu' informativo per noi, ed e' anche il piu' economico: togliere
un token e rifare il forward e' una misura, non un'interpretazione.

## Note collegate
[[attenzione-come-spiegazione]] · [[circuiti-e-interventi]] · [[E22-attenzione-laya]]
