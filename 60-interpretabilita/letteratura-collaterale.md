---
tags: [interpretabilita, letteratura, collaterale, insight]
---

# Letteratura collaterale (un hop piu' in la')

Materiale che **non** risponde alla domanda di [[E23-meccanismo-injection]] ma cambia gli
strumenti con cui la poniamo, o spiega perche' alcune strade sono chiuse. Ordinato per
quanto ci tocca da vicino.

## 1. Feature sparse e circuiti su feature

**Sparse Feature Circuits** (Marks et al., ICLR 2025, arXiv:2403.19647) e' l'evoluzione
naturale di [[verso-la-monosemanticita]]: invece di circuiti su teste o neuroni
(polisemantici), si scoprono sottografi causali su **feature** trovate da autoencoder
sparsi, e gli autori mostrano che le feature sono giudicate piu' interpretabili dei neuroni
dai valutatori umani. La parte che ci interessa e' **SHIFT**: si migliora la
generalizzazione di un classificatore rimuovendo le feature che un umano giudica
irrilevanti per il compito, cioe' si toglie la dipendenza da un segnale indesiderato
**senza dati con etichette**. Se E23 trova che Laya usa una direzione nello stato
residuo che codifica "e' stata nominata un'opzione", SHIFT e' il candidato naturale per
togliere quella dipendenza, ed e' l'unica famiglia di difese che non sia una regola
lessicale ([[E18-nomination-detector]]) ne' una soglia ([[E16-calibrazione-difesa]]).

Il costo da dichiarare: servono autoencoder sparsi addestrati sulle attivazioni di Laya,
quindi introduce un artefatto nostro dentro l'analisi. Non e' il primo passo.

## 2. Induction head e "pattern completion fuzzy"

**In-context Learning and Induction Heads** (Olsson et al., Anthropic 2022,
arXiv:2209.11895) descrive teste che completano `[A][B] ... [A] -> [B]`, con quota crescente
di generalizzazione: in modelli piccoli il match e' esatto sul token, in modelli grandi
diventa "fuzzy", nearest-neighbour in uno spazio semantico. Nei modelli piccoli ci sono
prove causali forti (ablare le induction head riduce l'ICL in proporzione), nei grandi solo
correlazionali.

L'insight per noi: **il meccanismo di E03 assomiglia a una induction head applicata alle
opzioni**. La frase nomina l'opzione X, e lo stato al `[MASK]` di X sale. Se dentro Laya
c'e' una testa che fa "copia il token nominato nel posto dove quel token ha un marker",
l'attacco non e' un attacco al testo ma l'uso normale di un circuito utile. E questo
cambia la difesa: non si difende un circuito utile bloccandolo (E18 blocca la nominazione),
ma si difende il **punto di lettura**. La predizione e' misurabile: le teste responsabili
dovrebbero mostrare un pattern di attenzione verso la frase proporzionale alla
sovrapposizione semantica, non alla posizione (E22b ha gia' mostrato che la posizione non
conta a distanza).

## 3. Causal scrubbing: come si verifica un'ipotesi meccanicistica

**Causal scrubbing** (Redwood Research, 2022, post su AI Alignment Forum) formalizza
"l'interpretabilita' equivalente del property-based testing": un'ipotesi dichiara quali
attivazioni si possono **ricampionare senza cambiare il comportamento**, e l'algoritmo
sostituisce tutte le attivazioni che l'ipotesi dichiara intercambiabili, poi misura se il
comportamento regge. Se l'ipotesi e' giusta, la performance non deve cambiare.

Ci serve perche' E23 formulera' ipotesi del tipo "la decisione dipende solo da X". Causal
scrubbing e' il modo di testarla in blocco invece di un intervento alla volta. Attenzione
documentata dagli autori stessi (e da un post successivo sui *practical pitfalls*): lo
scrubbing **non rifiuta** ipotesi false o incomplete per costruzione, e le ipotesi
estensionalmente equivalenti non si distinguono. Quindi e' un test di **sufficienza**, non
di verita'.

## 4. Editing distribuito

**MEMIT** (Meng et al. 2022, arXiv:2210.07229, citato in [[rome-editing]]) scala l'editing
rank-one da un fatto a migliaia, e il fatto che serva una versione distribuita e' un
risultato indiretto utile: le associazioni non stanno in un posto unico. Per noi significa
che se E23 trova un singoloato "responsabile" dell'attacco, il sosposto piu' probabile e'
che ne esista un altro di riserva.

## 5. Universalita' dei neuroni

**Universal Neurons in GPT-2** (arXiv:2401.12181) allena piu' GPT-2 da inizializzazioni
diverse e trova che solo circa **1-5%** dei neuroni sono universali, cioe' si corrispondono
fra modelli. E' un pezzo di evidenza contro il neurone come unita' d'analisi, ed e' la
ragione per cui [[unita-di-analisi]] mette i neuroni in fondo. Consequenza indiretta per la
campagna: se anche l'attacco trasferisse fra checkpoint (e lo fa), la componente condivisa
non e' probabilmente un neurone, e' piu' probabilmente una direzione o un pattern di
teste.

## 6. Il benchmark delle difese

**Formalizing and Benchmarking Prompt Injection Attacks and Defenses** (Liu et al.,
USENIX Security 2024, arXiv:2310.12815) formalizza l'attacco come task target + task
iniettato e valuta 5 attacchi e 10 difese su 10 LLM e 7 task. Ci interessa per due motivi:
il formalismo "target task / injected task" e' esattamente la struttura di E03 (la domanda
e' il task target, la frase nello stato e' il task iniettato), e il rilievo sulle difese di
rilevazione e' lo stesso che il progetto ha misurato: le difese basate su detection
perdono una frazione grande di dati compromessi **e** marcano come compromessi dati
puliti (da noi: 0.70% di falsi positivi sullo strict, 40% su una variante allentata, in
[[E18-nomination-detector]]).

## 7. Quanto e' distante la finestra

**How Far Does BERT Look At** (Guan et al., COLING 2020) clustera le mappe di attenzione
per pattern e studia le distanze. E' il riferimento piu' vicino alla domanda di
[[E22-attenzione-laya]]: quale porzione della sequenza un livello guarda davvero, e con
quale pattern. Nel nostro caso la risposta e' arrivata dai pesi (±64 token, misurato), ma
questo e' il paper da citare se il paper vuole posizionare la nostra misura di
raggiungibilita'.

## Cosa resta fuori, e perche'

- **Interpretabilita' su immagini** (saliency, GradCAM): il metodo dei sanity check viene
  da li' ([[sanity-check-salienza]]) e resta utile, ma le mappe sui token non hanno il
  problema della griglia spaziale, quindi molte conclusioni non trasferiscono.
- **Mechanistic interpretability su modelli grandi** (SAE su Claude, GPT-4): fuori scala
  per 421M parametri con una testa proprietaria, e le feature addestrate sono un artefatto
  in piu' da giustificare.
- **Editing come difesa**: promettente (SHIFT), ma richiede dati e una decisione su cosa
  togliere, che e' il passo successivo a E23, non questo.

## Note collegate
[[unita-di-analisi]] · [[circuiti-e-interventi]] · [[E22-attenzione-laya]] ·
[[E18-nomination-detector]]
