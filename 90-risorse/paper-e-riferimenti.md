---
tags: [risorse, paper]
---

# Paper e riferimenti

## Category / prodotto
1. **TypeSafe AI — Introducing System One Models & Jev** (blog, 15 set 2026). Definizione della categoria, RLCD, naming (Kahneman, Jevons). → https://typesafe.ai/blog/introducing-system-one-models-and-jev
2. **Towards AI — Introducing System One Models & Laya (and Jev)** (22 set 2026). Architettura per marker token, critiche ("era un classifier anche prima"), demo Rust, regole di prompting (frasi naturali, non JSON grezzo), limiti con input avversari. Due parti: `-2` è la più tecnica.
3. **heym.run — System One Models: Jev, Laya, and One Wire Format**. Migliore spiegazione del wire format condiviso; nota il refuso `null` vs `noul` nei tutorial di prime pagine.
4. **tanh.xyz — Can Jev be jailbroken? A prompt-injection test** (22 set 2026). Studio numerico su Laya+Jev: 14.495 call, policy variants, controlli di lunghezza. Il riferimento da battere → [[attacchi-pubblicati]] · https://tanh.xyz/research-notes/jev-prompt-manipulation
5. **navinpai — Decoding Jev** (architecture/inference/RLCD). Option-list sensitivity di Hume, limiti noti, riproduzioni della community (openjev, Mini-Jev, jev-on-a-laptop).
6. **Michał Chromiak — Typed Decision Models: Jev and Laya in Agentic AI** (17 set 2026). Tabella comparativa Jev vs Laya e i 5 boundary operativi.
7. **JevBench** (Florian S, v1.2.8): Jev 75.4 / SemIf 74.7 / djev 74.3. Righe per djev e calibrazione debole (65.4).

## Encoder e classificazione
4. **Smarter, Better, Faster, Longer (ModernBERT)** — ACL 2025 / arXiv 2412.13663. Backbone di `laya`: RoPE, attenzione alternata 1/3 global 2/3 local (window 128), GeGLU, unpadding, 8192 ctx, vocab 50368 (OLMo BPE). Sul nostro checkpoint la frazione realizzata è **10 globali su 28 = 0.357** e `local_attention=128` è una **mezza finestra** (raggiungibilità ±64 token, misurata sui pesi): `results/provenance/e22_attention_structure.json`.
5. **PangolinGuard** (HF blog, 2025) — fine-tune ModernBERT come prompt-guard: 84.72% su benchmark misto (BIPIA, NotInject, Wildguard-Benign, PINT) vs Claude 3.7 86.81%. Baseline utile per capire il potenziale di un encoder su injection.
6. **Do Encoders Suffice?** — arXiv 2606.25782 (2026). Encoder ModernBERT/Ettin come safety judge vs LLM judge su 563k pairs. **Tecniche di attacco catalogate**: single-turn, decomposition, escalation, context manipulation. Finding chiave per noi: decomposition → FNR altissimi sui piccoli, mitigato ma non eliminato dalla size.

## RL / calibrazione
7. **RLCD** (Reinforcement Learning for Calibrated Decisions) — descritto nel model card Laya e nel blog TypeSafe. Reward = strictly proper scoring rule (log + spherical + RPS), update REINFORCE con baseline di gruppo (GRPO-style), TD(λ=1.0) sui prefix per multi-turn.
8. **Temperature scaling** — refit per (question type, option count): ECE 0.466 → 0.081. Riferimento classico: Guo et al. 2017 (platt scaling/temperature scaling originali).

## Italiano / multilingue
- **Compromesso! Italian Many-Shot Jailbreaks** (Pernisi, Hovy, Röttger, ACL 2024 SRW, MilaNLP Bocconi). 418 unsafe QA italiane, many-shot su 6 LLM open. L'unico attacco strutturato in italiano che abbiamo trovato, ma su LLM generativi → [[dataset-italiano]].

## Adversarial (classici da citare)
9. **TextFooler** (Jin et al.), **BERT-Attack** (Li et al.) — perturbazioni word-level su encoder: template per le varianti di [[option-marker-injection]] in black box.
10. **HotFlip** (Ebrahimi et al.) — substitution via gradient: adatto al livello L2 di [[black-box-vs-gray-box]].
11. **Zou et al. 2023 (AdvBench/GCG)** e **Souly et al. 2024 (STRONGReject)** — cataloghi di tecniche, riadattabili a stati non-conversazionali come in [[prompt-injection-encoder]].

## Knowledge / naming
12. Kahneman, *Thinking, Fast and Slow* — System 1 / System 2.
13. Jevons, *The Coal Question* — efficienza → domanda crescente (perché il modello si chiama Jev).

## Interpretabilità (la sezione nata per E23)

Scaricati e verificati con sha256 in `60-interpretabilita/paper/`
(`uv run python tools/fetch_papers.py --verify`), una nota per paper in
`60-interpretabilita/`, concetti in [[MOC-interpretabilita]].

### Attenzione come spiegazione
14. **Jain & Wallace, Attention is not Explanation** (NAACL 2019, arXiv:1902.10186).
15. **Wiegreffe & Pinter, Attention is not not Explanation** (EMNLP 2019, arXiv:1908.04626).
16. **Serrano & Smith, Is Attention Interpretable?** (ACL 2019, arXiv:1906.03731).
17. **Abnar & Zuidema, Quantifying Attention Flow in Transformers** (ACL 2020, arXiv:2005.00928).

### Teste e struttura
18. **Michel, Levy & Neubig, Are Sixteen Heads Really Better than One?** (NeurIPS 2019, arXiv:1905.10650).
19. **Voita et al., Analyzing Multi-Head Self-Attention** (ACL 2019, arXiv:1905.09418).

### Bertology (il nostro backbone)
20. **Clark et al., What Does BERT Look At?** (BlackboxNLP 2019, arXiv:1906.04341).
21. **Tenney, Das & Pavlick, BERT Rediscovers the Classical NLP Pipeline** (ACL 2019, arXiv:1905.05950).
22. **Rogers, Kovaleva & Rumshisky, A Primer in BERTology** (TACL 2020, arXiv:2002.12327).

### Attribuzioni e controlli
23. **Adebayo et al., Sanity Checks for Saliency Maps** (NeurIPS 2018, arXiv:1810.03292).
24. **Chefer, Gur & Wolf, Transformer Interpretability Beyond Attention Visualization** (CVPR 2021, arXiv:2012.09838).

### Probe e causalità
25. **Hewitt & Liang, Designing and Interpreting Probes with Control Tasks** (EMNLP 2019, arXiv:1909.03368).
26. **Vig et al., Causal Mediation Analysis for Interpreting Neural NLP** (BlackboxNLP 2020, arXiv:2004.12265).

### Circuiti e interventi
27. **Wang et al., Interpretability in the Wild: IOI in GPT-2 small** (ICLR 2023, arXiv:2211.00593).
28. **Meng, Bau, Andonian & Belinkov, Locating and Editing Factual Associations in GPT** (NeurIPS 2022, arXiv:2202.05262).
29. **Syed, Rager & Conmy, Attribution Patching Outperforms Automated Circuit Discovery** (arXiv:2310.10348).

### Perché gli attacchi funzionano
30. **Wallace et al., Universal Adversarial Triggers** (EMNLP 2019, arXiv:1908.07125).
31. **Ilyas et al., Adversarial Examples Are Not Bugs, They Are Features** (NeurIPS 2019, arXiv:1905.02175).
32. **Wallace et al., The Instruction Hierarchy** (OpenAI, arXiv:2404.13208).

### Unità di analisi
33. **Bills et al., Language Models Can Explain Neurons in Language Models** (OpenAI 2023; GPT-2, 307.200 neuroni).
34. **Bricken et al., Towards Monosemanticity** (Transformer Circuits, Anthropic 2023).
35. **Huang, Geiger, D'Oosterlinck, Wu & Potts, Rigorously Assessing Natural Language Explanations of Neurons** (BlackboxNLP 2023, arXiv:2309.10312).
36. **Belrose et al., Eliciting Latent Predictions with the Tuned Lens** (arXiv:2303.08112).

## Note di lettura consigliate (in ordine)
1 → 2 → 6 → 4 (solo attention alternata e budget) → 7
