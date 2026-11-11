"""Scarica la letteratura di interpretabilita' usata da E23 e la fida con sha256.

I paper vivono in `60-interpretabilita/paper/` (PDF arXiv, o il `text/html` delle
due voci pubblicate fuori da arXiv) e il manifest accanto registra
`sha256`, `size` e la citazione. Il manifest e' l'unica prova che un PDF su
disco sia quello che il vault cita: senza, un download troncato o una versione
diversa della stessa pagina sarebbero indistinguibili da un paper giusto, che e'
la stessa classe di difetto del GGUF sparse documentato in `tools/fetch_gguf.py`.

Uso:
    uv run python tools/fetch_papers.py            # scarica i mancanti
    uv run python tools/fetch_papers.py --verify    # riscrive il manifest e controlla
    uv run python tools/fetch_papers.py --force     # riscarica tutto
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "60-interpretabilita" / "paper"
MANIFEST = PAPERS / "manifest.json"
USER_AGENT = "layaadv/papers (+adversarial-laya)"

# slug -> (arxiv id, url template, titolo, autori, venue)
# Le URL di arXiv sono `pdf/<id>` e il tag di versione e' omesso: arXiv serve
# l'ultima versione e la pagina astratta riporta qual e', quindi l'hash cambia se
# un paper viene aggiornato. Per questo il manifest registra anche l'id e la data
# di fetch: un PDF rifatto non e' silenzioso.
REGISTRY: dict[str, dict[str, str]] = {
    "jain-wallace-attention-not-explanation": {
        "arxiv": "1902.10186",
        "url": "https://arxiv.org/pdf/1902.10186",
        "titolo": "Attention is not Explanation",
        "autori": "Jain, Wallace",
        "venue": "NAACL-HLT 2019",
        "famiglia": "attenzione-come-spiegazione",
    },
    "wiegreffe-pinter-attention-not-not-explanation": {
        "arxiv": "1908.04626",
        "url": "https://arxiv.org/pdf/1908.04626",
        "titolo": "Attention is not not Explanation",
        "autori": "Wiegreffe, Pinter",
        "venue": "EMNLP-IJCNLP 2019",
        "famiglia": "attenzione-come-spiegazione",
    },
    "serrano-smith-is-attention-interpretable": {
        "arxiv": "1906.03731",
        "url": "https://arxiv.org/pdf/1906.03731",
        "titolo": "Is Attention Interpretable?",
        "autori": "Serrano, Smith",
        "venue": "ACL 2019",
        "famiglia": "attenzione-come-spiegazione",
    },
    "abnar-zuidema-attention-flow": {
        "arxiv": "2005.00928",
        "url": "https://arxiv.org/pdf/2005.00928",
        "titolo": "Quantifying Attention Flow in Transformers",
        "autori": "Abnar, Zuidema",
        "venue": "ACL 2020",
        "famiglia": "attenzione-come-spiegazione",
    },
    "michel-levy-neubig-sixteen-heads": {
        "arxiv": "1905.10650",
        "url": "https://arxiv.org/pdf/1905.10650",
        "titolo": "Are Sixteen Heads Really Better than One?",
        "autori": "Michel, Levy, Neubig",
        "venue": "NeurIPS 2019",
        "famiglia": "teste-e-struttura",
    },
    "voita-analyzing-multihead-self-attention": {
        "arxiv": "1905.09418",
        "url": "https://arxiv.org/pdf/1905.09418",
        "titolo": "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, "
                  "the Rest Can Be Pruned",
        "autori": "Voita, Talbot, Moiseev, Sennrich, Titov",
        "venue": "ACL 2019",
        "famiglia": "teste-e-struttura",
    },
    "clark-khandelwal-what-does-bert-look-at": {
        "arxiv": "1906.04341",
        "url": "https://arxiv.org/pdf/1906.04341",
        "titolo": "What Does BERT Look At? An Analysis of BERT's Attention",
        "autori": "Clark, Khandelwal, Levy, Manning",
        "venue": "BlackboxNLP@ACL 2019",
        "famiglia": "bertology-encoder-mascherati",
    },
    "tenney-das-bert-rediscovers-pipeline": {
        "arxiv": "1905.05950",
        "url": "https://arxiv.org/pdf/1905.05950",
        "titolo": "BERT Rediscovers the Classical NLP Pipeline",
        "autori": "Tenney, Das, Pavlick",
        "venue": "ACL 2019",
        "famiglia": "bertology-encoder-mascherati",
    },
    "rogers-kovaleva-bertology-primer": {
        "arxiv": "2002.12327",
        "url": "https://arxiv.org/pdf/2002.12327",
        "titolo": "A Primer in BERTology: What We Know About How BERT Works",
        "autori": "Rogers, Kovaleva, Rumshisky",
        "venue": "TACL 2020",
        "famiglia": "bertology-encoder-mascherati",
    },
    "adebayo-sanity-checks-saliency": {
        "arxiv": "1810.03292",
        "url": "https://arxiv.org/pdf/1810.03292",
        "titolo": "Sanity Checks for Saliency Maps",
        "autori": "Adebayo, Gilmer, Muelly, Goodfellow, Hardt, Kim",
        "venue": "NeurIPS 2018",
        "famiglia": "attribuzioni-e-sanity-check",
    },
    "chefer-transformer-interpretability": {
        "arxiv": "2012.09838",
        "url": "https://arxiv.org/pdf/2012.09838",
        "titolo": "Transformer Interpretability Beyond Attention Visualization",
        "autori": "Chefer, Gur, Wolf",
        "venue": "CVPR 2021",
        "famiglia": "attribuzioni-e-sanity-check",
    },
    "hewitt-liang-control-tasks": {
        "arxiv": "1909.03368",
        "url": "https://arxiv.org/pdf/1909.03368",
        "titolo": "Designing and Interpreting Probes with Control Tasks",
        "autori": "Hewitt, Liang",
        "venue": "EMNLP-IJCNLP 2019",
        "famiglia": "probe-e-controfattuali",
    },
    "vig-causal-mediation-gender-bias": {
        "arxiv": "2004.12265",
        "url": "https://arxiv.org/pdf/2004.12265",
        "titolo": "Causal Mediation Analysis for Interpreting Neural NLP: "
                  "The Case of Gender Bias",
        "autori": "Vig, Gehrmann, Belinkov, Qian, Nevo, Singer, Shieber",
        "venue": "BlackboxNLP 2020",
        "famiglia": "probe-e-controfattuali",
    },
    "meng-bau-rome-locating-editing": {
        "arxiv": "2202.05262",
        "url": "https://arxiv.org/pdf/2202.05262",
        "titolo": "Locating and Editing Factual Associations in GPT",
        "autori": "Meng, Bau, Andonian, Belinkov",
        "venue": "NeurIPS 2022",
        "famiglia": "circuiti-e-interventi",
    },
    "wang-ioi-circuit-gpt2-small": {
        "arxiv": "2211.00593",
        "url": "https://arxiv.org/pdf/2211.00593",
        "titolo": "Interpretability in the Wild: A Circuit for Indirect Object "
                  "Identification in GPT-2 Small",
        "autori": "Wang, Variengien, Conmy, Shlegeris, Steinhardt",
        "venue": "ICLR 2023",
        "famiglia": "circuiti-e-interventi",
    },
    "syed-attribution-patching": {
        "arxiv": "2310.10348",
        "url": "https://arxiv.org/pdf/2310.10348",
        "titolo": "Attribution Patching Outperforms Automated Circuit Discovery",
        "autori": "Syed, Rager, Conmy",
        "venue": "arXiv 2023",
        "famiglia": "circuiti-e-interventi",
    },
    "wallace-universal-adversarial-triggers": {
        "arxiv": "1908.07125",
        "url": "https://arxiv.org/pdf/1908.07125",
        "titolo": "Universal Adversarial Triggers for Attacking and Analyzing NLP",
        "autori": "Wallace, Feng, Kandpal, Gardner, Singh",
        "venue": "EMNLP-IJCNLP 2019",
        "famiglia": "perche-gli-attacchi-funzionano",
    },
    "ilyas-adversarial-examples-features": {
        "arxiv": "1905.02175",
        "url": "https://arxiv.org/pdf/1905.02175",
        "titolo": "Adversarial Examples Are Not Bugs, They Are Features",
        "autori": "Ilyas, Santurkar, Tsipras, Engstrom, Tran, Madry",
        "venue": "NeurIPS 2019",
        "famiglia": "perche-gli-attacchi-funzionano",
    },
    "wallace-instruction-hierarchy": {
        "arxiv": "2404.13208",
        "url": "https://arxiv.org/pdf/2404.13208",
        "titolo": "The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions",
        "autori": "Wallace, Xiao, Leike, Weng, Heidecke, Beutel",
        "venue": "arXiv 2024 (OpenAI)",
        "famiglia": "perche-gli-attacchi-funzionano",
    },
    "huang-geiger-neuron-explanations-audit": {
        "arxiv": "2309.10312",
        "url": "https://arxiv.org/pdf/2309.10312",
        "titolo": "Rigorously Assessing Natural Language Explanations of Neurons",
        "autori": "Huang, Geiger, D'Oosterlinck, Wu, Potts",
        "venue": "BlackboxNLP@EMNLP 2023",
        "famiglia": "unita-di-analisi",
    },
    "belrose-tuned-lens": {
        "arxiv": "2303.08112",
        "url": "https://arxiv.org/pdf/2303.08112",
        "titolo": "Eliciting Latent Predictions from Transformers with the Tuned Lens",
        "autori": "Belrose, Furman, Smith, Halawi, Ostrovsky, McKinney, Biderman, Steinhardt",
        "venue": "arXiv 2023",
        "famiglia": "unita-di-analisi",
    },
}

# Le due voci pubblicate fuori da arXiv: la pagina e' il documento, e va salvata
# per intero (l'HTML di transformer-circuits e' autosufficiente).
HTML_REGISTRY: dict[str, dict[str, str]] = {
    "bills-openai-language-models-explain-neurons": {
        "url": "https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html",
        "titolo": "Language Models Can Explain Neurons in Language Models",
        "autori": "Bills, Cammarata, Mossing, Tillman, Gao, Goh, Sutskever, Leike, Wu, Saunders",
        "venue": "OpenAI 2023",
        "famiglia": "unita-di-analisi",
    },
    "bricken-towards-monosemanticity": {
        "url": "http://transformer-circuits.pub/2023/monosemantic-features/index.html",
        "titolo": "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning",
        "autori": "Bricken, Templeton, Batson, Chen, Jermyn, Conerly, Turner, Anil, Denison, "
                  "Askell, Lasenby, Wu, Kravec, Schiefer, Maxwell, Joseph, Tamkin, Nguyen, "
                  "McLean, Burke, Hume, Carter, Henighan, Olah",
        "venue": "Transformer Circuits Thread 2023 (Anthropic)",
        "famiglia": "unita-di-analisi",
    },
}

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, *, attempts: int = 3) -> int:
    """Download `url` to `dest`, returning the byte count. Retries with backoff."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                blob = response.read()
            dest.write_bytes(blob)
            return len(blob)
        except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover - rete
            last = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


def looks_like_pdf(path: Path) -> bool:
    head = path.read_bytes()[:5]
    return head == b"%PDF-"


def verify_existing(manifest: dict, problems: list[str]) -> int:
    checked = 0
    for slug, entry in manifest["paper"].items():
        path = PAPERS / f"{slug}.{entry['ext']}"
        if not path.exists():
            problems.append(f"{slug}: manca {path.name}")
            continue
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            problems.append(f"{slug}: sha256 diverso da quello del manifest")
            continue
        if entry["ext"] == "pdf" and not looks_like_pdf(path):
            problems.append(f"{slug}: l'estensione e' pdf ma il contenuto non lo e'")
            continue
        checked += 1
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="riscarica anche quello che c'e'")
    parser.add_argument("--verify", action="store_true", help="solo verifica, non scarica")
    args = parser.parse_args()

    PAPERS.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"paper": {}}
    if MANIFEST.exists() and not args.force:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    for slug, meta in list(REGISTRY.items()) + list(HTML_REGISTRY.items()):
        ext = "html" if slug in HTML_REGISTRY else "pdf"
        dest = PAPERS / f"{slug}.{ext}"
        entry = dict(meta)
        entry["ext"] = ext
        if dest.exists() and not args.force and slug in manifest["paper"]:
            entry["sha256"] = manifest["paper"][slug]["sha256"]
            entry["bytes"] = manifest["paper"][slug]["bytes"]
            manifest["paper"][slug] = entry
            continue
        if args.verify:
            manifest["paper"][slug] = entry
            continue
        size = fetch(meta["url"], dest)
        if ext == "pdf" and not looks_like_pdf(dest):
            raise RuntimeError(f"{slug}: il download non e' un PDF ({size} byte)")
        if size < 20_000:
            raise RuntimeError(f"{slug}: {size} byte, troppo pochi per essere il paper")
        entry["bytes"] = size
        entry["sha256"] = sha256_file(dest)
        manifest["paper"][slug] = entry
        print(f"{slug}: {size} byte")

    if args.verify and not args.force:
        problems: list[str] = []
        checked = verify_existing(manifest, problems)
        for problem in problems:
            print(f"PROBLEMA {problem}", file=sys.stderr)
        print(f"{checked}/{len(manifest['paper'])} file verificati")
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return 1 if problems else 0

    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest: {MANIFEST} ({len(manifest['paper'])} voci)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
