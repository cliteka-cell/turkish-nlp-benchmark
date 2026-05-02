# Turkish NLP Benchmark: Cross-Lingual Transfer Study

🔗 **[Try the live demo](https://huggingface.co/spaces/Reshiman/turkish-nlp-benchmark)**

A three-way benchmark comparing how different BERT variants handle Turkish sentiment
analysis — from complete failure to near-human performance. The demo lets you test
all three models on the same sentence in real time.

## The Research Question

How much does language specialization matter in transformer models?
Can a multilingual model match a Turkish-specific one on a Turkish task?

## Results

| Model | Type | Accuracy | F1 | Notes |
|-------|------|----------|----|-------|
| English DistilBERT | Monolingual EN (zero-shot) | 0.4171 | — | Worse than random |
| mBERT | Multilingual (104 languages) | 0.8436 | 0.8736 | Strong recovery |
| **BERTurk** | **Monolingual TR** | **0.9431** | **0.9524** | Best by far |

**Random baseline: 0.50**

BERTurk outperforms mBERT by +9.95 accuracy points on identical training data
and identical training conditions. Language specialization wins decisively.

## The Story in Three Acts

**Act 1 — Zero-shot failure (0.4171)**
An English DistilBERT model tested on Turkish without any Turkish training.
It doesn't just fail — it scores *below* random chance. Turkish tokens get
split into meaningless subword fragments by the English WordPiece tokenizer,
which the model interprets as systematic signal, producing inverted predictions.
Confidently wrong is more dangerous than uncertain.

**Act 2 — Multilingual recovery (0.8436)**
mBERT pretrained on 104 languages fine-tuned on 4,486 Turkish reviews.
A massive +42.6% recovery over zero-shot. Cross-lingual transfer works
because languages share subword patterns — mBERT learned Turkish structure
even without Turkish-specific pretraining.

**Act 3 — Monolingual dominance (0.9431)**
BERTurk pretrained exclusively on Turkish Wikipedia, news, and web crawl data.
Same fine-tuning setup as mBERT — same dataset, same hyperparameters, same epochs.
Nearly 10 points better. When a model dedicates all its capacity to one language,
it learns deeper morphological and syntactic patterns than a multilingual model
spreading capacity across 104 languages.

## Dataset

[Turkish Sentiment Dataset](https://huggingface.co/datasets/sepidmnorozy/Turkish_sentiment)
— 4,486 Turkish restaurant reviews annotated with positive/negative sentiment.

| Split | Size | Negative | Positive |
|-------|------|----------|----------|
| Train | 4,486 | 48.5% | 51.5% |
| Validation | 105 | — | — |
| Test | 211 | — | — |

## Models Used

| Model | HuggingFace ID | Pretraining Data |
|-------|---------------|-----------------|
| English DistilBERT | `distilbert-base-uncased-finetuned-sst-2-english` | English only |
| mBERT | `bert-base-multilingual-cased` | 104 languages |
| BERTurk | `dbmdz/bert-base-turkish-cased` | Turkish only |

Fine-tuned models: [Reshiman/turkish-sentiment](https://huggingface.co/Reshiman/turkish-sentiment) · [Reshiman/berturk-sentiment](https://huggingface.co/Reshiman/berturk-sentiment)

## Training Configuration

All fine-tuned models used identical settings for a fair comparison:

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-5 |
| Batch size | 32 |
| Epochs | 3 |
| Weight decay | 0.01 |
| Max sequence length | 128 |

## Linguistic Observations

**Turkish is agglutinative:** A single Turkish word can carry the meaning
of an entire English sentence — *"yapamayacaklardandım"* = *"I was among
those who would not be able to do it."* English WordPiece tokenization
splits these long words into meaningless fragments, losing the morphological
structure entirely. BERTurk's tokenizer was trained on Turkish text and
handles this more gracefully.

**Casing matters in Turkish:** BERTurk uses a cased model — it distinguishes
uppercase and lowercase. This matters for Turkish proper nouns and the
Turkish-specific characters `İ/i` and `I/ı` which have different meanings.
mBERT is also cased, but English DistilBERT is uncased — another disadvantage
for zero-shot Turkish.

**Multilingual capacity dilution:** mBERT splits its model capacity across
104 languages. For high-resource languages like English, this is fine.
For Turkish — a moderately resourced language — BERTurk's dedicated
pretraining corpus produces meaningfully richer representations, as the
+9.95% accuracy gap demonstrates.

**Domain mismatch compounds cross-lingual mismatch:** Both the zero-shot
and multilingual models struggle more on casual, colloquial Turkish than
formal text. Restaurant reviews contain slang, abbreviations, and informal
constructions underrepresented in Wikipedia-based pretraining.

## How to Run

Open the notebook in Google Colab and run all cells.
No local setup required.

## Tech Stack

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [HuggingFace Datasets](https://github.com/huggingface/datasets)
- Python 3.x · PyTorch · Gradio
