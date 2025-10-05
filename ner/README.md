# file: ner/README.md

# NER Training Module for Skill Extraction

## Overview

This module provides a skeleton for training a custom Named Entity Recognition (NER) model to extract skills from job descriptions. While the main pipeline uses PhraseMatcher and embeddings, custom NER models can improve performance on:

- Skills with context-dependent meanings
- Multi-word skill expressions not in the lexicon
- Domain-specific terminology
- Skills mentioned in varied phrasings

## When to Use Custom NER

**Use custom NER when:**
- You have annotated training data (>500 examples recommended)
- Skills appear in varied contexts requiring semantic understanding
- You need to capture emerging skills not in the lexicon
- You want end-to-end learned representations

**Stick with PhraseMatcher when:**
- You have a comprehensive skill lexicon
- Skills are mentioned explicitly
- You need fast, interpretable extraction
- You have limited annotated data

## Training Data Format

The NER trainer expects data in spaCy's binary format or JSON format:

### JSON Format (for annotation)
```json
[
  {
    "text": "Looking for Python developer with experience in machine learning",
    "entities": [
      {"start": 12, "end": 18, "label": "SKILL"},
      {"start": 48, "end": 66, "label": "SKILL"}
    ]
  }
]
spaCy Binary Format
Use spaCy's DocBin format for efficient training:
pythonimport spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")
doc_bin = DocBin()

# Add annotated documents
for text, annots in training_data:
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annots["entities"]:
        span = doc.char_span(start, end, label=label)
        if span:
            ents.append(span)
    doc.ents = ents
    doc_bin.add(doc)

doc_bin.to_disk("./corpus/train.spacy")
Creating Training Data
Option 1: Manual Annotation
Use annotation tools:

Prodigy (commercial): https://prodi.gy/
Label Studio (open-source): https://labelstud.io/
Doccano (open-source): https://github.com/doccano/doccano

Option 2: Weak Supervision
Bootstrap from existing lexicon:
python# Use PhraseMatcher to create weak labels
from scripts.skill_extractor import SkillExtractor

extractor = SkillExtractor("skills/skills_lexicon.csv")

for job_desc in unlabeled_data:
    skills = extractor.extract_with_phrases(job_desc)
    # Convert to entity annotations
    # Review and correct manually
Option 3: Active Learning

Train initial model on small dataset
Use model to predict on unlabeled data
Manually review predictions with low confidence
Add to training set and retrain

Training Process
1. Prepare Data
bash# Split into train/dev/test
python ner/train_ner.py --prepare-data \
    --input annotations.json \
    --output-dir corpus/
2. Train Model
bash# Full training
python ner/train_ner.py \
    --train-path corpus/train.spacy \
    --dev-path corpus/dev.spacy \
    --output-dir models/skill_ner \
    --epochs 30 \
    --batch-size 8

# Quick test with mock data
python ner/train_ner.py --mock --epochs 5
3. Evaluate Model
bashpython ner/train_ner.py --evaluate \
    --model-path models/skill_ner/model-best \
    --test-path corpus/test.spacy
4. Use Trained Model
pythonimport spacy

nlp = spacy.load("models/skill_ner/model-best")
doc = nlp("We need Python and machine learning experts")

for ent in doc.ents:
    if ent.label_ == "SKILL":
        print(f"Skill: {ent.text}")
Model Configuration
Edit ner/config.cfg to customize:
ini[training]
max_epochs = 30
patience = 5
batch_size = 8
dropout = 0.2

[model]
@architectures = "spacy.TransitionBasedParser.v2"
hidden_width = 128
Performance Optimization
For Small Datasets (<1000 examples)

Use pre-trained transformers: en_core_web_trf
Enable transfer learning
High dropout (0.3-0.5)
Early stopping (patience=3)

For Large Datasets (>5000 examples)

Use efficient architectures: spacy.TransitionBasedParser
Larger batch sizes (16-32)
More epochs (50+)
Lower dropout (0.1-0.2)

For Production Deployment

Export to ONNX for faster inference
Quantize model weights
Use GPU acceleration
Implement batching

Integration with Main Pipeline
Replace PhraseMatcher in skill_extractor.py:
python# Load custom NER model
self.ner_model = spacy.load("models/skill_ner/model-best")

def extract_with_ner(self, text: str) -> Set[str]:
    doc = self.ner_model(text)
    skills = {ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL"}
    return skills
Evaluation Metrics
The trainer reports:

Precision: % of predicted skills that are correct
Recall: % of actual skills that were found
F1-Score: Harmonic mean of precision and recall
Entity-level accuracy: Exact boundary matches

Target metrics:

Precision: >0.85
Recall: >0.80
F1-Score: >0.82

Common Issues
Low Recall

Add more diverse training examples
Include negative examples (non-skills)
Use data augmentation

Low Precision

Review annotation consistency
Add hard negative examples
Increase model capacity

Overfitting

Increase dropout
Add more training data
Use early stopping
Reduce model size

Advanced Techniques
Multi-task Learning
Train on multiple related tasks simultaneously:

Skill extraction (NER)
Skill category classification
Experience level prediction

Contextual Embeddings
Use transformer-based models for better context:

en_core_web_trf (spaCy + transformers)
Custom BERT fine-tuning
Domain-adapted language models

Ensemble Methods
Combine multiple approaches:

PhraseMatcher (high precision)
Custom NER (good recall)
Embedding similarity (fuzzy matching)

Vote or union results based on confidence scores.
Resources

spaCy Documentation: https://spacy.io/usage/training
Explosion Blog: https://explosion.ai/blog
NER Best Practices: https://www.kdnuggets.com/2020/06/named-entity-recognition-best-practices.html

Next Steps

Collect 500+ annotated examples
Train initial model with --mock flag to verify setup
Evaluate on held-out test set
Iterate on difficult cases
Deploy to production pipeline

For questions, see inline documentation in train_ner.py.