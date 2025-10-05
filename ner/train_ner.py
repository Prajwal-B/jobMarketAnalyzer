# file: ner/train_ner.py
"""
NER training skeleton for custom skill extraction.
Supports: mock mode for testing, real training with annotated data.

Usage:
    # Mock mode (testing)
    python ner/train_ner.py --mock --epochs 5 --output-dir models/test_ner
    
    # Real training
    python ner/train_ner.py --train-path corpus/train.spacy --dev-path corpus/dev.spacy --output-dir models/skill_ner
    
    # Evaluation
    python ner/train_ner.py --evaluate --model-path models/skill_ner/model-best --test-path corpus/test.spacy
"""

import argparse
import logging
import sys
import random
from pathlib import Path
from typing import List, Tuple, Dict
import spacy
from spacy.tokens import DocBin, Doc
from spacy.training import Example
from spacy.util import minibatch, compounding
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mock training data for demonstration
MOCK_TRAIN_DATA = [
    ("Looking for Python developer with ML experience", {"entities": [(12, 18, "SKILL"), (35, 37, "SKILL")]}),
    ("Need expert in Java and Spring Boot framework", {"entities": [(14, 18, "SKILL"), (23, 34, "SKILL")]}),
    ("React and Node.js required for frontend role", {"entities": [(0, 5, "SKILL"), (10, 17, "SKILL")]}),
    ("Data scientist with SQL and Tableau skills", {"entities": [(0, 14, "SKILL"), (20, 23, "SKILL"), (28, 35, "SKILL")]}),
    ("AWS cloud architect position available", {"entities": [(0, 3, "SKILL")]}),
    ("Seeking DevOps engineer with Docker and Kubernetes", {"entities": [(8, 14, "SKILL"), (30, 36, "SKILL"), (41, 51, "SKILL")]}),
    ("Machine learning engineer with PyTorch experience", {"entities": [(0, 16, "SKILL"), (31, 38, "SKILL")]}),
    ("Backend developer: Python, Django, PostgreSQL", {"entities": [(19, 25, "SKILL"), (27, 33, "SKILL"), (35, 45, "SKILL")]}),
    ("Frontend: React, TypeScript, HTML, CSS", {"entities": [(10, 15, "SKILL"), (17, 27, "SKILL"), (29, 33, "SKILL"), (35, 38, "SKILL")]}),
    ("Data engineer with Apache Spark and Airflow", {"entities": [(0, 13, "SKILL"), (19, 31, "SKILL"), (36, 43, "SKILL")]}),
    ("Mobile developer: iOS, Android, React Native", {"entities": [(18, 21, "SKILL"), (23, 30, "SKILL"), (32, 44, "SKILL")]}),
    ("Need QA engineer with Selenium and pytest", {"entities": [(5, 7, "SKILL"), (22, 30, "SKILL"), (35, 41, "SKILL")]}),
    ("Cybersecurity analyst with penetration testing", {"entities": [(0, 13, "SKILL"), (27, 46, "SKILL")]}),
    ("Blockchain developer: Solidity and Ethereum", {"entities": [(0, 10, "SKILL"), (22, 30, "SKILL"), (35, 43, "SKILL")]}),
    ("Business analyst with Excel and Power BI", {"entities": [(0, 16, "SKILL"), (22, 27, "SKILL"), (32, 40, "SKILL")]}),
    ("Full stack: JavaScript, MongoDB, Express", {"entities": [(12, 22, "SKILL"), (24, 31, "SKILL"), (33, 40, "SKILL")]}),
    ("AI researcher with TensorFlow and deep learning", {"entities": [(0, 2, "SKILL"), (19, 29, "SKILL"), (34, 47, "SKILL")]}),
    ("Product manager with Agile and Scrum experience", {"entities": [(21, 26, "SKILL"), (31, 36, "SKILL")]}),
    ("UI/UX designer: Figma, Sketch, Adobe XD", {"entities": [(0, 5, "SKILL"), (16, 21, "SKILL"), (23, 29, "SKILL"), (31, 39, "SKILL")]}),
    ("Network engineer with Cisco and routing", {"entities": [(0, 7, "SKILL"), (22, 27, "SKILL"), (32, 39, "SKILL")]}),
]


def create_mock_data(num_examples: int = 20) -> List[Tuple[str, Dict]]:
    """
    Generate mock training data for testing.
    
    Args:
        num_examples: Number of examples to generate
        
    Returns:
        List of (text, annotations) tuples
    """
    logger.info(f"Generating {num_examples} mock training examples")
    
    # Use predefined mock data
    data = MOCK_TRAIN_DATA[:num_examples]
    
    # Shuffle for variety
    random.shuffle(data)
    
    return data


def convert_to_docbin(nlp: spacy.Language, data: List[Tuple[str, Dict]], 
                      output_path: str) -> None:
    """
    Convert training data to spaCy DocBin format.
    
    Args:
        nlp: spaCy language model
        data: List of (text, annotations) tuples
        output_path: Path to save DocBin file
    """
    logger.info(f"Converting {len(data)} examples to DocBin format")
    
    doc_bin = DocBin()
    
    for text, annots in data:
        doc = nlp.make_doc(text)
        ents = []
        
        for start, end, label in annots["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is None:
                logger.warning(f"Skipping invalid span: '{text[start:end]}' in '{text}'")
            else:
                ents.append(span)
        
        doc.ents = ents
        doc_bin.add(doc)
    
    doc_bin.to_disk(output_path)
    logger.info(f"Saved DocBin to {output_path}")


def train_ner(train_path: str, dev_path: str, output_dir: str,
              epochs: int = 30, batch_size: int = 8, dropout: float = 0.2) -> None:
    """
    Train custom NER model for skill extraction.
    
    Args:
        train_path: Path to training DocBin
        dev_path: Path to development DocBin
        output_dir: Directory to save model
        epochs: Number of training epochs
        batch_size: Training batch size
        dropout: Dropout rate
    """
    logger.info("=" * 60)
    logger.info("NER TRAINING")
    logger.info("=" * 60)
    
    # Create blank English model
    logger.info("Creating blank spaCy model")
    nlp = spacy.blank("en")
    
    # Add NER pipe
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Add SKILL label
    ner.add_label("SKILL")
    
    # Load training data
    logger.info(f"Loading training data from {train_path}")
    train_docbin = DocBin().from_disk(train_path)
    train_docs = list(train_docbin.get_docs(nlp.vocab))
    
    # Load dev data if available
    dev_docs = []
    if dev_path and Path(dev_path).exists():
        logger.info(f"Loading dev data from {dev_path}")
        dev_docbin = DocBin().from_disk(dev_path)
        dev_docs = list(dev_docbin.get_docs(nlp.vocab))
    
    # Create training examples
    train_examples = []
    for doc in train_docs:
        train_examples.append(Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}))
    
    logger.info(f"Training on {len(train_examples)} examples")
    if dev_docs:
        logger.info(f"Validation on {len(dev_docs)} examples")
    
    # Initialize model
    logger.info("Initializing model")
    nlp.initialize(lambda: train_examples)
    
    # Training loop
    logger.info(f"Training for {epochs} epochs")
    best_score = 0.0
    patience_counter = 0
    patience = 5
    
    for epoch in range(epochs):
        random.shuffle(train_examples)
        losses = {}
        
        # Batch training
        batches = minibatch(train_examples, size=compounding(4.0, batch_size, 1.001))
        
        for batch in batches:
            nlp.update(batch, drop=dropout, losses=losses)
        
        # Evaluate on dev set
        if dev_docs:
            scores = evaluate_ner(nlp, dev_docs)
            f1_score = scores['f1']
            
            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {losses.get('ner', 0.0):.4f} - "
                       f"P: {scores['precision']:.3f} R: {scores['recall']:.3f} F1: {f1_score:.3f}")
            
            # Save best model
            if f1_score > best_score:
                best_score = f1_score
                patience_counter = 0
                output_path = Path(output_dir) / "model-best"
                output_path.mkdir(parents=True, exist_ok=True)
                nlp.to_disk(output_path)
                logger.info(f"Saved best model (F1: {best_score:.3f}) to {output_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        else:
            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {losses.get('ner', 0.0):.4f}")
    
    # Save final model
    output_path = Path(output_dir) / "model-final"
    output_path.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_path)
    logger.info(f"Saved final model to {output_path}")
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


def evaluate_ner(nlp: spacy.Language, docs: List[Doc]) -> Dict[str, float]:
    """
    Evaluate NER model performance.
    
    Args:
        nlp: Trained spaCy model
        docs: List of gold-standard docs
        
    Returns:
        Dictionary of evaluation metrics
    """
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    for gold_doc in docs:
        pred_doc = nlp(gold_doc.text)
        
        gold_ents = {(ent.start_char, ent.end_char, ent.label_) for ent in gold_doc.ents}
        pred_ents = {(ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents}
        
        tp += len(gold_ents & pred_ents)
        fp += len(pred_ents - gold_ents)
        fn += len(gold_ents - pred_ents)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train custom NER model for skill extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mock mode for testing
  python ner/train_ner.py --mock --epochs 5 --output-dir models/test_ner
  
  # Real training
  python ner/train_ner.py --train-path corpus/train.spacy --dev-path corpus/dev.spacy --output-dir models/skill_ner
  
  # Evaluation
  python ner/train_ner.py --evaluate --model-path models/skill_ner/model-best --test-path corpus/test.spacy
        """
    )
    
    parser.add_argument('--mock', action='store_true',
                       help='Use mock data for testing')
    parser.add_argument('--train-path', type=str,
                       help='Path to training DocBin file')
    parser.add_argument('--dev-path', type=str,
                       help='Path to dev DocBin file')
    parser.add_argument('--test-path', type=str,
                       help='Path to test DocBin file (for evaluation)')
    parser.add_argument('--output-dir', type=str, default='models/skill_ner',
                       help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate mode')
    parser.add_argument('--model-path', type=str,
                       help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    try:
        if args.evaluate:
            # Evaluation mode
            if not args.model_path or not args.test_path:
                logger.error("Evaluation requires --model-path and --test-path")
                sys.exit(1)
            
            logger.info(f"Loading model from {args.model_path}")
            nlp = spacy.load(args.model_path)
            
            logger.info(f"Loading test data from {args.test_path}")
            test_docbin = DocBin().from_disk(args.test_path)
            test_docs = list(test_docbin.get_docs(nlp.vocab))
            
            logger.info("Evaluating model...")
            scores = evaluate_ner(nlp, test_docs)
            
            logger.info("=" * 60)
            logger.info("EVALUATION RESULTS")
            logger.info("=" * 60)
            logger.info(f"Precision: {scores['precision']:.4f}")
            logger.info(f"Recall:    {scores['recall']:.4f}")
            logger.info(f"F1-Score:  {scores['f1']:.4f}")
            logger.info(f"True Positives:  {scores['tp']}")
            logger.info(f"False Positives: {scores['fp']}")
            logger.info(f"False Negatives: {scores['fn']}")
            logger.info("=" * 60)
            
        elif args.mock:
            # Mock training mode
            logger.info("Running in MOCK mode with synthetic data")
            
            # Generate mock data
            mock_data = create_mock_data(num_examples=20)
            train_data = mock_data[:16]  # 80% train
            dev_data = mock_data[16:]    # 20% dev
            
            # Create temporary directory
            temp_dir = Path(args.output_dir) / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to DocBin
            nlp = spacy.blank("en")
            train_path = temp_dir / "mock_train.spacy"
            dev_path = temp_dir / "mock_dev.spacy"
            
            convert_to_docbin(nlp, train_data, str(train_path))
            convert_to_docbin(nlp, dev_data, str(dev_path))
            
            # Train
            train_ner(
                train_path=str(train_path),
                dev_path=str(dev_path),
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                dropout=args.dropout
            )
            
            logger.info("\nMock training complete! This was a demonstration only.")
            logger.info("For real training, prepare annotated data and use --train-path")
            
        else:
            # Real training mode
            if not args.train_path:
                logger.error("Training requires --train-path (or use --mock for testing)")
                parser.print_help()
                sys.exit(1)
            
            if not Path(args.train_path).exists():
                logger.error(f"Training file not found: {args.train_path}")
                sys.exit(1)
            
            train_ner(
                train_path=args.train_path,
                dev_path=args.dev_path,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                dropout=args.dropout
            )
        
    except Exception as e:
        logger.error(f"NER training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()