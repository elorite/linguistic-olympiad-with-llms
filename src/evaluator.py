import sacrebleu

def evaluate_predictions(predictions, references):
    """Calculates BLEU and chrF metrics."""
    # sacrebleu expects references as a list of lists
    refs = [[ref] for ref in references]
    
    bleu = sacrebleu.corpus_bleu(predictions, refs)
    chrf = sacrebleu.corpus_chrf(predictions, refs)
    
    return {
        "BLEU": round(bleu.score, 3),
        "chrF": round(chrf.score, 3)
    }