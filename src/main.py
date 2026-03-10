from evaluator import evaluate_predictions
from prompt_strategies import zero_shot
import json
import time
import re


def strip_prefix(text):
    """Removes 'Language: ' or 'English: ' from the start of a string."""
    # Matches any characters followed by a colon and a space at the start of the string
    return re.sub(r'^[^:]+:\s*', '', text).strip()

# Function to parse JSON dataset into a list of problems
def load_dataset(file_path="../dataset/final_modeLing.json"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        problems = []
        for problem in data.get("problems", []):
            # Extract source and target from the "data" strings (separator is a \n)
            src_ex, tgt_ex = [], []
            for item in problem['data']:
                parts = item.split('\n')
                if len(parts) == 2:
                    src_ex.append(parts[0].split(': ', 1)[-1])
                    tgt_ex.append(parts[1].split(': ', 1)[-1])
            
            problems.append({
                "name": problem['name'],
                "type": problem['type'],
                "difficulty": problem['difficulty'],
                "source_examples": src_ex,
                "target_examples": tgt_ex,
                "questions": problem['questions'],
                "answers": problem['answers']
            })

        return problems
    
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []


def main():
    print("Loading dataset...")
    dataset = load_dataset()
    
    test_problems = dataset[2:4]
    references =[]
    predictions = []

    # Baseline test
    print("Running Zero-Shot Baseline...\n")
    for problem in test_problems:
        print(f"Processing problem: {problem["name"]}")
        problem_preds = []
        problem_refs = problem['answers']

        for q_raw in problem['questions']:
            q = strip_prefix(q_raw)
            pred = zero_shot(problem,q)

            problem_preds.append(pred)
            predictions.append(pred)

            print(f"Question: {q}")
            print(f"Translation: {pred}")
            # Sleep to respect the API rate limits (we can do 30 requests per minute, 14.400 per day with gemma27b)
            time.sleep(5)

        references.extend(problem_refs)

        print()

    if predictions:
        final_metrics = evaluate_predictions(predictions, references)
        print("FINAL METRICS (Zero-Shot)")
        print(f"Total Questions: {len(predictions)}")
        print(f"BLEU: {final_metrics['BLEU']}")
        print(f"chrF: {final_metrics['chrF']}")


if __name__ == "__main__":
    main()