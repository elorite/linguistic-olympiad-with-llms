from evaluator import evaluate_predictions
from prompt_strategies import zero_shot, cot_linguistic, custom, generator_critic
import json
import time
import re
import argparse


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
    
def choose_problems(problems, language="All", difficulty="All", problem_type="All"):
    filtered_problems = problems
    if language != "All":
        filtered_problems = [problem for problem in filtered_problems if re.sub(r'\d+$', '', problem['name']).replace(" ", "") == language]
    if difficulty != "All":
        difficulty = int(difficulty)
        filtered_problems = [problem for problem in filtered_problems if problem['difficulty'] == difficulty]
    if problem_type != "All":
        filtered_problems = [problem for problem in filtered_problems if problem_type in problem['type']]
    return filtered_problems

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="../dataset/final_modeLing.json", help='Path to the dataset JSON file')
    parser.add_argument('--task', type=str, default="baseline", help='Task to run: "baseline", "cot", "custom", "generator-critic" or "comparison"')
    # TODO: update the --task argument to accept more tasks
    parser.add_argument('--language', type=str, default="All", help='Language to filter problems by (look at the dataset to see available languages, or use "All" for no filtering)')
    parser.add_argument('--difficulty', type=str, default="All", help='Difficulty level to filter problems by (1, 2, 3, 4, 5 or "All" for no filtering)')
    parser.add_argument('--type', type=str, default="All", help='Problem type to filter by (POSS, ORDER, NOUN-ADJ, SEM or "All" for no filtering)')
    args = parser.parse_args()

    #print("Loading dataset...")
    dataset = load_dataset(args.dataset_path)
    test_problems = choose_problems(dataset, language=args.language, difficulty=args.difficulty, problem_type=args.type)
    print(f"Configuration:\nLanguage: {args.language}\nDifficulty: {args.difficulty}\nProblem Type: {args.type}\nPrompt Strategy: {args.task}")
    print(f"\nProblems selected for evaluation: {len(test_problems)}\n")
    references =[]
    predictions = []
    results = {}

    print(f"Running {args.task} task...\n")
    for problem in test_problems:
        print(f"Processing problem: {problem['name']} - Type: {problem['type']} - Difficulty: {problem['difficulty']}")
        problem_preds = []
        problem_refs = [strip_prefix(a) for a in problem['answers']]

        for q_raw in problem['questions']:
            q = strip_prefix(q_raw)
            if args.task == "baseline":
                pred = zero_shot(problem,q)
            elif args.task == "cot":
                pred = cot_linguistic(problem, q)
            elif args.task == "custom":
                pred = custom(problem,q)
            elif args.task == "generator-critic":
                pred = generator_critic(problem, q)
            elif args.task == "comparison":
                print("Comparison of strategies not implemented yet.")
                pred = zero_shot(problem,q)
            else:
                print(f"Unknown task: {args.task}.")
                raise ValueError

            problem_preds.append(pred)
            predictions.append(pred)

            print(f"Question: {q}")
            print(f"Translation: {pred}")
            # Sleep to respect the API rate limits 
            time.sleep(5)

        references.extend(problem_refs)

        print()

        # Problem metrics
        problem_metrics = evaluate_predictions(problem_preds, problem_refs)
        print(f"Metrics for problem {problem['name']}:")
        print(f"Total Questions: {len(problem_preds)}")
        print(f"BLEU: {problem_metrics['BLEU']}")
        print(f"chrF: {problem_metrics['chrF']}")
        print()
        print("-" * 50)
        print()

        results[problem['name']] = {
            "type": problem['type'],
            "difficulty": problem['difficulty'],
            "metrics": problem_metrics
        }
        

    if predictions:
        #print(len(predictions),predictions)
        #print(len(references), references)
        final_metrics = evaluate_predictions(predictions, references)
        print(f"FINAL METRICS - ({args.task})")
        print(f"Total Questions: {len(predictions)}")
        print(f"BLEU: {final_metrics['BLEU']}")
        print(f"chrF: {final_metrics['chrF']}")

        print()

        # Sort results by BLEU score ascending (lowest first)
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['metrics']['BLEU']))

        # Save results to a JSON file
        with open(f"../results/{args.task}_{args.language}_{args.difficulty}_{args.type}.json", "w", encoding='utf-8') as f:
            json.dump(sorted_results, f, indent=4)

        # print a comprehensive table of results
        print("Comprehensive Results Table:")
        print(f"{'Problem Name':<30} {'Type':<10} {'Difficulty':<10} {'BLEU':<10} {'chrF':<10}")
        print("-" * 80)
        for problem_name, info in sorted_results.items():
            type_str = info['type'] if isinstance(info['type'], str) else ', '.join(info['type'])
            print(f"{problem_name:<30} {type_str:<10} {str(info['difficulty']):<10} {info['metrics']['BLEU']:<10.4f} {info['metrics']['chrF']:<10.4f}")