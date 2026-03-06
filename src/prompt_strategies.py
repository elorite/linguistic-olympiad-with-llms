from llm_client import generate_translation

# Baseline evaluation
def zero_shot(problem, question):
    examples = []
    for s,t in zip(problem['source_examples'], problem['target_examples']):
        examples.append(f"{s} = {t}")
    
    examples_str = "\n".join(examples)

    prefix = "English:" if question.startswith(problem['name']) else f"{problem['name']}:"

    prompt = f"""You are an expert translator specialized in solving linguistic puzzles. Translate the target sentence based strictly on the vocabulary and grammar rules implicitly shown in the examples. 
Provide only the translation line starting with '{prefix}'. Do not provide any explanations or extra text.

Examples:
{examples_str}

Sentence to translate from {question}
Translation:"""
    
    return generate_translation(prompt)


# Chain of Thought strategy
def cot_linguistic(problem):
    examples = []
    for s,t in zip(problem['source_examples'], problem['target_examples']):
        examples.append(f"{s} = {t}")
    
    examples_str = "\n".join(examples)
    
    prompt = ""
    
    response = generate_translation(prompt)
    
    return 