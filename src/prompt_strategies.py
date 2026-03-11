from llm_client import generate_translation

# Baseline evaluation
def zero_shot(problem, question):
    examples = []
    for s,t in zip(problem['source_examples'], problem['target_examples']):
        examples.append(f"{s} = {t}")
    
    examples_str = "\n".join(examples)
    
    prompt = f"""You are an expert translator specialized in solving linguistic puzzles. Translate the target sentence based strictly on the vocabulary and grammar rules implicitly shown in the examples. 
### EXAMPLES
{examples_str}

### YOUR OBJECTIVE
Translate the following sentence: {question}

### RESPONSE FORMAT
Provide only the translation line. Do not provide any explanations or extra text.
Translation:"""
    
    response = generate_translation(prompt)
    if ":" in response:
        response = response.split(":")[-1].strip()
    
    return response.strip()


# Chain of Thought strategy
def cot_linguistic(problem, question):
    examples = []
    for s,t in zip(problem['source_examples'], problem['target_examples']):
        examples.append(f"{s} = {t}")
    
    examples_str = "\n".join(examples)
    
    prompt = f"""You are a Linguistic Olympiad expert. Your task is to solve a translation puzzle by extracting rules from limited examples.
### EXAMPLES
{examples_str}

### YOUR OBJECTIVE
Translate the following sentence: {question}

### INSTRUCTIONS
Perform a step by step metalinguistic analysis. Before providing the translation, explicitly consider these 10 concepts if they are relevant to the patterns you see:
1. Agents and patients (Who does what? Who is being affected?)
2. Grammatical functions and cases (Endings for subjects/objects?)
3. Affixation (Prefixes, suffixes, infixes or circumfixes?)
4. Word order (SVO, SOV, VSO, free order?)
5. Morphological strategy (Agglutinative or fusional?)
6. Noun classes and gender (Categories like human/animal/inanimate?)
7. Number systems (Singular, plural, dual, paucal?)
8. Tense, aspect, and mood (When and how the action happens?)
9. Pro-drop (Are pronouns omitted?)
10. Agreement (Do adjectives/verbs match the gender, number or class of the nouns?)

### RESPONSE FORMAT
1. Analysis: [Your step by step reasoning]
2. Final Translation: [ONLY the translated text, no prefixes]
"""
    
    response = generate_translation(prompt)
    
    # Get only the text after "Final Translation:"
    if "Final Translation:" in response:
        prediction = response.split("Final Translation:")[-1].strip()
    else:
        # Fallback, take the last line
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        prediction = lines[-1] if lines else ""

    # Final cleanup to remove any language labels
    if ":" in prediction:
        prediction = prediction.split(":")[-1].strip()

    return prediction