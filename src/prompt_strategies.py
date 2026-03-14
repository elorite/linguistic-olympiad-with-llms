import time
from llm_client import generate_translation

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

def _clean_response(response):
    """Extract the final answer from a (possibly multi-line, markdown-formatted) response."""
    if ":" in response:
        response = response.split(":")[-1].strip()
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    response = lines[-1] if lines else response
    return response.replace("**", "").strip()

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

def _build_examples_str(problem):
    """Helper to build the examples string from a problem."""
    examples = []
    for s, t in zip(problem['source_examples'], problem['target_examples']):
        examples.append(f"{s} = {t}")
    return "\n".join(examples)


def _extract_final_translation(response):
    """Extract the final translation from a response containing 'Final Translation:'."""
    if "Final Translation:" in response:
        prediction = response.split("Final Translation:")[-1].strip()
    else:
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        prediction = lines[-1] if lines else ""
    if ":" in prediction:
        prediction = prediction.split(":")[-1].strip()
    return _clean_response(prediction)

def back_translation(problem, question):
    examples_str = _build_examples_str(problem)

    # Stage 1: Forward translation with cot_linguistic-style prompt
    forward_prompt = f"""You are a Linguistic Olympiad expert. Your task is to solve a translation puzzle by extracting rules from limited examples.
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

    forward_response = generate_translation(forward_prompt)
    forward_translation = _extract_final_translation(forward_response)

    time.sleep(5)

    # Stage 2: Back-translate and verify
    back_prompt = f"""You are a Linguistic Olympiad expert verifying a translation by translating it back.

### EXAMPLES
{examples_str}

### ORIGINAL SENTENCE
{question}

### PROPOSED TRANSLATION
{forward_translation}

### INSTRUCTIONS
1. Using the same examples and rules, translate the proposed translation BACK into the original language
2. Compare your back-translation with the original sentence
3. If the back-translation matches the original sentence, the proposed translation is correct — return it as is
4. If the back-translation does NOT match, identify the specific words/morphemes causing the mismatch and fix the forward translation accordingly

### RESPONSE FORMAT
1. Back-translation: [Your back-translation of the proposed translation]
2. Comparison: [Does the back-translation match the original? What mismatches exist?]
3. Final Translation: [The corrected translation of the original sentence, or the original proposed translation if correct]
"""

    back_response = generate_translation(back_prompt)
    return _extract_final_translation(back_response)

