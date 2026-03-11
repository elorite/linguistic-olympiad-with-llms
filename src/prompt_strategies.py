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


def _clean_response(response):
    """Extract the final answer from a (possibly multi-line, markdown-formatted) response."""
    if ":" in response:
        response = response.split(":")[-1].strip()
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    response = lines[-1] if lines else response
    return response.replace("**", "").strip()


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

def custom(problem, question):
    examples = []
    for s,t in zip(problem['source_examples'], problem['target_examples']):
        examples.append(f"{s} = {t}")
    
    examples_str = "\n".join(examples)
    
    prompt = f"""
You are an expert translator specialized in solving linguistic puzzles. 

You should follow a reasoning process that is based on sentence comparison, that is the task of comparing 
different translations to understand common words which lead to a confirmation of that translation in 
the vocabulary. Follow these steps:
1.  Analyze the sentences and relative translations in a sequential manner and try to identify simpler
    translations first, so we can start building the vocabulary. For example, if you have the following 
    sentences in an hypothetical dataset (Nung language) and the relative translations in English:
    
    Nung: Cáu ca vửhn nhahng kíhn.
    English: I was about to continue to eat it.
    
    Nung: Cáu cháhn slờng páy mi?
    English: Do I truly want to go?

    You can easily understand that 'Cáu' refers to 'I', since in both it appears at the beginning and in the 
    English translation you have 'I' at the beginning.

    So the key idea here is to find patterns in the sentences which help you understand the simpler words in 
    the vocabulary.

2.  Certain more complex words may even be understood the above way, but usually verbs, adjectives and nouns are 
    hard to translate. So now you have to take the remaining non-translated part of the sentence in the low-resource
    language and try to understand it's meaning given the remaining part in the relative translation. This is 
    a hard task and can be completed with sentence comparison. From it you can define some grammatical rules and 
    some basic word order. However, the objective up to this point is mainly to determine the vocabulary, so focus on that.

3.  Once created the vocabulary and the relative word-to-word translation, you can translate word-to-word a sentence and then 
    write the sentence translated in the other language. For example, one word in Nung may refer to 'negative imperative' and so 
    you have to give it a meaning in English, which could be 'Don't' if it is at the beginning of the sentence. 

4.  The final step is to figure out the word order. From the dataset you should try to understand the word order of each language
    so that you can then apply it to the relative translation. As an example, in English the word order is Subject-Verb-Objective, but in 
    another language it may be different. Finding it out will help you determine the final order of the translation considering that 
    you have translated word-by-word. 

Now, translate the target sentence following this step-by-step reasoning and use the examples below to understand 
all the semantic meaning and grammatical rules you need to make the correct translation. The final output has to be 
just the final translation of the relative target sentence.

Examples:
{examples_str}

Sentence to translate from {question}
Translation:"""
    
    response = generate_translation(prompt)
    return _clean_response(response)