from llm_client import generate_translation

# Baseline evaluation
def zero_shot(problem, question):
    examples = []
    for s,t in zip(problem['source_examples'], problem['target_examples']):
        examples.append(f"{s} = {t}")
    
    examples_str = "\n".join(examples)
    
    prompt = f"""
You are an expert translator specialized in solving linguistic puzzles. Translate the target sentence based strictly on the vocabulary and grammar rules implicitly shown in the examples. 
Provide only the translation line. Do not provide any explanations or extra text.

Examples:
{examples_str}

Sentence to translate from {question}
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
def cot(problem, question):
    examples = []
    for s,t in zip(problem['source_examples'], problem['target_examples']):
        examples.append(f"{s} = {t}")
    
    examples_str = "\n".join(examples)
    
    prompt = f"""
You are an expert translator specialized in solving linguistic puzzles.
Here is an explanation of some of the concepts or strategies a participant should consider when analyzing the data:
    1. Agents and patients: identify who is performing the action (the agent) and who is being affected by it (the patient). These roles are the foundation of sentence meaning.
    2. Grammatical functions and cases: languages often mark the role of a noun (subject, direct object, indirect object) using specific endings or changes in the word, known as grammatical cases.
    3. Affixation: many languages build words by adding pieces to a root. Look for prefixes (at the beginning), suffixes (at the end), infixes (inserted inside the root), or circumfixes (wrapped around the root).
    4. Word order: observe the sequence of elements. While English is typically SVO (subject-verb-object), other languages might prefer SOV, VSO, or even free word order.
    5. Morphological strategy: determine if the language is agglutinative, where each piece of information has a clear boundary (like beads on a string), or fusional, where several meanings are packed into a single, inseparable marker.
    6. Noun classes and gender: many languages categorize nouns into groups (human, animal, plant, inanimate, etc.). These categories often dictate how other words in the sentence behave.
    7. Number systems: look beyond the simple singular and plural. Some languages have dual forms for exactly two things, or paucal forms for a small group.
    8. Tense, aspect, and mood: these markers indicate when an action happens (tense), whether it is ongoing or completed (aspect), and the speaker's degree of certainty or intent (mood).
    9. Pro-drop: in some languages, pronouns are omitted if the subject is already clear from the verb ending or the context.
    10. Agreement: notice if adjectives, verbs, or articles change their form to match the gender, number, or class of the noun they accompany.

The right strategy to solve a problem is to sequentially analyze the examples, infer the vocabulary and grammar rules and then apply them to solve the question. Consider that some concepts and correct single-word translations can be found and/or confirmed by other examples, so the correct approach should be to look in other sentences if a word cannot be understood at first sight.

Translate the target sentence based strictly on the vocabulary and grammar rules implicitly shown in the examples. 
Provide only the translation line. Do not provide any explanations or extra text.

Think step-by-step.

Examples:
{examples_str}

Sentence to translate from {question}
Translation:
    """
    
    response = generate_translation(prompt)
    return _clean_response(response)

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