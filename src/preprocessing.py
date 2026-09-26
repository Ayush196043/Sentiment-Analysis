import re
import string

# Contraction mapping dictionary for preserving negations and expanding shorthand
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "doesn't": "does not",
    "don't": "do not",
    "didn't": "did not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "mightn't": "might not",
    "mustn't": "must not",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "what's": "what is",
    "where's": "where is",
    "how's": "how is",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would"
}

NEGATION_WORDS = {
    'no', 'not', 'nor', 'neither', 'never', 'none', 'nobody', 'nothing',
    'nowhere', 'cannot', 'cant', 'wont', 'isnt', 'arent', 'wasnt', 'werent',
    'havent', 'hasnt', 'hadnt', 'doesnt', 'dont', 'didnt', 'couldnt',
    'shouldnt', 'wouldnt'
}

def expand_contractions(text: str) -> str:
    """Expand common English contractions to preserve negation meaning."""
    text_lower = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text_lower = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text_lower)
    return text_lower

def clean_text(text: str, remove_stopwords: bool = False, custom_stopwords: set = None) -> str:
    """
    Reusable NLP text cleaning function.
    
    Steps:
    1. Handle NaN / non-string inputs safely.
    2. Expand contractions (preserves negation words like 'n\'t' -> 'not').
    3. Remove HTML tags (e.g. <br />, <html>).
    4. Remove URLs (http, https, www).
    5. Convert to lower case.
    6. Remove special symbols & punctuation, keeping alphanumeric and spaces.
    7. Normalize extra whitespaces.
    8. Optional negation-aware stopword removal.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Expand contractions
    text = expand_contractions(text)
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Remove HTML tags (replace with space so words don't merge)
    text = re.sub(r'<.*?>', ' ', text)
    
    # 4. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # 5. Remove unwanted symbols & punctuation (keep letters, digits, and spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 6. Normalize multiple whitespaces to a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. Optional Stopword Removal (preserving negations)
    if remove_stopwords and custom_stopwords:
        tokens = text.split()
        tokens = [word for word in tokens if word not in custom_stopwords or word in NEGATION_WORDS]
        text = " ".join(tokens)
        
    return text

if __name__ == "__main__":
    # Test quick demonstration
    sample = "This product isn't good at all! <br />Check http://example.com for MORE details... 100% TERRIBLE."
    print("Sample Before:", sample)
    print("Sample After: ", clean_text(sample))
