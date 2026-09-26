import re
import pandas as pd
from pathlib import Path

# Aspect Keyword Dictionaries tailored to Amazon Fine Food Corpus
ASPECT_KEYWORDS = {
    'Taste': [
        'taste', 'tastes', 'flavor', 'flavour', 'flavors', 'yummy', 'delicious',
        'tasty', 'sweet', 'sour', 'bitter', 'salty', 'bland', 'spicy', 'yuck', 'disgusting'
    ],
    'Packaging': [
        'package', 'packaging', 'box', 'boxes', 'bag', 'bags', 'container',
        'wrapper', 'bottle', 'can', 'canned', 'seal', 'sealed', 'wrapped', 'damaged', 'broken', 'leaked'
    ],
    'Delivery': [
        'delivery', 'delivered', 'shipping', 'shipment', 'arrived', 'arrives',
        'fast', 'slow', 'late', 'seller', 'vendor', 'carrier'
    ],
    'Price': [
        'price', 'priced', 'cost', 'expensive', 'cheap', 'worth', 'value',
        'money', 'deal', 'bargain', 'overpriced', 'cents', 'dollars'
    ],
    'Quality': [
        'quality', 'fresh', 'freshness', 'stale', 'rotten', 'expired',
        'expiration', 'spoiled', 'rancid', 'moldy', 'good quality', 'high quality'
    ],
    'Ingredients': [
        'ingredient', 'ingredients', 'sugar', 'organic', 'natural', 'calorie',
        'calories', 'healthy', 'chemical', 'preservative', 'additive', 'fat', 'sodium'
    ],
    'Customer Support': [
        'service', 'support', 'customer service', 'refund', 'replaced',
        'return', 'returned', 'guarantee', 'replacement'
    ]
}

# Lexicon for local clause sentiment analysis
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'delicious', 'yummy', 'awesome',
    'love', 'loved', 'best', 'wonderful', 'perfect', 'fresh', 'fast', 'cheap',
    'bargain', 'worth', 'enjoyed', 'tasty', 'like', 'liked', 'recommend', 'happy'
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'horrible', 'awful', 'poor', 'disappointed', 'disappointing',
    'stale', 'rotten', 'expired', 'damaged', 'broken', 'slow', 'late', 'expensive',
    'overpriced', 'disgusting', 'bland', 'rancid', 'waste', 'wasted', 'hate', 'hated',
    'nasty', 'tasteless', 'leaked', 'leaking'
}

NEGATION_TOKENS = {'not', 'no', 'never', 'nor', 'neither', 'cannot', 'cant', 'wont', 'dont', 'didnt', 'isnt', 'wasnt'}

def split_sentences(text: str) -> list:
    """Split text into sentence clauses."""
    return [s.strip() for s in re.split(r'[.!?;\n]+', str(text)) if len(s.strip()) > 0]

def analyze_clause_sentiment(clause: str) -> str:
    """
    Rule-based local sentiment analysis for a single sentence clause.
    Considers positive/negative lexicons and negation flips.
    """
    words = re.findall(r'\b[a-z]+\b', clause.lower())
    
    pos_score = 0
    neg_score = 0
    
    for i, w in enumerate(words):
        is_negated = False
        # Check window of 3 words before current word for negation
        window = words[max(0, i-3):i]
        if any(neg_tok in window for neg_tok in NEGATION_TOKENS):
            is_negated = True
            
        if w in POSITIVE_WORDS:
            if is_negated:
                neg_score += 1
            else:
                pos_score += 1
        elif w in NEGATIVE_WORDS:
            if is_negated:
                pos_score += 1
            else:
                neg_score += 1
                
    if pos_score > neg_score:
        return 'Positive'
    elif neg_score > pos_score:
        return 'Negative'
    else:
        return 'Neutral'

def analyze_aspect_sentiment(text: str) -> list:
    """
    Aspect-Based Sentiment Analysis pipeline.
    Identifies aspects mentioned in review and assigns local clause sentiment.
    Returns list of dicts: [{'aspect': 'AspectName', 'sentiment': 'Positive/Negative/Neutral', 'clause': '...'}]
    """
    clauses = split_sentences(text)
    detected_aspects = []
    
    for aspect_name, keywords in ASPECT_KEYWORDS.items():
        # Find clauses mentioning keywords for this aspect
        matching_clauses = []
        for clause in clauses:
            clause_lower = clause.lower()
            if any(re.search(r'\b' + re.escape(kw) + r'\b', clause_lower) for kw in keywords):
                matching_clauses.append(clause)
                
        if matching_clauses:
            # Combine matching clauses and analyze local sentiment
            combined_context = " ".join(matching_clauses)
            clause_sent = analyze_clause_sentiment(combined_context)
            
            detected_aspects.append({
                'aspect': aspect_name,
                'sentiment': clause_sent,
                'matching_clause': matching_clauses[0][:100]
            })
            
    return detected_aspects

if __name__ == "__main__":
    sample = "The taste is excellent and delicious, but the delivery was late and packaging was damaged."
    res = analyze_aspect_sentiment(sample)
    print("Sample:", sample)
    print("ABSA Extraction:", res)
