import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
import gensim
import re

import spacy
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stopwords.words('english')]
    
    return filtered_words



def topic_identification(documents):
    nlp = spacy.load("en_core_web_sm")
    keywords = Counter()

    for document in documents:
        doc = nlp(document)
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                keywords[token.lemma_] += 1
        for ent in doc.ents:
            keywords[ent.text] += 1

    return [keyword for keyword, _ in keywords.most_common(5)]


if __name__ == '__main__':
    documents = [f"""
    Product/Service: fitness equipment.
    Customer Persona: fitness enthusiasts. 
    """]
    print(topic_identification(documents, num_words=1))
