import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import RegexpParser


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def extract_key_phrases(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    tagged = pos_tag(filtered_words)
    pattern = "NP: {<DT>?<JJ>*<NN>}"
    cp = RegexpParser(pattern)
    cs = cp.parse(tagged)

    key_phrases = []
    for subtree in cs.subtrees():
        if subtree.label() == 'NP':
            key_phrases.append(' '.join(word for word, tag in subtree.leaves()))

    return key_phrases

