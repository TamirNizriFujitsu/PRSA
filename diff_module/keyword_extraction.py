import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist


nltk.download('punkt')
nltk.download('stopwords')

def extract_keywords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    freq_dist = FreqDist(filtered_words)
    return [word for word, freq in freq_dist.most_common(10)]


