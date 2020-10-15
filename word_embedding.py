from nltk.corpus import stopwords
import string
import spacy
import numpy as np

# USE THIS COMMAND FIRST!!
# python -m spacy download en_core_web_sm

table = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words('english'))
model = spacy.load('en_core_web_sm')


def word_embedding(category):
    """
    :param category: string, word
    :return: word embedding
    """
    category = category.strip()
    tokens = category.split(" ")
    # remove punctuation and convert to lowercase
    tokens = [w.translate(table).lower() for w in tokens]
    # filter out short tokens, stop words
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    if len(tokens) == 0:
        tokens = ["NA"]
    embedding = model(tokens[0]).vector
    for token in tokens[1:]:
        embedding = np.sum([embedding, model(token).vector], axis=0)
    embedding /= len(tokens)
    return embedding


if __name__ == '__main__':
    print(word_embedding("hello word").shape)
