# converts a whole piece of text into fixed-length vectors
# counts the frequency of the words
from sklearn.feature_extraction.text import CountVectorizer

document = ["NLP has changed the world. I love NLP. NLP is cool"]
vectorizer = CountVectorizer()
vectorizer.fit(document)

print(vectorizer.vocabulary_)

vector = vectorizer.transform(document)
print(vector.toarray())

# we dont extract any semnatic meaning from the words
