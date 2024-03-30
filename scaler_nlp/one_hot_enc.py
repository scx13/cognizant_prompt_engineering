from sklearn.preprocessing import OneHotEncoder
import itertools

document = ["The", "boy", "sat", "on", "the", "floor"]
tokens = [doc.split(" ") for doc in document]
token_chain = itertools.chain.from_iterable(tokens)
word_to_id = {token: idx for idx, token in enumerate(set(token_chain))}
token_ids = [[word_to_id[token] for token in toke] for toke in tokens]

vec = OneHotEncoder(categories="auto")
V = vec.fit_transform(token_ids)
print(V.toarray())

# highly sparse matrix with each sentence the majority of the values are zero
# document has different sizes we get vectors with different sizes
# does not capture semantic meaning of the words
