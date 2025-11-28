import gensim
import pandas as pd
from gensim.models import Word2Vec

df=pd.read_json('D:/Data Science/deep_learning/word2vec/Cell_Phones_and_Accessories_5.json', lines=True)
print(df.head())
review_text = df.reviewText.apply(gensim.utils.simple_preprocess)
model = Word2Vec(sentences=review_text, vector_size=100, window=5, min_count=2, workers=4, sg=1)
model.build_vocab(review_text, progress_per=1000)
model.train(review_text, total_examples=model.corpus_count, epochs=model.epochs)
model.save("word2vec.model")
loaded_model = Word2Vec.load("word2vec.model")
print(loaded_model.wv.most_similar("phone"))
print(loaded_model.wv.most_similar("battery"))
print(loaded_model.wv.most_similar("camera"))