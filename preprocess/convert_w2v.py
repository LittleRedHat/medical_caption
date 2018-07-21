import gensim
import pandas as pd 
import numpy as np


model = gensim.models.KeyedVectors.load_word2vec_format('~/download/PubMed-w2v.bin', binary=True)
# model.save_word2vec_format('~/download/PubMed-w2v.txt', binary=False)


df = pd.read_csv('../output/preprocess/IU_Chest_XRay/words.csv')
words = df['word'].values

w2v = {}

for word in words:
    vector = model.get_vector(word)
    w2v[word] = vector


w2v_df = pd.DataFrame(w2v)

w2v_df.to_csv('../output/preprocess/IU_Chest_XRay/w2v.csv',index=False)


    
