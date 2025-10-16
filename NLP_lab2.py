import gensim.models
import re

word2vec = gensim.models.KeyedVectors.load_word2vec_format("cbow.txt", binary=False)
pat = re.compile("(.*)_NOUN")

pos = ["процесс_NOUN", "регуляция_NOUN"]
similar = word2vec.most_similar(positive=pos, topn=20)

similar_nouns_tagged = [w for w, s in similar if pat.match(w) is not None]
similar_nouns_bases = [pat.match(w).group(1) for w in similar_nouns_tagged]

print("Существительные, близкие в векторном представлении к 'процесс_NOUN' и 'регуляция_NOUN':")
for base in similar_nouns_bases:
    print(base)


results = word2vec.most_similar(positive=["механизм_NOUN", "активность_NOUN"],  topn=10)
print("механизм + активность:")
for word, score in results:
    base_word = word.split('_')[0]
    print(f"  {base_word} (схожесть: {score:.4f})")


results = word2vec.most_similar(positive=["механизм_NOUN", "функция_NOUN"], negative=["система_NOUN"], topn=10)
print("механизм + функция - система:")
for word, score in results:
    base_word = word.split('_')[0]
    print(f"  {base_word} (схожесть: {score:.4f})")
