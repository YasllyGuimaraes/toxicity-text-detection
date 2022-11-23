import pickle
import pandas as pd
from nltk.corpus import stopwords
stopwords = stopwords.words("english")
from configs.limpando_dados import Limpeza

CLASSIFICADOR = 'notebooks\pre-processado-sw-lemma_Bayes_bow.pickle'
test_df = pd.read_csv("./data/test_binary_small.csv")


def preprocess(sample):

    sample['pre-processado-sw-lemma'] = sample['comment_text'].apply(
        lambda x: Limpeza.minusculo(x))

    sample['pre-processado-sw-lemma'] = sample['pre-processado-sw-lemma'].apply(
        lambda x: Limpeza.substituindo(x))

    sample['pre-processado-sw-lemma'] = sample['pre-processado-sw-lemma'].apply(
        lambda x: Limpeza.retirando_caracteres_especiais(x))

    sample['pre-processado-sw-lemma'] = sample['pre-processado-sw-lemma'].apply(
        lambda x: Limpeza.lemmatizacao(x))

    sample['pre-processado-sw-lemma'] = sample['pre-processado-sw-lemma'].apply(
        lambda x: Limpeza.remover_stopwords(x, stopwords))

    test_df.drop(["id", "comment_text", "Toxic"], axis=1, inplace=True)

    return sample


def perform_prediction(sample):
  
    new_sample = preprocess(sample)
    print(new_sample)

    with open(CLASSIFICADOR, 'rb') as f:
        clf = pickle.load(f)
        print(clf)

    new_sample['Toxic_predict'] = clf.predict(new_sample['pre-processado-sw-lemma'])

    test_df['Toxic_predict_str'] = test_df['Toxic_predict'].replace([1, 0], ['Toxic','Non-Toxic'])
    print(type(new_sample))

    new_sample.to_csv('result.csv')

    return new_sample

print(perform_prediction(test_df))

