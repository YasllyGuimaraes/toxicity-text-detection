import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('words')
import nltk
nltk.download('omw-1.4')
from nltk.stem import *
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer



class Limpeza():

    
    def retirando_caracteres_especiais(texto):
        texto_novo = re.sub(r"\W+"," ", texto)
        return texto_novo
    
    def retirando_numeros(texto):
        texto_novo = re.sub(r"\d+", " ", texto)
        return texto_novo

    def substituindo(texto):
        texto_novo = re.sub(r"< br />", "", texto)
        texto_novo = re.sub(r"'ll", "will", texto)
        texto_novo = re.sub(r"'s", "is", texto)
        texto_novo = re.sub(r"'re", "are", texto)
        texto_novo = re.sub(r"n't", "not", texto) 
        return texto_novo

    def minusculo(texto):
        texto_novo = texto.lower()
        return texto_novo
    
    def remover_stopwords(texto, stopwords):
        palavras = [word for word in word_tokenize(texto) if not word in stopwords]
        texto_novo = " ".join(palavras)
        return texto_novo

    def numero_de_palavras(texto):
        palavras = nltk.word_tokenize(texto)
        quantidade_de_palavras = len(palavras)
        return quantidade_de_palavras

    def lemmatizacao(texto):
        lemmatizer = nltk.WordNetLemmatizer()
        palavras = nltk.word_tokenize(texto)
        new_palavras = [lemmatizer.lemmatize(palavra) for palavra in palavras]
        texto_novo = (' '.join(new_palavras))
        return texto_novo

    def stemming(texto):
        stemmer = PorterStemmer()
        palavras = nltk.word_tokenize(texto)
        new_palavras = [stemmer.stem(palavra) for palavra in palavras]
        texto_novo = (' '.join(new_palavras))
        return texto_novo



    

    