from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


# Dicionário de hiperparâmetros:

C = [2 ** i for i in range(-5, 16, 2)]
gamma = [2 ** i for i in range(-15, 4, 2)]

MODELOS = {'Bayes': {'clf': MultinomialNB(),
                    'parameters': {
                        'alpha': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9 ]
                        }
    },
    'KNN': {'clf': KNeighborsClassifier(),
        'parameters': {
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],}
    },

    'DecisionTree': {'clf': DecisionTreeClassifier(),
        'parameters': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],}
    },

    'RandomForest': {'clf': RandomForestClassifier(),
        'parameters':{
        'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],}
     },

    'SVM': {'clf': SVC(),
        'parameters':{
        'C': C,
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'degree':[1, 2, 3, 4, 5]},
    }
}

# Dicionário de hiperparâmetros dos vetorizadores:

vetorizadores_params = {
    'vetorizador__max_df': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'vetorizador__ngram_range':[(1,1),(1,2),(2,2),(1,3)]
}

# Dicionário de modelos:

modelos = {
    'SVMLinear':SVC(kernel="linear"),
    'Bayes': MultinomialNB(),
    'KNN':KNeighborsClassifier(n_neighbors=3),
    'DecisionTree':DecisionTreeClassifier(random_state=42),
    'MLP1CamadaOculta':MLPClassifier(hidden_layer_sizes=(12,)),
    'MLP2CamadasOcultas':MLPClassifier(hidden_layer_sizes=(20,100)),
    }

# Dicionário de vetorizadores:

vetorizadores = {
    'tdidf':TfidfVectorizer(),
    'bow':CountVectorizer()
}
