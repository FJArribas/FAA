from abc import ABCMeta, abstractmethod
import numpy as np
from Datos import Datos

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

# https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/naive_bayes.py#L750
"""
    Naive Bayes classifier for multinomial models.
    The multinomial Naive Bayes classifier is suitable for classification with
    discrete features (e.g., word counts for text classification). The
    multinomial distribution normally requires integer feature counts. However,
    in practice, fractional counts such as tf-idf may also work.
    Read more in the :ref:`User Guide <multinomial_naive_bayes>`.
"""
dataset = Datos("ConjuntosDatosIntroFAA/german.csv")
features = dataset.nominalAtributos[:-1]

ct = ColumnTransformer(dataset.nominalAtributos[:-1])

validacionSimple = ShuffleSplit(len(dataset.datos), test_size=.2, random_state=0)
validacionCruzada = 5

f = open("outputScikit.txt", "w")


# Validacion Simple, sin Laplace
atributos = preprocessing.OneHotEncoder(sparse = False)
X = atributos.fit_transform(dataset.datos[:, :-1])
Y = dataset.datos[:, -1]

clasificador = MultinomialNB(alpha = 0.1, class_prior = None, fit_prior = True) # Alpha 0 -> No Laplace
clasificador.fit(X, Y)

valSimple = cross_val_score(clasificador, X, Y, cv = validacionSimple)

f.write("Validacion Simple sin usar Laplace:\n")
f.write("\tMedia: ")
f.write(str(1 - np.mean(valSimple)))
f.write("\n\tVarianza: ")
f.write(str(np.var(valSimple)))
f.write("\n\n")

# Validacion Simple, con Laplace
atributos = preprocessing.OneHotEncoder(sparse = False)

X = atributos.fit_transform(dataset.datos[:, :-1])
Y = dataset.datos[:, -1]

clasificador = MultinomialNB(alpha = 1, class_prior = None, fit_prior = True) # Alpha 1 -> Laplace
clasificador.fit(X, Y)

valSimple = cross_val_score(clasificador, X, Y, cv = validacionSimple)

f.write("Validacion Simple usando Laplace:\n")
f.write("\tMedia: ")
f.write(str(1 - np.mean(valSimple)))
f.write("\n\tVarianza: ")
f.write(str(np.var(valSimple)))
f.write("\n\n")

# Validacion Cruzada, sin Laplace
atributos = preprocessing.OneHotEncoder(sparse = False)
X = atributos.fit_transform(dataset.datos[:, :-1])
Y = dataset.datos[:, -1]

clasificador = MultinomialNB(alpha = 0.1, class_prior = None, fit_prior = True) # Alpha 0 -> No Laplace
clasificador.fit(X, Y)

valCruzada = cross_val_score(clasificador, X, Y, cv = validacionCruzada)

f.write("Validacion Cruzada sin usar Laplace:\n")
f.write("\tMedia: ")
f.write(str(1 - np.mean(valCruzada)))
f.write("\n\tVarianza: ")
f.write(str(np.var(valCruzada)))
f.write("\n\n")

# Validacion Cruzada, con Laplace
atributos = preprocessing.OneHotEncoder(sparse = False)
X = atributos.fit_transform(dataset.datos[:, :-1])
Y = dataset.datos[:, -1]

clasificador = MultinomialNB(alpha = 1, class_prior = None, fit_prior = True) # Alpha 1 -> Laplace
clasificador.fit(X, Y)

valCruzada = cross_val_score(clasificador, X, Y, cv = validacionCruzada)

f.write("Validacion Cruzada usando Laplace:\n")
f.write("\tMedia: ")
f.write(str(1 - np.mean(valCruzada)))
f.write("\n\tVarianza: ")
f.write(str(np.var(valCruzada)))
f.write("\n")

f.close()
