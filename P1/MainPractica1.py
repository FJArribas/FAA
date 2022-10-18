import numpy as np
from Datos import Datos
from EstrategiaPatrocinado import ValidacionSimple, ValidacionCruzada
from Clasificador import Clasificador
from ClasificadorNB import ClasificadorNaiveBayes

if __name__ == "__main__":
    # Primer set de datos
    dataset = Datos("ConjuntosDatosIntroFAA/tic-tac-toe.csv")

    # DEBUG

    # print(dataset.nominalAtributos)
    # print(dataset.diccionarios)
    # print(dataset.datos)

    # Clasificador
    NB = ClasificadorNaiveBayes() # Laplace - False
    NBL = ClasificadorNaiveBayes(laplace = True) # Laplace - True

    # EstrategiaPatrocinado
    estrategia = ValidacionSimple()
    estrategia.creaParticiones(dataset)

    estrategia2 = ValidacionCruzada()
    estrategia2.creaParticiones(dataset)

    # DEBUG

    # print("\nValidacionSimple")
    # print(estrategia)

    # print("\nValidacionCruzada")
    # print(estrategia2)

    # Naive Bayes tic tac toe
    print("Tic-tac-toe, sin Laplace")
    listaErrores, mConf = NB.validacion(estrategia, dataset, NB, seed=None)

    print("\nErrores en Naive Bayes, validacion simple: ")
    print(listaErrores[0])

    listaErrores, mConf = NB.validacion(estrategia2, dataset, NB, seed=None)
    print("\nErrores en Naive Bayes, validacion cruzada: ")
    print(listaErrores)
    print("\nMedia errores: ")
    print(np.mean (listaErrores))

    print("\n\nTic-tac-toe, con Laplace")
    listaErrores, mConf = NBL.validacion(estrategia, dataset, NBL, seed=None)

    print("\nErrores en Naive Bayes, validacion simple: ")
    print(listaErrores[0])

    listaErrores, mConf = NBL.validacion(estrategia2, dataset, NBL, seed=None)
    print("\nErrores en Naive Bayes, validacion cruzada: ")
    print(listaErrores)
    print("\nMedia errores: ")
    print(np.mean (listaErrores))



    # Segundo set de datos
    dataset2 = Datos("ConjuntosDatosIntroFAA/german.csv")

    # DEBUG

    # print(dataset2.nominalAtributos)
    # print(dataset2.diccionarios)
    # print(dataset2.datos)

    # Clasificador
    NB2 = ClasificadorNaiveBayes() # Laplace - False
    NB2L = ClasificadorNaiveBayes(laplace = True) # Laplace - True

    # EstrategiaPatrocinado
    estrategia3 = ValidacionSimple()
    estrategia3.creaParticiones(dataset2)

    estrategia4 = ValidacionCruzada()
    estrategia4.creaParticiones(dataset2)

    # DEBUG

    # print("\nValidacionSimple")
    # print(estrategia3)

    # print("\nValidacionCruzada")
    # print(estrategia4)

    # Naive Bayes german

    print("\n\n\nGerman, sin Laplace")
    listaErrores2, mConf = NB2.validacion(estrategia3, dataset2, NB2, seed=None)

    print("\nErrores en Naive Bayes, validacion simple: ")
    print(listaErrores2[0])

    listaErrores2, mConf = NB2.validacion(estrategia4, dataset2, NB2, seed=None)
    print("\nErrores en Naive Bayes, validacion cruzada: ")
    print(listaErrores2)
    print("\nMedia errores: ")
    print(np.mean (listaErrores2))

    print("\n\nGerman, con Laplace")
    listaErrores2, mConf = NB2L.validacion(estrategia3, dataset2, NB2L, seed=None)

    print("\nErrores en Naive Bayes, validacion simple: ")
    print(listaErrores2[0])

    listaErrores2, mConf = NB2L.validacion(estrategia4, dataset2, NB2L, seed=None)
    print("\nErrores en Naive Bayes, validacion cruzada: ")
    print(listaErrores2)
    print("\nMedia errores: ")
    print(np.mean (listaErrores2))