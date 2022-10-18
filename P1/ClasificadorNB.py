from abc import ABCMeta, abstractmethod
import numpy as np
import math
import statistics
from Datos import Datos
from Clasificador import Clasificador

class ClasificadorNaiveBayes(Clasificador):
    def __init__(self, laplace = False):
        self.laplace = laplace
        self.atributos = []
        self.aPriori = {}

    def entrenamiento(self, datosTrain, nominalAtributos, diccionario):
        self.valoresClase, counts = np.unique (datosTrain[:, -1], return_counts = True)
        nClases = len(self.valoresClase)

        # Calculamos las probabilidades a priori de la hipotesis
        for i in range(nClases):
            self.aPriori[self.valoresClase[i]] = counts[i] / len(datosTrain)

        # Tablas de cada atributo
        j = 0
        for key, i in diccionario.items():
            if(j == len(diccionario) - 1):
                break

            if nominalAtributos[j]:                     # Es un atributo nominal
                # Creamos una tabla vacia
                tabla = np.zeros((len(i), nClases))

                for fila in datosTrain:
                    tabla[int(fila[j]), np.where (self.valoresClase == fila[-1])] += 1

                if self.laplace and np.any (tabla == 0):
                    tabla += 1
            else:                                       # Es un atributo numérico
                tabla = np.zeros((2, nClases))

                k = 0
                for l in range(nClases):
                    tabla[0][l] = np.mean (datosTrain[(datosTrain[:, -1] == self.valoresClase[k]), j])  # Media
                    tabla[1][l] = np.var (datosTrain[(datosTrain[:, -1] == self.valoresClase[k]), j])   # Varianza 

                    k += 1

            j += 1
            self.atributos.append(tabla)

    def clasifica(self, datosTest, nominalAtributos, diccionario):
        predicciones = []
        hipotesis = {}

        # (P(Xi|Hipotesis) * P(Hipotesis)
        for fila in datosTest:

            j = 0
            for i in self.aPriori:
                # P (Hipotesis)
                total = self.aPriori[i]
                k = 0
                while k < len(fila) - 1:
                    if nominalAtributos[k]:             # Es un atributo nominal
                        # Producto de P(Xi|Hipotesis)
                        # print("\nDEBUG")
                        # print(k)
                        # print(fila[k])
                        # print(j)
                        # print(len(fila))
                        # print(self.atributos)
                        total *= self.atributos[k][int(fila[k])][j] / sum(self.atributos[k][:, j])

                    else:                               # Es un atributo numérico
                        op1 = (1 / (math.sqrt (2 * math.pi * self.atributos[k][1][j])))
                        op2 = math.exp ((-(fila[k] - self.atributos[k][0][j])) / (2 * self.atributos[k][1][j]))
                        res = op1 * op2
                        total *= res

                    k += 1

                hipotesis[i] = total
                j += 1

            predicciones.append(max(hipotesis, key = hipotesis.get))

        return predicciones