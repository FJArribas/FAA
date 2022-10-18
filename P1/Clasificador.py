from abc import ABCMeta, abstractmethod
import numpy as np
import math
import statistics
from Datos import Datos

class Clasificador:

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # Esta funcion debe ser implementada en cada clasificador concreto. Crea el modelo a partir de los datos de entrenamiento
    # datosTrain: matriz numpy con los datos de entrenamiento
    # nominalAtributos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def entrenamiento(self, datosTrain, nominalAtributos, diccionario):
        pass


    @abstractmethod
    # Esta funcion debe ser implementada en cada clasificador concreto. Devuelve un numpy array con las predicciones
    # datosTest: matriz numpy con los datos de validación
    # nominalAtributos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def clasifica(self, datosTest, nominalAtributos, diccionario):
        pass


    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    def error(self, datos, pred):
        # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error    
        err = 0
        n = len(datos)
        TP, TN, FP, FN = 0, 0, 0, 0 # true positive (TP), true negative (TN), # false positive (FP), # false negative (FN)

        i = 0
        while i < n:
            if datos[i][-1] != pred[i]:
                err += 1
                if self.valoresClase[0] == pred[i]:
                    FP += 1
                else:
                    FN += 1
            else:
                if self.valoresClase[0] == pred[i]:
                    TP += 1
                else:
                    TN += 1

            self.TP.append(TP)
            self.TN.append(TN)
            self.FP.append(FP)
            self.FN.append(FN)

            return err / len(datos)


    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    # TODO: implementar esta funcion
    def validacion(self, particionado, dataset, clasificador, seed=None):
        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test. Otra opción es repetir la validación simple un número especificado de veces,
        # obteniendo en cada una un error. Finalmente se calcularía la media.
        particionado.creaParticiones(dataset, seed)

        self.TP = [] # true positive (TP)
        self.TN = [] # true negative (TN)
        self.FP = [] # false positive (FP)
        self.FN = [] # false negative (FN)

        listaErrores = []

        for particion in particionado.particiones:
            # print("\nDEBUG\n")
            # print (dataset.extraeDatos(particion.indicesTrain))
            datosTrain = dataset.extraeDatos(particion.indicesTrain)
            datosTest = dataset.extraeDatos(particion.indicesTest)
            # print("\nDEBUG 2\n")

            clasificador.entrenamiento(datosTrain, dataset.nominalAtributos, dataset.diccionarios)
            predicciones = clasificador.clasifica(datosTest, dataset.nominalAtributos, dataset.diccionarios)

            listaErrores.append(self.error(datosTest, predicciones))
      
            mConf = np.array ([[statistics.mean(self.TP), statistics.mean(self.FP)], [statistics.mean(self.FN), statistics.mean(self.TN)]])

        return listaErrores, mConf

##############################################################################