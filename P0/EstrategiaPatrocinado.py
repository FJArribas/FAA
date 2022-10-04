from abc import ABCMeta, abstractmethod
import random
from Datos import Datos

class Particion():
    # Esta clase mantiene la lista de índices de Train y Test para cada partición del conjunto de particiones  
    def __init__(self):
        self.indicesTrain=[]
        self.indicesTest=[]

    def __str__(self):
        return "\nTrain:\n" + str(self.indicesTrain) + "\nTest:\n" + str(self.indicesTest) + "\n"

#####################################################################################################

# Clase abstracta - contiene las clases de validación
class EstrategiaParticionado:
    __metaclass__ = ABCMeta

    # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor
    @abstractmethod
    def creaParticiones(self, datos, seed=None):
        pass

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):
    def __init__(self, pTest = 20, nEjecuciones = 1):
        self.pTest = pTest  # Porcentaje de datos de prueba usados
        self.nEjecuciones = nEjecuciones
        self.particiones=[]

    def __str__(self):
        x = ""
        for particion in self.particiones:
            x += "".join(particion.__str__())
        return x

    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el número de ejecuciones deseado
    # Devuelve una lista de particiones (clase Particion)
    def creaParticiones(self, datos, seed=None):
        random.seed(seed)

        rows = datos.datos.shape[0]
        index = list(range(rows))

        porcentaje = (rows * self.pTest) // 100

        # El bucle hace nEjecuciones
        for i in range(self.nEjecuciones):
            # Debemos permutar las filas para evitar sesgos
            random.shuffle(index)

            # Creamos la particion y la añadimos a la lista
            particion = Particion()
            particion.indicesTrain = index[porcentaje:]
            particion.indicesTest = index[:porcentaje]

            self.particiones.append(particion)

#####################################################################################################

class ValidacionCruzada(EstrategiaParticionado):
    def __init__(self, k = 5):
        self.k = k
        self.particiones=[]

    def __str__(self):
        x = ""
        for particion in self.particiones:
            x += "".join(particion.__str__())
        return x

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    def creaParticiones(self, datos, seed=None):
        random.seed(seed)

        rows = datos.datos.shape[0]
        index = list(range(rows))

        proporcion = rows // self.k

        # Debemos permutar las filas para evitar sesgos
        random.shuffle(index)

        # El bucle hace k veces
        for i in range(self.k):
            # Creamos la particion y la añadimos a la lista
            particion = Particion()

            # Ponemos todos los indices en la lista de Train - Después, eliminamos los que pertenecen a Test
            indicesTrain = index.copy()

            if i == (self.k - 1):
                indicesTest = index[i * proporcion:]
                del indicesTrain[i*proporcion:]
            else: 
                indicesTest = index[i * proporcion:(i+1) * proporcion]
                del indicesTrain[i * proporcion:(i+1) * proporcion]

            particion.indicesTrain = indicesTrain
            particion.indicesTest = indicesTest

            self.particiones.append(particion)