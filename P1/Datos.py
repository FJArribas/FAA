import pandas as pd
import numpy as np

class Datos:
    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero):
        self.nominalAtributos = []
        self.datos = None
        self.diccionarios = {}

        # Cargamos el set de datos y creamos el array bidimensional usando Pandas
        df = pd.read_csv(nombreFichero)
        self.datos = np.array(df)

        columnas = []
        for i in df.head():
            columnas.append(i)

        for i in df.dtypes:
            if i == np.object:                          # np.object es un atributo nominal
                self.nominalAtributos.append(True)
            elif i == np.int64 or i == np.float64:      # np.int64 es un atributo numérico
                self.nominalAtributos.append(False)
            else:
                raise ValueError("Tipo de dato incorrecto en el conjunto de datos")

        j = 0

        # Por cada atributo de nuestro set de datos:
        for atributo in self.nominalAtributos:
            # Creamos un diccionario vacío por cada atributo
            aux_dict = {}
            # Si el atributo es nominal
            if atributo == True:
                # Convertir nominal a numerico
                aux_set = set()
                k = 0

                # Añadimos al set cada dato del atributo nominal
                for dato in self.datos:
                    aux_set.add(dato[j])

                # Hacemos sorted() al set, y ahora que está ordenado, añadimos los datos al diccionario
                for aux_dato in sorted(aux_set):
                    aux_dict[aux_dato] = k
                    k += 1
            else:
                pass # Atributo númérico, dejarlo vacío

            self.diccionarios[columnas[j]] = aux_dict
            j += 1

        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
        self.datos = np.array(df.replace(self.diccionarios))

    # Devuelve el subconjunto de los datos cuyos índices se pasan como argumento
    def extraeDatos(self, idx):
        return self.datos[idx]