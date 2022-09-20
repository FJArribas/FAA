import pandas as pd
import numpy as np

class Datos:

    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionario
    def __init__(self, nombreFichero):
        # Lista de valores booleanos con la misma longitud que el número de atributos del problema,
        # True en caso de que el atributo sea nominal (discreto) y False en caso contrario (numérico)
        # True si es nominal y False si es entero o real.
        # Si algún dato no corresponde a uno de estos tres tipos, devuelve ValueError.
        self.nominalAtributos = []

        # Array bidimensional que se utilizará para almacenar los datos, con tantas filas y columnas como el archivo
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#dataframe
        # https://www.w3schools.com/python/pandas/pandas_dataframes.asp
        # TODO ejemplo:
        # self.datos = {
        #     "calories": [420, 380, 390],
        #     "duration": [50, 40, 45]
        # }
        self.datos = None

        # Actúa como un conversor de datos nominales a datos numéricos. Para cada uno de los atributos del dataset,
        # contendrá un conjunto de pares clave-valor, donde clave es el dato nominal original del dataset y valor es
        # el valor numérico que asociamos a ese dato nominal. Si el atributo en el dataset original ya es un valor
        # continuo, su conjunto de pares clave-valor estará vacío.
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
            elif i == np.float64 or i == np.int64:      # np.int64 es un atributo numérico
                self.nominalAtributos.append(False)
            else:
                raise ValueError("Tipo de dato incorrecto en el conjunto de datos")

        j = 0

        # Por cada atributo de nuestro set de datos:
        for atributo in self.nominalAtributos:
            # Creamos un diccionario vacío por cada atributo
            aux_dict = {}
            # Si el atributo es nominal
            if self.nominalAtributos[atributo] == True:
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
        self.datos = df.replace(self.diccionarios)

    # Devuelve el subconjunto de los datos cuyos índices se pasan como argumento
    def extraeDatos(self, idx):
        pass
