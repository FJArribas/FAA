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
        # TODO https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#dataframe
        # TODO https://www.w3schools.com/python/pandas/pandas_dataframes.asp
        # TODO ejemplo:
        self.datos = {
            "calories": [420, 380, 390],
            "duration": [50, 40, 45]
        }

        # Actúa como un conversor de datos nominales a datos numéricos. Para cada uno de los atributos del dataset,
        # contendrá un conjunto de pares clave-valor, donde clave es el dato nominal original del dataset y valor es
        # el valor numérico que asociamos a ese dato nominal. Si el atributo en el dataset original ya es un valor
        # continuo, su conjunto de pares clave-valor estará vacío.
        self.diccionarios = 

    # Devuelve el subconjunto de los datos cuyos índices se pasan como argumento
    def extraeDatos(self, idx):
