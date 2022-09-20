import pandas as pd
import numpy as np

class Datos:
    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionario
    def __init__(self, nombreFichero):

    # Devuelve el subconjunto de los datos cuyos índices se pasan como argumento
    def extraeDatos(self, idx):