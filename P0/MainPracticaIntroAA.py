from Datos import Datos
from EstrategiaPatrocinado import ValidacionSimple, ValidacionCruzada

if __name__ == "__main__":
    # Primer set de datos
    dataset = Datos("ConjuntosDatosIntroFAA/tic-tac-toe.csv")

    print(dataset.nominalAtributos)
    print(dataset.diccionarios)

    print(dataset.datos)

    # Segundo set de datos
    dataset2 = Datos("ConjuntosDatosIntroFAA/german.csv")

    print(dataset2.nominalAtributos)
    print(dataset2.diccionarios)

    print(dataset2.datos)



    # EstrategiaPatrocinado
    estrategia = ValidacionSimple()
    estrategia.creaParticiones(dataset)


    print("\nValidacionSimple")
    print(estrategia)

    estrategia2 = ValidacionCruzada()
    estrategia2.creaParticiones(dataset)

    print("\nValidacionCruzada")
    print(estrategia2)