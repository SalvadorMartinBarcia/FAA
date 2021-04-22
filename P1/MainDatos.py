from Datos import Datos
from ValidacionCruzada import ValidacionCruzada
from ValidacionSimple import ValidacionSimple
from ClasificadorNaiveBayes import ClasificadorNaiveBayes


if __name__ == '__main__':
    dataset = Datos('data/tic-tac-toe.data')
    dataset2 = Datos('data/german.data')
    estrategia=ValidacionCruzada(3)
    clasificador=ClasificadorNaiveBayes(True)
    errores=clasificador.validacion(estrategia,dataset2,clasificador)
    print (errores)

