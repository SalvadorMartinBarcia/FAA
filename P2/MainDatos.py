from Datos import Datos
from ValidacionCruzada import ValidacionCruzada
from ValidacionSimple import ValidacionSimple
from ClasificadorNaiveBayes import ClasificadorNaiveBayes
from ClasificadorVecinosProximos import ClasificadorVecinosProximos
from ClasificadorRegresionLogistica import ClasificadorRegresionLogistica


if __name__ == '__main__':
    dataset = Datos('data/pima-indians-diabetes.data')
    dataset3 = Datos('data/wdbc.data')
    estrategia = ValidacionCruzada(4)
    clasificador = ClasificadorVecinosProximos(distancia="mahalanobis")
    errores_indians = clasificador.validacion(estrategia, dataset, clasificador)
    estrategia1 = ValidacionCruzada(4)
    clasificador1 = ClasificadorVecinosProximos(distancia="mahalanobis")
    errores_wdbc = clasificador1.validacion(estrategia1, dataset3, clasificador1)
    print ('indians: ', errores_indians)
    print ('wdbc: ', errores_wdbc)

