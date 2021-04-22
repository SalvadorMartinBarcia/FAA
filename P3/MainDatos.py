# pylint: skip-file
from Datos import Datos
from ValidacionCruzada import ValidacionCruzada
from ValidacionSimple import ValidacionSimple
from ClasificadorNaiveBayes import ClasificadorNaiveBayes
from ClasificadorVecinosProximos import ClasificadorVecinosProximos
from ClasificadorRegresionLogistica import ClasificadorRegresionLogistica
from ClasificadorAlgoritmoGenetico import ClasificadorAlgoritmoGenetico
import numpy as np
import time


if __name__ == '__main__':
    dataset = Datos('data/tic-tac-toe.data')
    dataset3 = Datos('data/titanic.data')
    estrategia = ValidacionSimple(0.7, 1)
    clasificador = ClasificadorAlgoritmoGenetico(0.5, 0.1, nGeneraciones=20, nReglas=20, nPoblacion=50, debug=True)
    ini = time.time()
    print('Inicio: ', ini)
    print('Error: ', clasificador.validacion(estrategia, dataset3, clasificador))
    print(clasificador.mejorIndividuo)
    print(clasificador.mejorIndividuoFitness)
    fin = time.time()
    print('Fin: ', fin, ' Duracion:', fin-ini)


    # particiones = estrategia.creaParticiones(dataset.datos, None)
    # for particion in particiones:
    #     data_train = dataset.extraeDatos(particion.indicesTrain)
    #     poblacion = clasificador.inicializaPoblacion(data_train)
    #     break

    
    # for i, p in enumerate(poblacion):
    #     print('Individuo ', i, ' con ', len(p), ' reglas: \n')
        
    #     for j, r in enumerate(p):
    #         print('Regla ', j, ': ', r, '\n')


    
    # # print('Individuo ', p1, ' con ', len(poblacion[p1]), ' reglas\n')
    # # print('Individuo ', p2, ' con ', len(poblacion[p2]), ' reglas\n')
    

    # # r1 = np.random.randint(0, len(poblacion[p1]))
    # # r2 = np.random.randint(0, len(poblacion[p2]))

    # # print('Individuo ', p1, ' con regla ', r1, ': ', poblacion[p1][r1], '\n')
    # # print('Individuo ', p2, ' con regla ', r2, ': ', poblacion[p2][r2], '\n')
    
    # # print('Individuo ', p1, ' con reglas ', '\n')
    # # for j, r in enumerate(poblacion[p1]):
    # #     print('Regla ', j, ': ', r, '\n')

    # # print('Individuo ', p2, ' con reglas \n')
    # # for j, r in enumerate(poblacion[p2]):
    # #     print('Regla ', j, ': ', r, '\n')

    # # v1, v2 = clasificador.intraCruce(poblacion[p1], poblacion[p2])

    # # print('Individuo ', p1, ' con vástago ', v1, '\n')
    # # print('Individuo ', p2, ' con vástago ', v2, '\n')

    # # pp = clasificador.mutacionEstandar(poblacion, False)

    # # for i, p in enumerate(pp):
    # #     print('Individuo ', i, ' con ', len(p), ' reglas: \n')
        
    # #     for j, r in enumerate(p):
    # #         print('Regla ', j, ': ', r, '\n')

    # fitness, m_fitness = clasificador.fitnessPoblacion(data_train, poblacion)
    # print(m_fitness)
    # # g = clasificador.seleccionProporcional(poblacion, fitness)
    # # fitness, m_fitness = clasificador.fitnessPoblacion(data_train, g)
    # # print(m_fitness)

    # e = clasificador.elitismo(poblacion, fitness)
    # print(len(e))






