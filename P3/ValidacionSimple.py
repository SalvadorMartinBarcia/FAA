from EstrategiaParticionado import EstrategiaParticionado
from Particion import Particion
import random


class ValidacionSimple(EstrategiaParticionado):
      
  def __init__(self, proporcionTest, numeroEjecuciones):
      self.proporcionTest = proporcionTest
      self.numeroEjecuciones = numeroEjecuciones
      self.particiones = []

  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el n�mero de ejecuciones deseado
  # Devuelve una lista de particiones (clase Particion)
  def creaParticiones(self,datos,seed=None):
    indices = list(datos.index)
    nFilas = len(indices)
    nTrain = round(nFilas * self.proporcionTest)
    random.seed(seed)
    for i in range(self.numeroEjecuciones):
      random.shuffle(indices)
      particion = Particion()
      for j, indice in enumerate(indices):
        if j <= nTrain:
          particion.indicesTrain.append(indice)
        else:
          particion.indicesTest.append(indice)
      self.particiones.append(particion)
    return self.particiones
