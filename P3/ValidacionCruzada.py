from EstrategiaParticionado import EstrategiaParticionado
import random
from Particion import Particion

class ValidacionCruzada(EstrategiaParticionado):
      
  def __init__(self, numeroParticiones):
    self.numeroParticiones = numeroParticiones
    self.particiones = []  
  
      
  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  # 70%
  def creaParticiones(self,datos,seed=None):   
    random.seed(seed)
    nFilas = datos.shape[0] #(10, 1)
    nPart =  nFilas // self.numeroParticiones
    indices = list(datos.index)
    random.shuffle(indices)

    if self.numeroParticiones <= 1:
      raise Exception('ValueError: El número de particiones tiene que ser mayor a 1')

    for k in range(self.numeroParticiones):
      particion = Particion()
      
      if k == 0:
        particion.indicesTest.extend(indices[(k * nPart): (nPart * (k + 1))])
        particion.indicesTrain.extend(indices[(nPart * (k + 1)):])
      elif k == (self.numeroParticiones - 1):
        particion.indicesTest.extend(indices[(k * nPart): ])
        particion.indicesTrain.extend(indices[:(k * nPart)])
      else:
        particion.indicesTest.extend(indices[(k * nPart): (nPart * (k + 1))])
        particion.indicesTrain.extend(indices[:(k * nPart)])
        particion.indicesTrain.extend(indices[(nPart * (k + 1)):])
      
      self.particiones.append(particion)
    
    return self.particiones







    