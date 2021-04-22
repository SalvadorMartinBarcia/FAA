from abc import ABCMeta,abstractmethod
import pandas as pd 
import numpy as np
from Datos import Datos


class Clasificador:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
  def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
    pass
  
  
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto
  # devuelve un numpy array con las predicciones
  def clasifica(self,datosTest, atributosDiscretos, diccionario):
    pass
  
  
  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  # TODO: implementar
  def error(self, datos, pred):
    pred = pred.tolist()
    if not pred:
      return 1
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
    nError = 0
    for i in range(len(pred)):
      if pred[i] is not datos.iloc[i, -1]:
        nError += 1
    nError /= len(pred)
    return nError
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  def validacion(self,particionado, dataset, clasificador, seed=None):
       
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.
    
    particiones = particionado.creaParticiones(dataset.datos, seed)
    list_errores = []
    
    for particion in particiones:
      data_train = dataset.extraeDatos(particion.indicesTrain)
      data_test = dataset.extraeDatos(particion.indicesTest)
      clasificador.entrenamiento(data_train, dataset.nominalAtributos, dataset.diccionario)
      prediccion = clasificador.clasifica(data_test, dataset.nominalAtributos, dataset.diccionario)
      list_errores.append(clasificador.error(data_test, prediccion))

    return list_errores

  def calculaMediasDesv(self, datos, nominalAtributos):
    self.medias = []
    self.desviaciones = []
    # bucle columnas
    for i, atributo in enumerate(list(datos.columns)[:-1]):
      if not nominalAtributos[i]:
            self.medias.append(np.mean(datos[atributo]))
            self.desviaciones.append(np.std(datos[atributo]))

  def normalizarDatos(self, datos, nominalAtributos):
        for i, atributo in enumerate(list(datos.columns)[:-1]):
          if not nominalAtributos[i]:
                datos[atributo]=(datos[atributo]-self.medias[i])/self.desviaciones[i]
        return datos
    
   


