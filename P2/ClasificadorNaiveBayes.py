from Clasificador import Clasificador
from collections import Counter, OrderedDict
import math
import pandas as pd 
import numpy as np

class ClasificadorNaiveBayes(Clasificador):
    
  def __init__(self, laplace=False):
    self.laplace = laplace
        

  # TODO: implementar
  def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
    nDatos = len(datostrain) # número de datos
    nAtributos = len(atributosDiscretos) - 1 # número de atributos sin tomar en cuenta la clase
    nClass = len(diccionario['Class']) # número de clases unicas
    
    # creamos una tabla por cada clase de a prioris
    self.tPrioris = np.empty(nClass, dtype=object)
    listClass = list(datostrain['Class']) # lista de clases completa
    # probabilidades a priori de cada clase
    for i, value in enumerate(diccionario['Class'].values()):
      self.tPrioris[i] = listClass.count(value)/nDatos

    # tabla para calcular las probabilidades a posteriori
    self.tPosterioris = np.empty(nAtributos, dtype=object)
    # bucle que rellena las tablas a posteriori con los datos continuos o discretos
    for i, atributo in enumerate(list(datostrain.columns)[:-1]):
      
      # Atributos discretos
      if atributosDiscretos[i]:
        tPosterioriAtr = np.zeros((nClass, len(diccionario[atributo]))) # tabla del atributo: nClass x num de posibles valores del atributo
        nAtrClase = {} # numero de atributos por clase

        # bucle que cuenta el numero de datos de cada atributo por clase
        for j, clas in enumerate(diccionario['Class'].values()):
          datosClase = datostrain[datostrain['Class']==clas] # datos que sean de la clase correspondiente
          datosAtrClase = datosClase[atributo] # datos de la clase y atributo correspondiente
          nAtrClase = Counter(list(datosAtrClase)) # numero de datos de cada atributo por cada clase

          # bucle que introduce los datos en la tabla de posterioriatrib
          for value in range(len(diccionario[atributo])):
            tPosterioriAtr[j, value] = nAtrClase[value]

        # si pide laplace y hay algun 0 en la tabla suma 1 a todos
        if self.laplace and (tPosterioriAtr == 0).any():
          tPosterioriAtr += 1
        
        # guardamos la probabilidad sobre la clase en las tablas
        for j in range(nClass):
          tPosterioriAtr[j] /= sum(tPosterioriAtr[j])

        # guardar la tabla del atributo en el array de todas las tablas
        self.tPosterioris[i] = tPosterioriAtr
      # Atributos continuos
      else:
        tPosterioriAtr = np.zeros((nClass, 2)) # tabla del atributo: nClass x num de posibles valores del atributo
        # bucle que cuenta el numero de datos de cada atributo por clase
        for j, clas in enumerate(diccionario['Class'].values()):
          datosClase = datostrain[datostrain['Class']==clas] # datos que sean de la clase correspondiente
          datosAtrClase = datosClase[atributo] # datos de la clase y atributo correspondiente
          tPosterioriAtr[j, 0] = np.mean(datosAtrClase) # media
          tPosterioriAtr[j, 1] = np.std(datosAtrClase) # desviacion tipica
        self.tPosterioris[i] = tPosterioriAtr
    
  # TODO: implementar
  def clasifica(self,datostest,atributosDiscretos,diccionario):
    predicciones = []
    
    # bucle filas
    for row in datostest.iterrows():
      allResult = {}
      #bucle clases
      for i, value in enumerate(diccionario['Class'].values()):
        resultado = 1

        # bucle columnas
        for j, atributo in enumerate(list(datostest.columns)[:-1]):
          if atributosDiscretos[j]:
            resultado *= self.tPosterioris[j][i, row[1][j]] 
          
          else:
            if self.tPosterioris[j][i,1] != 0:
              raiz = math.sqrt(2 * (np.pi) * (self.tPosterioris[j][i,1]**2))
              numerador = (row[1][j] - self.tPosterioris[j][i,0])**2
              denominador = 2 * (self.tPosterioris[j][i,1]**2)
              exponente = math.exp((-numerador)/denominador)
              resultado *= (exponente/raiz)


        resultado *= self.tPrioris[i]
        allResult[value] = resultado
      
      predicciones.append(max(allResult, key=allResult.get))

    return np.array(predicciones)