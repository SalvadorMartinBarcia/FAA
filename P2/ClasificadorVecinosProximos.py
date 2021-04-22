from Clasificador import Clasificador
from collections import Counter, OrderedDict
import math
import pandas as pd 
import numpy as np
from scipy.spatial.distance import mahalanobis
from statistics import mode

class ClasificadorVecinosProximos(Clasificador):
    
  def __init__(self, distancia="euclidea", k=1, normalizacion = True):
        self.distancia = distancia
        self.k = k
        self.normalizacion = normalizacion
              
  def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        if self.normalizacion:
          super().calculaMediasDesv(datostrain, atributosDiscretos)
          self.datosTrainNorm = super().normalizarDatos(datostrain, atributosDiscretos)
        else:
          self.datosTrainNorm = datostrain

  # TODO: implementar
  def clasifica(self, datostest, atributosDiscretos, diccionario):
    predicciones = []

    if self.normalizacion:
        testAux = super().normalizarDatos(datostest, atributosDiscretos)
    else:
        testAux = datostest

    #bucle data train
    for _, rowTest in testAux.iterrows():
      dist = []
      if self.distancia == "manhattan":
        dist = self.distanciaManhattan(self.datosTrainNorm.iloc[:,:-1], rowTest[:-1])
      elif self.distancia == "mahalanobis":
        dist = self.distanciaMahalanobis(self.datosTrainNorm.iloc[:,:-1], rowTest[:-1])
      else:
        dist = self.distanciaEuclidea(self.datosTrainNorm.iloc[:,:-1], rowTest[:-1])
      
      #ahora tenemos todos los distancias de la rowTest a todos los datos de Train y queremos los k menores
      kindices = np.argsort(dist)[:self.k]
      kclases = self.datosTrainNorm.iloc[kindices,-1]

      clase = mode(kclases)
      
      predicciones.append(clase)
    return np.array(predicciones)

  def distanciaMahalanobis(self, datosTrain, rowTest):
    resultados = []
    for _, rowTrain in datosTrain.iterrows():
      v = np.cov(datosTrain.astype(float), rowvar=False)
      iv = np.linalg.inv(v)
      resultado = mahalanobis(rowTest, rowTrain, iv)
      resultados.append(resultado)
    return resultados

  def distanciaManhattan(self, datosTrain, rowTest):
    resultados = []
    for _, rowTrain in datosTrain.iterrows():
      resultado = 0
      for i, _ in enumerate(rowTest):
        resultado += abs(rowTest[i] - rowTrain[i])
      resultados.append(resultado)
    return resultados

  def distanciaEuclidea(self, datosTrain, rowTest):
    resultados = []
    for _, rowTrain in datosTrain.iterrows():
      resultado = 0
      for i, _ in enumerate(rowTest):
        resultado += (rowTest[i] - rowTrain[i])**2
      resultado = math.sqrt(resultado)
      resultados.append(resultado)
    return resultados