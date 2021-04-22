from Clasificador import Clasificador
from collections import Counter, OrderedDict
import math
import pandas as pd 
import numpy as np

class ClasificadorRegresionLogistica(Clasificador):
    
  def __init__(self, epocas, cAprendizaje, normalizacion=True):
    self.epocas = epocas
    self.cAprendizaje = cAprendizaje
    self.normalizacion = normalizacion

  def calculaSigmoidal(self, w, x):
    
    # Calcula producto escalar de w, x
    
    try: 
      product = np.dot(w, x)
    except OverflowError:
      if product > 0:
        return 1.
      else:
        return 0.
    denominador = 1. + np.exp(-product)
    return 1. / denominador

              
  def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
    
    if self.normalizacion:
      super().calculaMediasDesv(datostrain, atributosDiscretos)
      self.datosTrainNorm = super().normalizarDatos(datostrain, atributosDiscretos)

    # Generamos w aleatorio
    w = np.random.uniform(-0.5, 0.5, len(diccionario))

    # Empezamos bucle principal
    for epoca in range(self.epocas):
      for index, row in self.datosTrainNorm.iterrows():
        t = row['Class']
        x = np.append([1], row[:-1])
        sigmoidal = self.calculaSigmoidal(w, x)
        w = np.subtract(w, self.cAprendizaje * (sigmoidal - t) * x)
    self.w = w            
    return w

  # TODO: implementar
  def clasifica(self, datostest, atributosDiscretos, diccionario):
        
    if self.normalizacion:
      super().calculaMediasDesv(datostest, atributosDiscretos)
      self.datosTestNorm = super().normalizarDatos(datostest, atributosDiscretos)
    prediccion = []

    for index, row in self.datosTestNorm.iterrows():
      x = np.append([1], row[:-1])
      escalar = np.dot(self.w, x)
      if escalar > 0.0:
        prediccion.append(1)
      else:
        prediccion.append(0)    

    return np.asarray(prediccion)