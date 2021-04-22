from Clasificador import Clasificador
from collections import Counter, OrderedDict
import math
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from scipy.stats import mode
import copy
import random

class ClasificadorAlgoritmoGenetico(Clasificador):
    
  def __init__(self, pCruce, pMutacion, pElitismo=0.05, nGeneraciones=1, nReglas=1, nPoblacion=100, intra=True, todas=True, debug=False):
    self.pCruce = pCruce
    self.pMutacion = pMutacion
    self.pElitismo = pElitismo
    self.nGeneraciones = nGeneraciones
    self.nReglas = nReglas
    self.nPoblacion = nPoblacion
    self.intra = intra
    self.todas = todas
    self.debug = debug
    self.mejorIndividuoFitness = []
    self.mediaFitness = []



  def calculaLongitudReglas(self, datos):
    
    n_unicos = []
    for _, col in enumerate(list(datos.columns)[:-1]):
      n_unicos.append(len(datos[col].unique()))

    self.longitudReglas = sum(n_unicos) + 1 # Como trabajamos con clases binarias, añadimos 1 bit extra

    return n_unicos

  def creaReglas(self, datos):
    reglas = []
    nAtr = self.calculaLongitudReglas(datos) # Lista con valores por atributo

    n_reglas = np.random.randint(1, self.nReglas)

    for _ in range(n_reglas):
      # Generamos reglas
      aux = []
      for atr in nAtr: # con n bits de atributos + 1 bit de clase
        number = np.random.randint(2, size=atr).tolist() # Generamos numero aleatorio entre 0 y 1 de tamaño atr
        while all(number) or not any(number): # Mientras sea todo unos o todo ceros
          number = np.random.randint(2, size=atr).tolist() # Generamos otro
        aux.extend(number)
      
      aux.append(np.random.randint(2)) # Metemos clase

      reglas.append(aux) # Conjunto de n reglas


    
    return reglas

  def inicializaPoblacion(self, datos):
    poblacion = []

    for _ in range(self.nPoblacion): # Por cada individuo
      poblacion.append(self.creaReglas(datos)) # Creamos reglas
    
    return poblacion

  def cruce(self, r_p1, r_p2):

    # Calculamos punto de cruce aleatorio
    punto = np.random.randint(1, len(r_p1))
    v1, v2 = [], []
    # Probabilidad
    p = np.random.uniform(0, 1)
    if p <= self.pCruce: # Hacemos cruce
      v1.extend(r_p1[:punto])
      v1.extend(r_p2[punto:])
      v2.extend(r_p2[:punto])
      v2.extend(r_p1[punto:])
      return v1, v2

    return r_p1, r_p2

  def intraCruce(self, p1, p2):
    # Se intercambia material de una sola regla por progenitor

    # Escogemos aleatoriamente las reglas a cruzar
    r_p1 = np.random.randint(0, len(p1)) 
    r_p2 = np.random.randint(0, len(p2))

    # Cruzamos en 1 punto
    v1, v2 = self.cruce(p1[r_p1], p2[r_p2])

    p1[r_p1] = v1
    p2[r_p2] = v2

    return p1, p2

  def interCruce(self, p1, p2):
    
    # Calculamos punto de cruce aleatorio (regla)
    minimo = min(len(p1), len(p2))
    if minimo is 1:
      punto = 0
    else:
      punto = np.random.randint(1, minimo)

    # Intercambiamos reglas
    v1, v2 = [], []

    # Probabilidad
    p = np.random.uniform(0, 1)
    if p <= self.pCruce: # Hacemos cruce
      v1.extend(p1[:punto])
      v1.extend(p2[punto:])
      v2.extend(p2[:punto])
      v2.extend(p1[punto:])
      return v1, v2

    return p1, p2

  def crucePoblacion(self, poblacion, intra=True):

    p_aux = []
    poblacion_aux = copy.deepcopy(poblacion)

    while len(poblacion_aux) > 1:
      n1 = np.random.randint(len(poblacion_aux))
      n2 = np.random.randint(len(poblacion_aux))
      while n2 is n1:
        n2 = np.random.randint(len(poblacion_aux))
    
      if intra:
        p_aux.extend(self.intraCruce(poblacion_aux[n1], poblacion_aux[n2]))
      else:
        p_aux.extend(self.interCruce(poblacion_aux[n1], poblacion_aux[n2]))
      
      poblacion_aux.pop(n1)
      if n1 < n2:
        poblacion_aux.pop(n2-1)
      else:
        poblacion_aux.pop(n2)
    
    if poblacion_aux:
      p_aux.extend(poblacion_aux)
    return p_aux

  def mutacionIndividuo(self, individuo, todas=True):

    # Mutacion para una sola regla
    if not todas:
      regla_m = np.random.randint(0, len(individuo)) # elegimos regla a mutar
      if self.debug:
        print(regla_m)

      for i, bit in enumerate(individuo[regla_m]): # Vamos bit a bit
        p = np.random.uniform(0, 1)
        if p <= self.pMutacion: # Si muta
          # Hacemos bit flip
          if bit is 1:
            individuo[regla_m][i] = 0
          else:
            individuo[regla_m][i] = 1
      
      return individuo

    # Mutacion para todas las reglas
    else:

      for j, regla in enumerate(individuo): # Por cada regla
        for i, bit in enumerate(regla): # Vamos bit a bit
          p = np.random.uniform(0, 1)
          if p <= self.pMutacion: # Si muta
            # Hacemos bit flip
            if bit is 1:
              regla[i] = 0
            else:
              regla[i] = 1
        individuo[j] = regla
        
      return individuo

  def mutacionEstandar(self, poblacion, todas=True):
    p_mutada = []
    pob_aux = copy.deepcopy(poblacion)
    for p in pob_aux: # Por cada individuo
      p_m = self.mutacionIndividuo(p, todas) # Mutamos reglas si es pertinente
      p_mutada.append(p_m)

    return p_mutada

  def fitness(self, datos, individuo):

    # Codificamos datos de entrada con OneHotEncoder
    encodedDataset = preprocessing.OneHotEncoder(sparse=False)
    data = datos
    encodedDataset.fit(data)
    data = encodedDataset.transform(data)

    # Convertimos a lista para facilitar las comparaciones
    data = np.array(data, dtype=int).tolist()

    nValoresAtr = self.calculaLongitudReglas(datos) # Lista con longitud por atributo
    aciertos = 0
    predClases = [] # predicciones

    for row in data:
      reglasActivadas = [] # Lista con las clases de las reglas que se activen por inidivuo por fila en el dataset
      for regla in individuo:
        limiteInf = 0
        orAtr = []
        for atr in nValoresAtr: # Comparamos atributo a atributo
          flag = []
          for i in range(limiteInf, limiteInf+atr): # Comparamos bit a bit
            flag.append( all([row[i], regla[i]]) ) # Solo es True si 1 - 1
          
          orAtr.append(any(flag)) # Es True si alguno es True
          limiteInf += atr

        if all(orAtr): # Es True si todos son True
          reglasActivadas.append(regla[-1]) # Guardamos clase

      if reglasActivadas: # Si se han activado
        moda = mode(reglasActivadas) # Cogemos el valor que más se repite de clase
        if moda[0][0] == row[-1]: # Si coincide
          aciertos += 1 # Es un acierto
        predClases.append(moda[0][0]) # Metemos predicciones que serán utiles en clasifica
      else: # Si no, lanzamos una moneda
        predClases.append(random.choice([0, 1]))

    return aciertos/len(data), predClases
  
  def fitnessPoblacion(self, datos, poblacion):
    f = []
    for individuo in poblacion: # Por cada inidividuo
      fitness, _ = self.fitness(datos, individuo) # Ignoramos predicciones, pues estamos entrenando
      f.append(fitness) # Calculamos fitness

    return f, np.mean(f) # Devolvemos lista con fitness y la media

  def seleccionProporcional(self, poblacion, fitness, sizeG):
    sum_fitness = np.sum(fitness)
    generacion = []
    
    for _ in range(sizeG): # Generamos generacion de n inidividuos
      punto = np.random.uniform(0, sum_fitness) # Cogemos un punto donde cortar (aleatorio)
      suma = 0
      for i, f in enumerate(fitness):
        suma += f
        if suma >= punto: # Si llegamos al punto
          generacion.append(poblacion[i]) # Seleccionamos individuo
          break
    
    return generacion # Devolvemos siguiente generacion

  def elitismo(self, poblacion, fitness):
    cantidad = math.ceil(self.nPoblacion * self.pElitismo) # Calculamos indidivuos pertenecientes a la elite
    indicesMejores = np.argsort(fitness)[self.nPoblacion - cantidad:] # Ordenamos y cogemos la elite
    p = np.array(poblacion)
    f = np.array(fitness)
    return p[indicesMejores].tolist(), cantidad # Devolvemos individuos y cantidad

  def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
    poblacion = self.inicializaPoblacion(datostrain) # Inicializamos población

    g = 0
    while g < self.nGeneraciones:
      p_aux = []
      # Calculamos fitness
      fitness, m_fitness = self.fitnessPoblacion(datostrain, poblacion)
      indiceMejor = np.argsort(fitness)[-1] # Mejor individuo de la generación
      if self.debug:
        print('Generacion: ', g, ' Media fitness: ', m_fitness, ' Mejor indv: ', round(fitness[indiceMejor], 4))
      
      # Si hay elitismo, lo aplicamos y reservamos los individuos
      if self.pElitismo > 0.:
        elite, nElite = self.elitismo(poblacion, fitness)
      else:
        nElite = 0
    
      poblacion = self.seleccionProporcional(poblacion, fitness, self.nPoblacion - nElite) # Seleccionamos progenitores
      cruzados = self.crucePoblacion(poblacion, self.intra) # Cruzamos según pCruce
      mutados = self.mutacionEstandar(cruzados, self.todas) # Mutamos según pMutación
      # Guardamos siguiente generación
      if nElite > 0:
        p_aux.extend(elite)
      p_aux.extend(mutados)


      poblacion = copy.deepcopy(p_aux)
      indiceMejor = np.argsort(fitness)[-1]
      self.mejorIndividuoFitness.append(round(fitness[indiceMejor], 4))
      self.mediaFitness.append(m_fitness)
      g += 1

    # Cogemos al mejor individuo
    fitness, m_fitness = self.fitnessPoblacion(datostrain, poblacion)
    if self.debug:
      print('Media fitness ultima poblacion: ', m_fitness)
    indiceMejor = np.argsort(fitness)[-1]
    if self.debug:
      print('Fitness mejor ind final: ', round(fitness[indiceMejor], 4), ' Max fitness: ', round(max(fitness), 4))
    self.mejorIndividuo = poblacion[indiceMejor] # Guardamos



  def clasifica(self,datostest,atributosDiscretos,diccionario):
    
    _, prediccion = self.fitness(datostest, self.mejorIndividuo) # prediccion contiene las clases que han dado acierto (predicciones)

    return np.array(prediccion) # Devolvemos
    
