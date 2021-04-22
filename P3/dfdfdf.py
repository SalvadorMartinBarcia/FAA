#!/usr/bin/python
# -*- coding: <encoding name> -*-
from abc import ABCMeta,abstractmethod
from EstrategiaParticionado import EstrategiaParticionado, Particion, ValidacionSimple, ValidacionCruzada
from Datos import Datos
import pandas as pd
import numpy as np
import collections
import math
import random
from scipy.stats import norm, mode
from sklearn import preprocessing
import copy



# imporscipy.spatial.distance as distance



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
  def clasifica(self,datosTest,atributosDiscretos,diccionario):
    pass



# Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  # TODO: implementar
  def error(self,datos,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
    # se comparan los valores de predicción y perror = fallos/totales :D
    errores = 0
    i = 0
    self.clases = []
    self.matrizConfusion = {}
    # Inicializamos la matriz de confusión
    for cell in np.unique(datos[:,-1]):
      #self.matrizConfusion["Verdaderos " + str(int(cell))] = 0
      #self.matrizConfusion["Falsos " + str(int(cell))] = 0
      self.clases.append(str(cell))
    for p in pred:
      if p != datos[i, -1]:
        #self.matrizConfusion["Falsos " + str(p)] += 1
        errores = errores + 1
      #else:
        #self.matrizConfusion["Verdaderos " + str(p)] += 1
      i += 1

    error = errores / len(pred)
    return error


  def calcular_tasas(self):
      self.TPR = self.matrizConfusion["Verdaderos " + self.clases[0][0]] / (self.matrizConfusion["Verdaderos " + self.clases[0][0]] + self.matrizConfusion["Falsos " + self.clases[1][0]])
      self.FNR = self.matrizConfusion["Falsos " + self.clases[1][0]] / (self.matrizConfusion["Verdaderos " + self.clases[0][0]] + self.matrizConfusion["Falsos " + self.clases[1][0]])
      self.FPR = self.matrizConfusion["Falsos " + self.clases[0][0]] / (self.matrizConfusion["Verdaderos " + self.clases[1][0]] + self.matrizConfusion["Falsos " + self.clases[0][0]])
      self.TNR = self.matrizConfusion["Verdaderos " + self.clases[1][0]] / (self.matrizConfusion["Verdaderos " + self.clases[1][0]] + self.matrizConfusion["Falsos " + self.clases[0][0]])

  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  def validacion(self,particionado,dataset,clasificador,seed=None):

    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.

    error = []

    particionado.creaParticiones(dataset.datos)

    for p in particionado.particiones:
      clasificador.entrenamiento(dataset.extraeDatos(p.indicesTrain), dataset.nominalAtributos, dataset.diccionario)
      datostest = dataset.extraeDatos(p.indicesTest)
      pred = clasificador.clasifica(datostest, dataset.nominalAtributos, dataset.diccionario)
      error.append(self.error(datostest, pred))

    return error

  """Si bien no se usará para todos los Clasificadores, son métodos bastante generalistas
  por lo que en futuras ocasiones se pueden necesitar en Clasificadores distintos de Knn. Además,
  por su funcionalidad es de una capa superior a la clase Datos, y no tienen relación con las
  particiones. Otra opción sería añadir una clase nueva, pero para dos métodos nos parece
  redundante"""

  """Esta función calculará las medias y desviaciones típicas de uno o varios atributos
  continuos según lo que se pase en el argumento datos."""
  def calcularMediasDesv(self,datos,nominalAtributos):
    for attr in range(datos.shape[1]):
      contador = []
      for dato in datos:
        contador.append(dato[attr])
      self.medias.append(np.mean(contador))
      self.desviacion.append(np.std(contador))

  """Esta función normalizará cada uno de los atributos continuos en la matriz datos
  utilizando las medias y desviaciones típicas obtenidas en calcularMediasDesv"""
  def normalizarDatos(self,datos,nominalAtributos):
    for dato in datos:
      for attr in range(datos.shape[1]-1):
        dato[attr] -= self.medias[attr]/self.desviacion[attr]

##############################################################################

class ClasificadorNaiveBayes(Clasificador):

  def __init__(self, laplace=0):
        self.laplace = laplace


  # TODO: implementar
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
    numFilas = float(datostrain.shape[0])
    clases = datostrain[:,-1]
    self.nominalAtributos = atributosDiscretos
    #Cantidad de atributos
    nAtributos = datostrain.shape[1] - 1
    #Cantidad de clases
    self.nClases = len(clases)
    #Lista de los índices que tenemos en datos train
    lista_indices = range(len(datostrain))
    #El último valor de la lista de valores es la cantidad de índices
    nIndices = lista_indices[-1]

    priori = np.zeros(self.nClases)
    #Calculamos probabilidad a priori
    i = 0
    for clase in clases:
      priori[i] = len(np.where(clases == i)[0]) / self.nClases
      i += 1

    #Calculamos probabildad a posteriori
    posteriori = np.zeros(self.nClases,dtype=object)
    i = 0
    for atributo in atributosDiscretos[:-1]:
      #En caso de que sea discreto
      if atributo == True:
        nValores = len(diccionario[i])
        attrPosteriori = np.zeros((self.nClases, nValores))
        for row in datostrain:
          attrPosteriori[int(row[-1]),int(row[i])] += 1

        if(self.laplace) and (attrPosteriori == 0).any():
          #Se suma 1 a todos los valores para que no haya 0s
          attrPosteriori += 1

      #En caso de que sea continuo
      else:
        #Usamos 2 porque solo se necesita la media y la desviación
        attrPosteriori = np.zeros((self.nClases, 2))
        for iClase in np.unique(clases):
          contador = datostrain[clases == iClase]
          attrPosteriori[int(iClase)][0] = np.mean(contador)
          attrPosteriori[int(iClase)][1] = np.std(contador)


      posteriori[i] = attrPosteriori
      i += 1


    if numFilas == 0:
      raise ValueError("There must be data to be able to train")

    self.posterori = posteriori
    self.priori = priori


  # TODO: implementar

  def calculaProbabilidadClase(self, dato):

    # Siendo iClase el índice de la clase con el que se está trabajando en esta iteración
    for iClase in range(self.nClases):
      p = np.copy(self.priori)
      # Siendo iAttr el índice del atributo con el que se está trabajando en esta iteración
      for iAttr, n in enumerate(self.nominalAtributos[:-1]):
        if n:
          p[iClase] = p[iClase] * self.posterori[iAttr][iClase, int(dato[iAttr])]
        else:
          if(self.posterori[iAttr][iClase, 0] != 0):
            p[iClase] = p[iClase] * norm.pdf(dato[iAttr], self.posterori[iAttr][iClase, 0], self.posterori[iAttr][iClase, 1])

  
    return p

  def clasifica(self,datostest,atributosDiscretos,diccionario):

    index = range(len(datostest))
    clases = np.full(len(index), -1)

    for iClase, dato in enumerate(datostest):
      p = self.calculaProbabilidadClase(dato)
      #ES POR AQUI
      clases[iClase-1] = np.argmax(p)


    return clases

class ClasificadorVecinosProximos(Clasificador):

  def __init__(self, k, distanc):
    self.k = k
    self.distancia = distanc
    self.medias = []
    self.desviacion = []


  #Habrá que poder calcular la distancia Euclídea, Manhattan y Mahalanobis.
  def calcular_distancia_euclidea(self,objetivo, dato):
    resultado = 0
    for attr in range(len(dato)-1):
      resultado += math.pow((dato[attr] - objetivo[attr]), 2)

    resultado = math.sqrt(resultado)
    return resultado

  def calcular_distancia_manhattan(self,objetivo, dato):
    resultado = 0
    for attr in range(len(dato)-1):
      resultado += max(objetivo[attr],dato[attr]) - min(objetivo[attr],dato[attr])

    return resultado

  def calcular_distancia_mahalanobis(self,objetivo, dato):
    resultado = 0
    array = np.stack((objetivo[:-1], dato[:-1]), axis=-1)
    covarianza = np.cov(array.astype(float))
    #DUDAS
    #invcovarianza = np.linalg.inv(covarianza)

    resultado = distance.mahalanobis(objetivo[:-1], dato[:-1], covarianza.T)
    return resultado

  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
    super().calcularMediasDesv(datostrain,atributosDiscretos)
    super().normalizarDatos(datostrain,atributosDiscretos)
    self.datostrain = datostrain

  def clasifica(self,datostest,atributosDiscretos,diccionario):
    # Se inicializan las variables
    index = range(len(datostest))
    clases = np.full(len(index), -1)
    i = 0
    # Se itera sobre todos los datos sobre los que obtener la clase, es decir, de test
    for dato in datostest:
      lista = []
      # Se itera sobre todos los elementos de entrenamiento, para calcular las distancias
      # respecto al dato del que se quiere conocer la clase
      for dato_comparar in self.datostrain:

        # Existen 3 métodos diferentes para calcular distancias
        if self.distancia == 0:
          distancia = self.calcular_distancia_euclidea(dato,dato_comparar)
        elif self.distancia == 1:
          distancia = self.calcular_distancia_manhattan(dato,dato_comparar)
        else:
          distancia = self.calcular_distancia_mahalanobis(dato,dato_comparar)

        sub = (distancia,dato_comparar[-1])
        lista.append(sub)

      # Se ordena la lista de distancia para los menores primero, clase
      lista.sort(key=lambda x: x[0])
      # Se hace una segunda lista con los más cercanos, es decir, menor distancia
      k_vecinos = []
      for ele in range(self.k):
        k_vecinos.append(lista[ele][1])
      # Se cuenta el número de veces qué aparece cada clase en los vecinos próximos
      contador = collections.Counter(k_vecinos)
      # Se obtiene la clase que más aparece
      clases[i] = list(contador.elements())[0]
      i += 1

    return clases



class ClasificadorRegresionLogistica(Clasificador):
    def __init__(self, w=None, cteAprend=1, nEpocas=1):
        self.w = w
        self.cteAprend = cteAprend
        self.nEpocas = 1


    """ Calculamos la probabilidad de clase 1 sabiendo que
        P(Clase 1) = sigmoidal(valor) donde
        sigmoidal(valor) = 1/(1 + e^(-valor)) """
    def sigmoidal(self, val):
        try:
            aux = 1./(1+np.exp(-val))
        except OverflowError:
            if val > 0:
                aux = 1.0
            else:
                aux = 0.0
        return aux


    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        """En caso de que no se nos pase el vector por parámetros,
        creamos un vector con valores aleatorios entre [-0.5, 0.5)"""

        if self.w is None:
            self.w = np.random.uniform(-0.5, 0.5, len(diccionario))

        for n in range(self.nEpocas):
            for dato in datostrain:
                #Vector x = (1, x1, x2, x3...)
                x = np.append([1], dato[:-1])
                #Producto escalar: <w, x>
                pEsc = np.dot(self.w, x)
                # MV: w = w-cteAprend*(S(<w,x>)-dato) * x
                sigm = self.cteAprend * (self.sigmoidal(pEsc) - dato[-1]) * x
                self.w = self.w - sigm
        pass


    def clasifica(self,datostest,atributosDiscretos,diccionario):
        prediccion = np.empty(len(datostest), dtype=int)

        for i, dato in enumerate(datostest):
            x = np.append([1], dato[:-1])
            pEsc = np.dot(self.w, x)

            if self.sigmoidal(pEsc) >= 0.5:
                #Si está por encima de la recta
                prediccion[i] = 1.0
            else:
                prediccion[i] = 0.0

        return prediccion

  
class ClasificacionGenetico(Clasificador):

    def __init__(self, pMutation, pCruce, tamanoPoblacion, maximoReglas=1, nEpocas=1, elitismo=0.05, poblacion_inicial = []):
        self.elitism = elitismo
        self.pCruce = pCruce
        self.pMutation = pMutation
        self.maximoReglas = maximoReglas
        self.tamanoPoblacion = tamanoPoblacion
        self.nEpocas = nEpocas


        self.atributos = {}
        self.listaValores = []
        self.poblacion_inicial = poblacion_inicial
        self.aciertos =[]
        self.poblacion =[]

    """En esta función se definirán el conjunto de reglas que genera cada 
    individuo de la población, además, se evolucionará la población"""
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
      #La población que va a ir evolucionando
      poblacion = []
      # Comprobamos si tenemos una poblacion externa
      if len(self.poblacion_inicial) == 0:
        #Para cada dato del dataset de entrenamiento se crear las reglas que define
        for _ in range(self.tamanoPoblacion):
          self.atrib_lens = []
          # Obtenemos la longitud de los atributos
          for atrib_idx in range(datostrain.shape[1] - 1):
            columna = datostrain[:, atrib_idx]
            self.atrib_lens.append(len(np.unique(columna)))
          # Creamos una serie de reglas acordes a la longitud de los atributos
          cromosoma = self.defReglas(np.random.randint(self.maximoReglas + 1))
          poblacion.append(cromosoma)
      else:
        poblacion = self.poblacion_inicial
      self.poblacion = poblacion
      #Se define como es el modelo de atributos en este dataset, es decir, 
      #cantidad de bits de los cromosomas y sus atributos
      self.defAtributos(datostrain, diccionario)

      print("OLE:", poblacion[0])
      #Se evoluciona la población
      self.evoluciona(datostrain)


    """En esta función se definirá la estructura de los atributos en el dataset
    en caso de que no fuera inicialmente un dato numérico se usará el diciconario
    en caso contrario, se usará el valor máximo de la población. Se harán los modelos
    de los genes usando exclusivamente datostrain, pese a que esto puede causar que 
    haya un dato en datostest de valor mayor al posible en el modelo."""
    def defAtributos(self, datostrain, diccionario):
      i = 0
      # datos = datostrain[:-1]
      for atributo in diccionario:
        #Si el atributo era numérico en el dataset original
        if not atributo:
          #Se obtiene la lista de todos los atributos
          listaValores = datostrain[:,i]
          #Se dejan los únicos
          listaValores = np.unique(listaValores)
          print(listaValores)
          listaValores.sort()
          #Guardamos esta lista porque la necesitaremos para obtener la regla 
          #de estos atributos
          self.listaValores.append(listaValores)
          #Obtenemos la cantidad de valores únicos
          bits = len(listaValores)

        #Si el atributo no era numérico y necesitó traducción
        else:
          #Definimos la lista para que la comprobación en reglas funcione
          self.listaValores.append([])
          #La cantidad de bits será la clave mayor +1 porque empieza en 0
          bits = int(list(atributo.values())[-1]) + 1
          print('bits', bits)
          print('listaValores', len(listaValores))
          # bits = len(listaValores)
          # print(bits)

        self.atributos[i] = bits
        i += 1


    """En esta función se definirá el conjunto de reglas de cada individuo de 
    datos train siguiendo el formato de la memoria"""
    def defReglas(self, maximoReglas):
      individuo = []
      for reglas in range(maximoReglas):
        reglas = []
        i = 0
        #Para cada dato se define la regla, ya que según Pittsburg,
        #cada individuo define un conjunto de reglas
        for atrib_len in self.atrib_lens:
          regla = list(np.random.randint(2, size=atrib_len))
          # Nos aseguramos de que no haya una regla no activable
          regla[int(np.random.randint(atrib_len))] = 1          
          #Añadimos la regla al conjunto de reglas del individuo
          reglas.append(regla)
          i += 1
        reglas += [[np.random.randint(2)]]
        individuo.append(reglas)

      return individuo


    """En esta función se evoluciona la población hasta obtener la población final
    se ha decidido hacer el bucle por las épocas dentro de esta función para evitar el 
    coste añadido de hacer tantas llamadas a la función como épocas. Se puede tener 
    cruce inter-regla, intra-regla"""
    def evoluciona(self, datos):
      cantidad_elite = int(len(self.poblacion) * self.elitism)
      
      for epoca in range(self.nEpocas):
        poblacion_nueva = []
        fit = []
        
        #Aplicamos la función de fitness a todos los individuos de la antigua
        for i, individuo in enumerate(self.poblacion):
          resultado = self.fitness(individuo, datos)
          fit.append([individuo, resultado])
          print(individuo, resultado)
        fit = (sorted(fit, key = lambda x: x[1], reverse=True))
        
        # Introducimos a la elite en la nueva poblacion
        for f in fit[:cantidad_elite]:
          poblacion_nueva.append(f[0])

        # Ruleta para escoger otros elementos que no sean la elite
        aux_pob = [i for i, _ in fit]
        aux_fit = [i for _, i in fit]
        num_cruzados = int(self.pCruce * self.tamanoPoblacion)
        for i in range(num_cruzados):
          seed = np.random.random() * np.sum(aux_fit)
          i = 0
          while seed > 0:
            seed -= aux_fit[i]
            i += 1
          poblacion_nueva.append(aux_pob[i-1])
          aux_pob.pop(i-1)
          aux_fit.pop(i-1)
        
        print("Longitud poblacion nueva: ", len(poblacion_nueva))

        # Cruce
        while len(poblacion_nueva) < (self.tamanoPoblacion):
          i, j = np.random.randint(0, len(self.poblacion), size=2)
          #Aquí poner un if para comprobar que tipo de cruce es
          hijo_1, hijo_2 = self.cruce(self.poblacion[i], self.poblacion[j])
          poblacion_nueva.append(hijo_1)
          poblacion_nueva.append(hijo_2)

        # MUTASIONES AQUI
        
        # En caso de que sea impar el número de individuos, el último individuo no se cruza
        # con otro  y por tanto se pasa directamente a la siguiente generación
        if len(poblacion_nueva) == self.tamanoPoblacion+1:
          poblacion_nueva.pop()

        self.poblacion = poblacion_nueva
        
        for i, individuo in enumerate(self.poblacion):
          fit.append([individuo, self.fitness(individuo, datos)])
        fit = (sorted(fit, key = lambda x: x[1], reverse=True))

        print("Tam pobl:", len(self.poblacion) )
        print("Mejor individuo: ", fit[0][0])
        print("Best fit de esta generacion: ", fit[0][1])
      return

    """Se hará con un solo punto de cruce para cada caso. En inter-regla un punto de cruce,
    en intra regla, un punto de cruce aleatorio en cada regla."""
    def cruce(self, padre1, padre2, tipo_cruce=0):
      aux_1 = copy.deepcopy(padre1)    
      aux_2 = copy.deepcopy(padre2)    
      
      hijo1 = []
      hijo2 = []
      #Cruce inter-regla
      
      if tipo_cruce == 0:
        #Desde después del primer atributo, hasta después del penúltimo
        punto_cruce = random.randint(1, len(aux_1) - 1)
        hijo1 = aux_1[:punto_cruce]
        hijo1 += (aux_2[punto_cruce:])
        hijo2 = aux_2[:punto_cruce]
        hijo2 += (aux_1[punto_cruce:])
      else:
        i = 0
        hijo1 = []
        hijo2 = []
        for regla in aux_1:
          punto_cruce = random.randint(1,len(regla) - 1)
          
          regla1 = regla[:punto_cruce]
          regla1 += (aux_2[i][punto_cruce:])
          hijo1.append(regla1)

          regla2 = aux_2[i][:punto_cruce]
          regla2 += (regla[punto_cruce:])
          hijo2.append(regla2)
          i+=1
          
      return hijo1, hijo2

    """Esta función muta la cadena que se pasa como argumento. Para esto se altera uno de los bits.
    Se mutará cambiando a 1 valores que sean 0 y viceversa. 
    Se realiza la mutación tan solo una vez por individuo, 
    los individuos sobre los que se realiza la mutación."""
    def mutacion(self, individuo):     
      atributo_mutado = random.randint(0,len(individuo) - 1)
      bit_mutado = random.randint(0,len(individuo[atributo_mutado]) - 1)
      if individuo[atributo_mutado][bit_mutado] == 0:
        individuo[atributo_mutado][bit_mutado] = 1
      else:
        individuo[atributo_mutado][bit_mutado] = 0
        
      return

    """Función de fitness del clasificador, en este caso se nos ha pedido que sea la probabilidad
    de acierto del conjunto de reglas con los datos de entrenamiento"""
    def fitness(self, conjunto_reglas, datos, training=True):
      """ Queremos evaluar el conjunto de reglas sobre los datos de entrenamiento
      (o sobre clasificacion). Para ello:
        -Iteramos sobre cada uno de los datos que tenemos
        -Para cada dato, vemos que decreta nuestro conjunto de reglas
        -Comprobamos si lo que decreta coincide y está acorde con la clase real
         (esto será solo si estamos en training = True)
        -Si coincide, sumamos un acierto, si no, un error
        -El fitness devuelto será el número de aciertos entre la longitud de datos
      """
      # Codificamos datos de entrada con OneHotEncoder
      encodedDataset = preprocessing.OneHotEncoder(sparse=False)
      encodedDataset.fit(datos[:-1])
      data = encodedDataset.transform(datos[:-1])

      aciertos = 0
      #Tenemos en cuenta los datos de entrenamiento
      for fila, dato in enumerate(data):
        print(dato)
        print(len(dato))
        # Vemos que decreta nuestro conjunto de reglas
        eval_regla = []
        # Evaluamos la regla
       
        for regla in conjunto_reglas:
          or_atributos = []
          for atrib_idx, atributo in enumerate(regla):
            flag_atributos = []
            for bit in atributo:
              flag_atributos.append(all( [bit, dato[atrib_idx]] )) # 1 - 1 True
          
            # Comprobamos si alguno de ellos es True (or)
            or_atributos.append(any(flag_atributos)) # True si al menos uno es True
          
          if all(or_atributos): # Si se activa la regla
            eval_regla.append(regla[-1][0]]) # Guardo la clase 

        if eval_regla:
          pred = mode(eval_regla)[0][0]
          if pred == datos[fila, -1]:
            aciertos += 1
        return

      return aciertos / len(datos)

          

    def clasifica(self,datostest,atributosDiscretos,diccionario):
      prediccion = []
      poblacionclf = []

      for i, dato in enumerate(datostest):
        conjuntoReglas = self.defReglas(dato)
        poblacionclf.append(conjuntoReglas)
      
      aux = self.fitness(self.poblacion[0], poblacionclf, 1)

      print(aux)
      return prediccion
