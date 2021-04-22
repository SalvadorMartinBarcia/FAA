import pandas as pd 
import numpy as np
from pandas.api.types import is_string_dtype, is_numeric_dtype


class Datos:     
    
    
    
    def __init__(self, nombreFichero):         # Leer archivo         
        self.nominalAtributos = []     
        self.datos = pd.DataFrame()
        self.diccionario = {}
        self.attributesTypes(nombreFichero)
        self.createDictionary(nombreFichero)
        self.useDictionary(nombreFichero)
    

    def attributesTypes(self, nombreFichero):
        datosAux = pd.read_csv(nombreFichero, dtype={'Class':'object'})  
        for tipo in datosAux.dtypes:
            if is_string_dtype(tipo):
                self.nominalAtributos.append(True)
            elif is_numeric_dtype(tipo):
                self.nominalAtributos.append(False)
            else:
                raise Exception('ValueError: Tipo de dato no pertenece a int, float u object')

    def createDictionary(self, nombreFichero):
        datosAux = pd.read_csv(nombreFichero, dtype={'Class':'object'})
        headers = list(datosAux.columns) 
        
        for index, header in enumerate(headers):
            self.diccionario[header] = {}
            if self.nominalAtributos[index] is False:
                continue
            sortedArray = np.sort(datosAux[header].unique())
            for index2, item in enumerate(sortedArray):
                self.diccionario[header][item] = index2

    def useDictionary(self, nombreFichero):
        datosAux = pd.read_csv(nombreFichero, dtype={'Class':'object'})  
        headers = list(datosAux.columns)
        self.datos = pd.DataFrame(columns=datosAux.columns, index=datosAux.index)

        for index, header in enumerate(headers):
            if self.nominalAtributos[index] is False:
                self.datos[header] = datosAux[header]
            else:
                for index2, ele in enumerate(datosAux[header]):
                    self.datos.at[index2, header] = self.diccionario[header][ele]


        
        
    def extraeDatos(self, idx):         
        pass