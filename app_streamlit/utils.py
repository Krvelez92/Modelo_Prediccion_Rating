##########################################
#####          LIBRERIAS             #####
##########################################

import numpy as np
import pandas as pd
from unidecode import unidecode 

##########################################
#####          FUNCIONES             #####
##########################################

''' 
Resumen:
- json_to_dataframe(data_entrada)
- eliminar_acentos(texto:str)
'''
#----------------------------------------------------------------------------------------------------------------------------------
def json_to_dataframe(data_entrada):
    ''' 
    Funcion que transforma json a un DataFrame específico.

    Inputs:
        data_entrada:json
    
    Outputs:
        df: DataFrame
    '''
    places = {'nombre': [], 'id': [], 'lat':[], 'lon':[]}

    for key, val in data_entrada.items():
        nombre = data_entrada[key]['name']
        lat = data_entrada[key]['geometry']['location']['lat']
        lon = data_entrada[key]['geometry']['location']['lng']
        id = key
        
        places['nombre'].append(nombre)
        places['id'].append(id)
        places['lat'].append(lat)
        places['lon'].append(lon)

    df = pd.DataFrame(places)
    return df

#----------------------------------------------------------------------------------------------------------------------------------
def eliminar_acentos(texto:str):
    ''' 
    Funcion para quitar los acentos y hacer minusculas.

    Input:
        texto:str
    
    Output:
        texto:str
    '''
    if isinstance(texto, str):
        return unidecode(texto.lower())
    return texto 
