##########################################
#####          LIBRERIAS             #####
##########################################

import streamlit as st
import geopandas as gpd
from data_processing import enc_cocina, enc_distrito, enc_barrio, encoder
import pandas as pd

##########################################
#####          FUNCIONES             #####
##########################################

''' 
Resumen:
- preferencias_culinarias_params(config, readme)
- servicios_params(config, readme)
- atcliente_params(config, readme)
- transformar_datos_user(df_user, barrios_mapa, barrios_m30)
- dicretizar_prediccion(pred)
'''
#----------------------------------------------------------------------------------------------------------------------------------

def preferencias_culinarias_params(config, readme):
    ''' 
    Función para prefencias culinarias de Streamly.

    Input:
        config:str
        readme:str
    
    Output:
        seleccion_comida:str
        comida_vegetariana:bool
        vino:bool 
        cerveza:bool
    '''
    comida = ['Española', 'Latinoamericana', 'Italiana', 'China', 'Otros',
       'Japonesa', 'Americana / Burgers', 'Fusión', 'Mexicana',
       'Asiática']
    
    seleccion_comida = st.selectbox("Selecciona un tipo:", comida)
    comida_vegetariana = st.radio("¿Tienes menú vegetariano?", ["Sí", "No"])
    vino = st.radio("¿Tienes un carta de vinos?", ["Sí", "No"])
    cerveza = st.radio("¿Sirves Cerveza?", ["Sí", "No"])

    return seleccion_comida, comida_vegetariana, vino, cerveza

#----------------------------------------------------------------------------------------------------------------------------------

def servicios_params(config, readme):
    ''' 
    Función para servicios del restaurante de Streamly.

    Input:
        config:str
        readme:str
    
    Output:
        comer_dentro:bool
        acepta_reservas:bool
        takeout:bool 
        delivery:bool
        weelchair:bool
    '''
    comer_dentro = st.radio("¿Se puede comer en el local?", ["Sí", "No"])
    acepta_reservas = st.radio("¿Se puede hacer reservas?", ["Sí", "No"])
    takeout = st.radio("¿Pueden hacer pedidos para llevar?", ["Sí", "No"])
    delivery = st.radio("¿Vas a habilitar servicios de delivery?", ["Sí", "No"])
    weelchair = st.radio("¿Tu local será accesible para personas en silla de ruedas?", ["Sí", "No"])

    return comer_dentro, acepta_reservas, takeout, delivery, weelchair

#----------------------------------------------------------------------------------------------------------------------------------

def atcliente_params(config, readme):
    ''' 
    Función para atención del cliente en Streamlit.

    Input:
        config:str
        readme:str
    
    Output:
        serves_breakfast:bool
        serves_brunch:bool
        serves_dinner:bool 
        serves_lunch:bool
        open_weekends:bool
    '''
    serves_breakfast = st.radio("¿Servirán desayunos?", ["Sí", "No"])
    serves_brunch = st.radio("¿Servirán brunch?", ["Sí", "No"])
    serves_lunch = st.radio("¿Servirán comida?", ["Sí", "No"])
    serves_dinner = st.radio("¿Servirán cena?", ["Sí", "No"])
    open_weekends = st.radio("¿Abrirás los fines de semana?", ["Sí", "No"])
    
    return serves_breakfast, serves_brunch, serves_lunch, serves_dinner, open_weekends

#----------------------------------------------------------------------------------------------------------------------------------

def transformar_datos_user(df_user, barrios_mapa, barrios_m30):
    ''' 
    Funcion para formatear los datos del usuario para el modelo.

    Input:
        df_user:DataFrame
        barrios:GeoDataFrame
        barrios_m30:list
    
    Output:
        barrios:DataFrame
    '''
    barrios = barrios_mapa[barrios_mapa['COD_BAR'].isin(barrios_m30)] #filtrar lo barrio de la m30

    # Proceso de Geopadas
    barrios_proj = barrios.to_crs(epsg=25830)
    barrios_proj['centroid'] = barrios_proj.geometry.centroid # calculamos centroides de barrios
    centroides_latlon = barrios_proj.set_geometry('centroid').to_crs(epsg=4326)
    barrios['lon'] = centroides_latlon.geometry.x
    barrios['lat'] = centroides_latlon.geometry.y
    barrios['centroid'] = centroides_latlon.geometry
    barrios = barrios[['CODDIS', 'NOMDIS', 'COD_BAR', 'NOMBRE','lon', 'lat', 'centroid']]

    # Para cada cada centroide calculamos un buffer de 500 para mirar que restaurantes tenemos cerca
    barrios_geo = gpd.GeoDataFrame(barrios, geometry='centroid', crs='EPSG:4326')
    barrios_geo = barrios_geo.to_crs(epsg=25830)
    barrios_geo['buffer_500'] = barrios_geo.centroid.buffer(500) #creamos el campo de buffer
    barrios_geo = barrios_geo.set_geometry('buffer_500')
    # leemos los restaurantes de madrid
    restaurantes = pd.read_csv('../data/processed/restaurantes.csv')
    restaurantes = restaurantes[['lat', 'lon', 'place_id', 'price_level', 'rating', 
                                'user_ratings_total', 'anio_medio_constr_vivendas',
                                'dur_media_credito_viviendas', 'edad_media_poblacion',
                                'num_locales_alta_abiertos', 'num_locales_alta_cerrados',
                                'poblacion_densidad','renta_media_persona',
                                'pct_crecimiento_demografico','valor_catast_inmueble_residen',
                                'cod_barrio']]
    
    # creamos un df de los kpi por barrio
    kpi = restaurantes.groupby('cod_barrio')[['anio_medio_constr_vivendas', 'dur_media_credito_viviendas',
                                'edad_media_poblacion', 'num_locales_alta_abiertos',
                                'num_locales_alta_cerrados', 'poblacion_densidad',
                                'renta_media_persona','pct_crecimiento_demografico',
                                'valor_catast_inmueble_residen']].max().reset_index()
    
    restaurantes_geo = gpd.GeoDataFrame(restaurantes, geometry=gpd.points_from_xy(restaurantes['lon'], restaurantes['lat']), crs='EPSG:4326')
    restaurantes_geo = restaurantes_geo.to_crs(epsg=25830)
    result_restaurantes = gpd.sjoin(restaurantes_geo, barrios_geo, how='right', predicate='intersects')
    result = result_restaurantes.groupby(['COD_BAR'])[['price_level', 'rating', 'user_ratings_total']].mean().reset_index()
    result2 = result_restaurantes.groupby(['COD_BAR'])[['place_id']].count().reset_index()
    result = pd.merge(result, result2, left_on=['COD_BAR'], right_on=['COD_BAR'])
    result.rename({
                'price_level':'price_level_mean',
                'rating':'rating_mean',
                'user_ratings_total':'user_ratings_mean',
                'place_id':'num_restaurantes'}, axis=1, inplace=True)
    
    # Resultado de buffer
    barrios = pd.merge(barrios, result, how='left', left_on='COD_BAR', right_on='COD_BAR')
    restaurantes.fillna(0, inplace=True)

    # Convertimos a int
    barrios['COD_BAR'] = barrios['COD_BAR'].astype('int')
    barrios['CODDIS'] = barrios['CODDIS'].astype('int')

    # Incluir los kpi a barrios
    barrios = pd.merge(barrios, kpi, how='left', left_on='COD_BAR', right_on='cod_barrio')

    # Incluimos datos del usuario
    for col in df_user.columns:
        barrios[col] = df_user.iloc[0][col]

    # One hot enconder
    tip_coci= enc_cocina.transform(barrios[['tipo_cocina']]).toarray()
    tip_cocina_dummy = pd.DataFrame(tip_coci, columns=[cat for cat in enc_cocina.categories_[0]])

    tip_distrito= enc_distrito.transform(barrios[['NOMDIS']]).toarray()
    tip_distrito_dummy = pd.DataFrame(tip_distrito, columns=[cat for cat in enc_distrito.categories_[0]])

    tip_barrio= enc_barrio.transform(barrios[['NOMBRE']]).toarray()
    tip_barrio_dummy = pd.DataFrame(tip_barrio, columns=[cat for cat in enc_barrio.categories_[0]])

    barrios = pd.concat([barrios, tip_cocina_dummy, tip_distrito_dummy, tip_barrio_dummy], axis=1)

    # Label enconder
    tipo_cocina_encoder = encoder.transform(barrios['tipo_cocina'])
    barrios['tipo_cocina_encoder'] = tipo_cocina_encoder

    # Quitamos Columnas
    barrios.drop(['NOMDIS', 'NOMBRE', 'centroid', 'cod_barrio', 'tipo_cocina'], axis=1, inplace=True)

    barrios.rename({'CODDIS':'cod_distrito',
                    'COD_BAR':'cod_barrio'}, inplace=True, axis=1)
    
    # Ordenamos columnas para el modelo
    barrios = barrios[['lat', 'lon', 'dine_in', 'price_level', 'reservable', 'serves_beer', 
                       'serves_breakfast', 'serves_brunch', 'serves_dinner','serves_lunch', 
                       'serves_vegetarian_food', 'serves_wine', 'takeout', 'delivery',
                       'weelchair', 'hours_open', 'num_days_open', 'open_weekends', 'cod_distrito',
                       'cod_barrio', 'price_level_mean', 'rating_mean', 'user_ratings_mean',
                       'num_restaurantes', 'anio_medio_constr_vivendas', 'dur_media_credito_viviendas', 
                       'edad_media_poblacion', 'num_locales_alta_abiertos', 'num_locales_alta_cerrados', 
                       'poblacion_densidad', 'renta_media_persona', 'pct_crecimiento_demografico',
                       'valor_catast_inmueble_residen', 'Americana / Burgers', 'Asiática',  'China', 
                       'Española', 'Fusión', 'Italiana', 'Japonesa', 'Latinoamericana', 'Mexicana',
                       'Otros', 'Arganzuela', 'Carabanchel', 'Centro', 'Chamartín', 'Chamberí',
                       'Ciudad Lineal', 'Fuencarral - El Pardo', 'Hortaleza', 'Latina', 
                       'Moncloa - Aravaca', 'Moratalaz', 'Puente de Vallecas', 'Retiro', 'Salamanca',
                       'Tetuán', 'Usera', 'Abrantes', 'Acacias', 'Adelfas', 'Almagro', 'Almenara', 
                       'Almendrales', 'Aluche', 'Apóstol Santiago', 'Arapiles', 'Argüelles', 'Atalaya',
                       'Atocha', 'Bellas Vistas', 'Berruguete', 'Canillas', 'Casa de Campo',
                       'Castellana', 'Castilla', 'Castillejos', 'Chopera', 'Ciudad Jardín', 
                       'Ciudad Universitaria', 'Colina', 'Comillas', 'Cortes', 'Costillares', 
                       'Cuatro Caminos', 'Delicias', 'El Pardo', 'El Viso', 'Embajadores', 
                       'Entrevías', 'Estrella', 'Fontarrón', 'Fuente del Berro', 'Fuentelarreina',
                       'Gaztambide', 'Goya', 'Guindalera', 'Hispanoamérica', 'Ibiza', 'Imperial',
                       'Justicia', 'La Concepción', 'La Paz', 'Legazpi', 'Lista', 'Los Cármenes',
                       'Los Jerónimos', 'Lucero', 'Marroquina', 'Media Legua', 'Mirasierra', 
                       'Moscardó', 'Niño Jesús', 'Nueva España', 'Numancia', 'Opañel', 'Pacífico',
                       'Palacio', 'Palomeras Bajas', 'Palos de la Frontera', 'Peñagrande', 'Pilar', 
                       'Pinar del Rey', 'Piovera', 'Portazgo', 'Pradolongo', 'Prosperidad', 
                       'Puerta Bonita', 'Puerta del Ángel', 'Quintana', 'Recoletos', 'Ríos Rosas', 
                       'San Diego', 'San Isidro', 'San Juan Bautista',     'San Pascual', 'Sol',
                       'Trafalgar', 'Universidad', 'Valdeacederas', 'Valdefuentes', 'Valdezarza', 
                       'Vallehermoso', 'Valverde', 'Ventas', 'Vista Alegre', 'Zofío', 'tipo_cocina_encoder'
                       ]]
    return barrios

#----------------------------------------------------------------------------------------------------------------------------------

def dicretizar_prediccion(pred):
    '''
    Funcion que discretiza la prediccion en categorias.

    Input:
        pred:float
    
    Output:
        rango_pred:str
    '''
    if pd.isna(pred):
        rango_pred = 'Sin predicción'
    if pred >= 40:
        rango_pred = 'Extremadamente popular'
    elif pred >= 30.8:
        rango_pred = 'Muy popular' 
    elif pred >= 26.5:
        rango_pred = 'Popular' 
    elif pred >= 20.85:
        rango_pred = 'Poco popular' 
    else:
        rango_pred = 'Muy poco popular'
    return rango_pred