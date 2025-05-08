##########################################
#####          FUNCIONES             #####
##########################################


import streamlit as st
import geopandas as gpd
from streamlit_folium import st_folium
import folium
from PIL import Image
import util_streamlit as u
import pandas as pd
import pickle
import random


# ----------------- Configuración Pagina Tab ---------------------

apptitle = 'MadFood'

st.set_page_config(page_title=apptitle, page_icon=":ramen:")


#---------------------------------------------------------------#
#-------------------        Side Bar     -----------------------#
#---------------------------------------------------------------#

# Header Pagina Principal ---------------------------------------------------------------------
st.sidebar.image(Image.open('../doc/imagenes/Streamlit_logo.png'), use_container_width=True)

st.sidebar.markdown("""Para calcular el exito de tu futuro restaurante rellena este formulario,
            tranquila/o no te voy a robar la idea. 😆""")

# Calculo Precio
precio_opciones = [
    "1 - $ (Económico)", 
    "2 - $$ (Moderado)", 
    "3 - $$$ (Exclusivo)", 
    "4 - $$$$ (Premium)"
]

precio_seleccionado = st.sidebar.radio(
    "Selecciona el rango de precio de tu restaurante:",
    options=precio_opciones
)

st.sidebar.write("Precio seleccionado:", precio_seleccionado)

# Configuración de Opciones de Restaurante  ---------------------------------------------------
st.sidebar.title("1. Preferencias Culinarias")

with st.sidebar.expander("Tipo de Cocina", expanded=False):
    params_cocina = u.preferencias_culinarias_params({}, "")

seleccion_comida, comida_vegetariana, vino, cerveza = params_cocina

#---------------------------------------------------------------
st.sidebar.title("2. Servicios")

with st.sidebar.expander("Tipo de Servicios", expanded=False):
    params_servicios = u.servicios_params({}, "")

comer_dentro, acepta_reservas, takeout, delivery, weelchair = params_servicios

#---------------------------------------------------------------
st.sidebar.title("3. Atención al Cliente")

with st.sidebar.expander("Información de horarios", expanded=False):
    params_atcliente = u.atcliente_params({}, "")

serves_breakfast, serves_brunch, serves_lunch, serves_dinner, open_weekends = params_atcliente
#---------------------------------------------------------------

with st.sidebar.expander("Distribución de horarios", expanded=False):
    horas_abierto = st.slider("¿Cuántas horas a la semana estará abierto?", 1, 150, 80)
    dias_abierto = st.slider("¿Cuántos días a la semana estará abierto?", 1, 7, 5)

#------------------------- Data Frame Resultados Usuario ------------------------------------
data_usuario = {
    "price_level": [float(precio_seleccionado[0])],
    "tipo_cocina":[seleccion_comida],
    "serves_vegetarian_food":[int(bool(comida_vegetariana))],
    "serves_wine":[int(bool(vino))],
    "serves_beer":[int(bool(cerveza))],
    "dine_in":[int(bool(comer_dentro))],
    "reservable":[int(bool(acepta_reservas))],
    "takeout":[int(bool(takeout))],
    "delivery":[int(bool(delivery))],
    "weelchair":[int(bool(weelchair))],
    "serves_breakfast": [int(bool(serves_breakfast))],
    "serves_brunch": [int(bool(serves_brunch))],
    "serves_lunch": [int(bool(serves_lunch))],
    "serves_dinner": [int(bool(serves_dinner))],
    "open_weekends": [int(bool(open_weekends))],
    "hours_open": [float(horas_abierto)],
    "num_days_open": [int(dias_abierto)]
}

df_usuario = pd.DataFrame(data_usuario)


#---------------------------------------------------------------#
#-------------------   Pagina Principal  -----------------------#
#---------------------------------------------------------------#

st.markdown("""
    <style>
    .header-banner {
        background-image: url("https://raw.githubusercontent.com/Krvelez92/Modelo_Prediccion_Rating/main/doc/imagenes/header.png");
        background-repeat: repeat-x;
        background-position: top center;
        background-size: auto 150px; 
        height: 150px;
    }
    </style>

    <div class="header-banner"></div>
""", unsafe_allow_html=True)

# Title the app
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@800&display=swap');

    .titulo-premium {
        font-size: 88px;
        font-weight: 800;
        text-align: center;
        margin-top: 30px;
        margin-bottom: 20px;
        font-family: 'Poppins', sans-serif;
        color: #ee682e !important; 
        text-shadow: 3px 3px 6px rgba(0,0,0,0.1);
    }

    .subtitulo {
        text-align: center;
        font-size: 24px;
        color: #555;
        margin-bottom: 40px;
        font-family: 'Poppins', sans-serif;
    }
    </style>

    <div class="titulo-premium">Madfood</div>
    <div class="subtitulo">Descubre la mejor ubicación para tu restaurante en el corazón de Madrid</div>
""", unsafe_allow_html=True)

st.markdown("""
¡Bienvenido a **Madfood**!

Nos emociona acompañarte en el emocionante camino de abrir tu nuevo restaurante.

Nuestro objetivo es ayudarte a encontrar la ubicación ideal para tu negocio dentro de la M30, el corazón gastronómico de Madrid.  
Para hacerlo de la mejor manera, necesitamos que nos cuentes un poco más sobre tu concepto e ideas.

Con esa información, te recomendaremos el barrio que mejor se alinee con el estilo y las características de tu restaurante, maximizando tus oportunidades de éxito.

¡Empecemos a diseñar tu futuro juntos!
""")


barrios_mapa = gpd.read_file('../data/raw/Barrios.json')
barrios_mapa = gpd.GeoDataFrame(barrios_mapa, geometry='geometry', crs='EPSG:4326') 

#---------------------------------------------------------------
# Modelo

codigos_barrios = [
    "011", "012", "013", "014", "015", "016",
    "021", "022", "023", "024", "025", "026", 
    "031", "032", "033", "034", "035", "036",
    "041", "042", "043", "044", "045", "046",
    "051", "052", "053", "074", "075","076",
    "061", "062", "063", "064", "065", "066",
    "071", "072", "073", "092", "054", "055",
    "081", "091", "121", "056", "085", "027",
    "131", "132", "133", "151", "093", "094",
    "084"
]

barrios = u.transformar_datos_user(df_usuario, barrios_mapa, codigos_barrios)


modelo_importado = pickle.load(open("../models/1_randomforest_model.pkl", 'rb'))

#---------------------------------- Mapa --------------------------------------
if "prediccion" not in st.session_state:
    st.session_state.prediccion = None

if st.sidebar.button("Predecir"):
    st.session_state.prediccion = modelo_importado.predict(barrios)
    st.success("✅ Predicción realizada")

# ---------------------------------------
# Mostrar mapa según predicción

if st.session_state.prediccion is not None:

    barrios["rating_predicho"] = st.session_state.prediccion
    barrios["rating_discreto"] = barrios["rating_predicho"].apply(u.dicretizar_prediccion)

    barrios_mapa['COD_BAR'] = barrios_mapa['COD_BAR'].astype(int)
    barrios_pred = pd.merge(barrios_mapa, barrios[['cod_barrio', 'rating_discreto', 'rating_predicho']],
                            left_on='COD_BAR', right_on='cod_barrio', how='left')
    

    m = folium.Map(location=[40.4165, -3.70256], zoom_start=12, tiles="Cartodb Positron")

    folium.Choropleth(
        geo_data=barrios_pred,
        data=barrios_pred,
        columns=['COD_BAR', 'rating_predicho'],
        key_on='feature.properties.COD_BAR',
        fill_color='BuPu',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Rating Predicho por Barrio'
    ).add_to(m)

    for _, r in barrios_pred.iterrows():
        folium.GeoJson(
            r['geometry'],
            tooltip=f"{r['NOMBRE']} - Rating predicho: {r['rating_discreto']}"
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st.write("🗺️ **Mapa de Rating Predicho por Barrio**")
    st_folium(m, width=700)

    st.write("📊 **Top 5 Barrios con mayor rating predicho**")

    barrios_pred_copy = barrios_pred.copy()
    barrios_pred_copy.rename(columns={'NOMBRE':'Barrio',
                              'rating_discreto':'Rating',
                              'NOMDIS': 'Distrito'}, inplace=True)
    top = barrios_pred_copy[['Distrito', 'Barrio', 'Rating', 'rating_predicho']].sort_values(
        by='rating_predicho', ascending=False).reset_index(drop=True)
    
    top_5 = top[['Distrito', 'Barrio', 'Rating']].head(5)

    st.markdown("""
    Ahora que tenemos el top 5 de barrios puedes empezar a buscar el mejor local para tu restaurante:

    * [Alquilar Local](https://www.idealista.com/alquiler-locales/madrid-madrid/con-publicado_ultima-semana,restauracion/)
    * [Comprar Local](https://www.idealista.com/venta-locales/madrid-madrid/con-publicado_ultima-semana,restauracion/)
    """)

    st.dataframe(top_5, hide_index=True)

    kpi_barrios = pd.read_csv('../data/raw/kpi_barrios_madrid.csv')
    
    barrio_elegido = random.choice(top_5['Barrio'])
    
    kpi_barrios = kpi_barrios[kpi_barrios['barrio']==barrio_elegido]

    kpi_elegido = random.choice(kpi_barrios['indicador_completo'].tolist())

    kpi_barrios = kpi_barrios[kpi_barrios['indicador_completo']==kpi_elegido]

    valor_elegido = float(kpi_barrios['valor_indicador'].values[0])

    valor_formateado = f"{valor_elegido:,.0f}".replace(",", ".")

    st.write(f"Fun fact del barrio {barrio_elegido}: {kpi_elegido} es de {valor_formateado}.")

    
else:
    barrios_mapa = gpd.read_file('../data/raw/Barrios.json')
    barrios_mapa = gpd.GeoDataFrame(barrios_mapa, geometry='geometry', crs='EPSG:4326') 

    mapa_simple = folium.Map(location=[40.4165, -3.70256], tiles="OpenStreetMap", zoom_start=12.5)

    for _, row in barrios_mapa.iterrows():
        folium.GeoJson(row['geometry'],
                       tooltip=row['NOMBRE']).add_to(mapa_simple)

    st.write("🗺️ **Mapa de Barrios de Madrid (sin predicción todavía)**")
    st_folium(mapa_simple, width=700)


