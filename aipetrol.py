# Librerias Externas
import gzip
import io
import streamlit as st
import pandas as pd
import lasio
import matplotlib.pyplot as plt
import altair as alt
from PIL import Image
from io import StringIO
# Librerias Internas
import sismica
import petrofisica
import produccion
# Portada - Logo
portada = Image.open('portada.png')
st.image(portada)
# Sidebar-Menu
st.sidebar.title('Menú Prinicipal')
# Modulos a Elegir
modulo = st.sidebar.selectbox("Escoja un Módulo",['Select an option', 'Producción', 'Petrofísica','Sísmica'], format_func=lambda x: 'Select an option' if x == '' else x)
if modulo == 'Sísmica':
	sismica.sismica()
elif modulo == 'Petrofísica':
	petrofisica.petrofisica()
elif modulo == 'Producción':
	produccion.produccion()
else:
	st.warning("Bienvenido a AI-Petrol") #Amarillo


