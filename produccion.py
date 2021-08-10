import streamlit as st
from PIL import Image
 

def menu_evaluador():
	proceso = st.sidebar.radio('Select the process',['🏠 Overview','⏲️ Performance','📈 Analytics','🚀 Landing','📋 Projects'])
	return proceso

def produccion():
	st.sidebar.warning("Bienvenido al módulo petrofísico elija el metodo de evaluacion") #Amarillo

	proceso = menu_evaluador()

	if proceso == '🏠 Overview':
		st.write('Módulo de produccion')
		overview = Image.open('1 reservas.jpeg')
		st.image(overview)

	if proceso == '⏲️ Performance':
		st.write('Módulo de produccion')
		Performance = Image.open('2 Arima_model.jpeg')
		st.image(Performance)

	if proceso == '📈 Analytics':
		st.write('Módulo de produccion')
		Analytics = Image.open('3 Oil_Production.jpeg')
		st.image(Analytics)
