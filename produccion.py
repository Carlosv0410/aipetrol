import streamlit as st
from PIL import Image
 

def menu_evaluador():
	proceso = st.sidebar.radio('Select the process',['馃彔 Overview','鈴诧笍 Performance','馃搱 Analytics','馃殌 Landing','馃搵 Projects'])
	return proceso

def produccion():
	st.sidebar.warning("Bienvenido al m贸dulo de producci贸n elija el metodo de evaluacion") #Amarillo

	proceso = menu_evaluador()

	if proceso == '馃彔 Overview':
		st.write('M贸dulo de produccion')
		overview = Image.open('1 reservas.jpeg')
		st.image(overview)

	if proceso == '鈴诧笍 Performance':
		st.write('M贸dulo de produccion')
		Performance = Image.open('2 Arima_model.jpeg')
		st.image(Performance)

	if proceso == '馃搱 Analytics':
		st.write('M贸dulo de produccion')
		Analytics = Image.open('3 Oil_Production.jpeg')
		st.image(Analytics)
