import streamlit as st
 

def menu_evaluador():
	proceso = st.sidebar.radio('Select the process',['🏠 Overview','⏲️ Performance','📈 Analytics','🚀 Landing','📋 Projects'])
	return proceso

def produccion():
	menu_evaluador()
	st.write('produccion')