import streamlit as st
import segyio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from shutil import copyfile
from skimage import exposure

def menu_evaluador():
	proceso = st.sidebar.radio('Select the process',['🏠 Overview','⏲️ Performance','📈 Analytics','🚀 Landing','📋 Projects'])
	return proceso

def sismica_2d():

	with st.beta_expander('Introducción'):
		st.info("Utilice SEGY-IO para importar dos volúmenes sísmicos en formato de archivo SEGY desde el conjunto de datos F3, costa afuera de los Países Bajos, con licencia CC-BY-SA: un volumen de similitud y un volumen de amplitud (con suavizado de filtro mediano dirigido por inmersión aplicado") # Azul
	with st.beta_expander('Grafica del sector'):

		filename = 'data/basic/F3_Similarity_FEF_subvolume_IL230-430_XL475-675_T1600-1800.sgy'
		similarity = 1-segyio.tools.cube(filename)
		filename1 = 'data/basic/F3_Dip_steered_median_subvolume_IL230-430_XL475-675_T1600-1800.sgy'
		seismic = segyio.tools.cube(filename1)
		fig = plt.figure(figsize=(14,6))

		ax = fig.add_subplot(121)
		sim = ax.imshow(similarity[:,:,15], cmap='gray_r');
		fig.colorbar(sim, ax=ax)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.invert_xaxis()

		ax1 = fig.add_subplot(122)
		amp = ax1.imshow(seismic[:,:,15], cmap='gray_r');
		fig.colorbar(amp, ax=ax1)
		ax1.set_xticks([])
		ax1.set_yticks([])
		ax1.invert_xaxis()
		st.pyplot(fig)

		st.success("NÓTESE BIEN. Eso no parece correcto: el segmento de amplitud es más ancho que el segmento de similitud. Según los nombres de los archivos, se supone que tienen una forma idéntica. ¿Qué está pasando?")
		st.success('NÓTESE BIEN. Eso lo confirmó: el volumen de amplitud tiene 10 líneas adicionales y 55 líneas cruzadas adicionales')	
	
	with st.beta_expander('Recorte sísmico para visualización'):	
		np.shape(seismic) == np.shape(similarity)
		seismic = seismic[0:-10:1, 0:-55:1,:]
		assert (np.shape(seismic) == np.shape(similarity))
		fig2 = plt.figure(figsize=(14,6))
		ax = fig2.add_subplot(121)
		sim = ax.imshow(similarity[:,:,15], cmap='gray_r');
		fig2.colorbar(sim, ax=ax)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.invert_xaxis()

		ax1 = fig2.add_subplot(122)
		amp = ax1.imshow(seismic[:,:,15], cmap='gray_r');
		fig2.colorbar(amp, ax=ax1)
		ax1.set_xticks([])
		ax1.set_yticks([])
		ax1.invert_xaxis()

		st.pyplot(fig2)

	with st.beta_expander('Análisis de histograma - Definición de umbral'):
		st.info(""" Umbral para hacer una imagen de falla binaria
		Primero calculo el cuadrado de la similitud para estirar su rango. Esto ayudará con los siguientes umbrales basados en histograma
		""")

		col1, col2 = st.beta_columns(2)
		st.info('Eso muestra un valor de similitud (al cuadrado) de 0.05 es un buen umbral para hacer el volumen de falla binaria.')
		with col1:
			similarity = np.power(similarity,2)
			fig3 = plt.figure(figsize=(6, 6))
			ax = fig3.add_subplot(1, 1, 1)
			ax.set_xticks([])
			ax.set_yticks([])
			plt.imshow(similarity[:,:,15], cmap='gray_r',  vmin =0, vmax=0.4);
			plt.gca().invert_xaxis()
			plt.colorbar();
			st.pyplot(fig3)
		with col2:

			hi_sim = exposure.histogram(similarity)

			hist = plt.plot(hi_sim[1], hi_sim[0], 'brown')
			plt.ylim(0, 300000)
			plt.axvline(0.1, color='b', ls='--')
			plt.axvline(0.05, color='g', ls='--')
			plt.xlim(0,0.3)
			#st.set_option('deprecation.showPyplotGlobalUse', False)



			
			binary = np.zeros(similarity.shape, dtype=np.uint8)
			binary[similarity > 0.05] = 1

			fig4 = plt.figure(figsize=(6, 6))
			ax = fig4.add_subplot(1, 1, 1)
			ax.set_xticks([])
			ax.set_yticks([])
			plt.imshow(binary[:,:,15], cmap='gray_r');
			plt.gca().invert_xaxis()
			plt.colorbar();
			st.pyplot(fig4)
	with st.beta_expander('Visualización sísmica con máscara de falla superpuesta'):
		col3, col4 = st.beta_columns(2)
		st.success('Eso se ve bien. El siguiente bit es limpiar pequeños detalles en la imagen de falla binaria. Está tomado de mi tutorial 2D Diversión con fallas. En este caso, uso 10 píxeles después de un poco de prueba y error.')
		with col3:			
			label_objects, nb_labels = ndi.label(binary)
			sizes = np.bincount(label_objects.ravel())
			mask_sizes = sizes > 10
			mask_sizes[0] = 0
			cleaned = mask_sizes[label_objects]*1

			fig5 = plt.figure(figsize=(6, 6))

			ax = fig5.add_subplot(1, 1, 1)
			ax.set_xticks([])
			ax.set_yticks([])
			plt.imshow(cleaned[:,:,15], cmap='gray_r');
			plt.gca().invert_xaxis()
			plt.colorbar();

			st.pyplot(fig5)
		with col4:

			masked = np.zeros((np.shape(cleaned)))
			masked[cleaned == 0] = np.nan
			fig6 = plt.figure(figsize=(8, 6))

			ax = fig6.add_subplot(1, 1, 1)
			ax.set_xticks([])
			ax.set_yticks([])

			plt.imshow(seismic[:,:,15], cmap='gray_r')
			plt.colorbar()

			plt.imshow(masked[:,:,15], cmap='Reds')
			plt.gca().invert_xaxis()
			cb = plt.colorbar()
			cb.set_ticks([])
			st.pyplot(fig6)

def performance_sismica_2d():
	st.info(""" 
		Archivo Segy - Atributos básicos, análisis de encabezados y trazado En este cuaderno se explicarán los conceptos básicos del manejo de archivos segyio. Cubre:
		- Lectura de archivos
		- Análisis de encabezados bin, text y trace
		- Seguimiento rápido de encabezados
		- Sección (asumiendo que el segy es una sección post-apilamiento 2D)
		- La línea sísmica 2D en esta publicación es del Archivo de Datos Sísmicos NPRA de USGS y es de dominio público. El número de línea es 3X_75_PR.

	""") # Azul

	import re
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	import segyio
	filename = 'data/basic/3X_75_PR.SGY'

	with st.beta_expander('Funciones de utilidad y Obtener atributos básicos y analizar encabezados'):
		st.warning("Las siguientes funciones están definidas para dividir el flujo de trabajo en partes independientes. Permiten analizar tanto el texto como los encabezados de seguimiento en un dict y un marco de datos de pandas, respectivamente.") # Azul
		st.warning('La lectura del archivo se realiza dentro de una declaración with. El argumento ignore_geometry = True se pasa para segys no específicos de 3D.')
		def parse_trace_headers(segyfile, n_traces):
		    '''
		    Parse the segy file trace headers into a pandas dataframe.
		    Column names are defined from segyio internal tracefield
		    One row per trace
		    '''
		    # Get all header keys
		    headers = segyio.tracefield.keys
		    # Initialize dataframe with trace id as index and headers as columns
		    df = pd.DataFrame(index=range(1, n_traces + 1),
		                      columns=headers.keys())
		    # Fill dataframe with all header values
		    for k, v in headers.items():
		        df[k] = segyfile.attributes(v)[:]
		    return df


		def parse_text_header(segyfile):
		    '''
		    Format segy text header into a readable, clean dict
		    '''
		    raw_header = segyio.tools.wrap(segyfile.text[0])
		    # Cut on C*int pattern
		    cut_header = re.split(r'C ', raw_header)[1::]
		    # Remove end of line return
		    text_header = [x.replace('\n', ' ') for x in cut_header]
		    text_header[-1] = text_header[-1][:-2]
		    # Format in dict
		    clean_header = {}
		    i = 1
		    for item in text_header:
		        key = "C" + str(i).rjust(2, '0')
		        i += 1
		        clean_header[key] = item
		    return clean_header

		with segyio.open(filename, ignore_geometry=True) as f:
		    # Get basic attributes
		    n_traces = f.tracecount
		    sample_rate = segyio.tools.dt(f) / 1000
		    n_samples = f.samples.size
		    twt = f.samples
		    data = f.trace.raw[:]  # Get all data into memory (could cause on big files)
		    # Load headers
		    bin_headers = f.bin
		    text_headers = parse_text_header(f)
		    trace_headers = parse_trace_headers(f, n_traces)
		f'N Traces: {n_traces}, N Samples: {n_samples}, Sample rate: {sample_rate}ms'
		st.write('N Traces: {}, N Samples: {}, Sample rate: {}ms'.format(n_traces,sample_rate,n_samples))

		st.warning('La variable twt almacena el tiempo bidireccional para cada muestra.')
		st.write(bin_headers)
		st.write(text_headers)
		st.write(trace_headers.columns)

		st.write(trace_headers.head())

	with st.beta_expander('Plotting Headers'):
		st.warning('El control de calidad básico del encabezado se puede lograr trazando juntos los valores relevantes. Aquí se muestra un ejemplo rápido de cómo crear un diagrama de encabezado 2D (sin ningún significado)')
		plt.style.use('ggplot')  # Use ggplot styles for all plotting
		fig = plt.figure(figsize=(12, 8))
		ax = fig.add_subplot(1, 1, 1)
		ax.plot(trace_headers['FieldRecord'], trace_headers['TRACE_SEQUENCE_FILE'], '-k')
		ax.set_xlabel('Field Record Number (FFID)')
		ax.set_ylabel('Trace sequence within file')
		ax.set_title('Basic Header QC')

		st.pyplot(fig)

	with st.beta_expander('Plotting section'):
		st.warning('La forma sencilla de mostrar un archivo segy desde el formato segyio es usar matplotlib.pyplot.imshow, que trata las matrices 2D como imágenes. Para escalar y estandarizar la pantalla, es útil obtener el percentil n de las amplitudes')
		clip_percentile = 99
		vm = np.percentile(data, clip_percentile)
		f'The {clip_percentile}th percentile is {vm:.0f}; the max amplitude is {data.max():.0f}'
		st.warning('¡Empecemos a trazar! Observe que tenemos que transponer la matriz para trazarla así. La razón es que estamos almacenando cosas con trazas en la primera dimensión , por conveniencia. De esta forma, los datos [0] se refieren a la primera traza, no a la primera muestra. Pero imshow asume que estamos viendo una especie de imagen, con filas que atraviesan la imagen. La extensión de la imagen también se completa a partir del número de trazas que se mostrarán y los tiempos bidireccionales de las muestras.')
		fig2 = plt.figure(figsize=(18, 8))
		ax = fig2.add_subplot(1, 1, 1)
		extent = [1, n_traces, twt[-1], twt[0]]  # define extent
		ax.imshow(data.T, cmap="RdBu", vmin=-vm, vmax=vm, aspect='auto', extent=extent)
		ax.set_xlabel('CDP number')
		ax.set_ylabel('TWT [ms]')
		ax.set_title(f'{filename}')

		st.pyplot(fig2)
		
def sismica():
	st.sidebar.warning("Bienvenido al módulo de sísmica") #Amarillo
	seleccion_sismica = st.sidebar.selectbox("Escoja un método de evaluación",['Select an option', '2D', '3D'], format_func=lambda x: 'Select an option' if x == '' else x)
	if seleccion_sismica == '2D':
		pro = menu_evaluador()
		if pro == '🏠 Overview':
			boton_iniciar_demo =st.sidebar.button('Iniciar Overview')
			if boton_iniciar_demo == True:
				sismica_2d()
		if pro == '⏲️ Performance':
			performance_sismica_2d()


			
			
			

			

			
			
			

			

	