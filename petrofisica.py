import streamlit as st
from io import StringIO
import pandas as pd
import lasio
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

def menu_evaluador():
	proceso = st.sidebar.radio('Select the process',['🏠 Overview','⏲️ Performance','📈 Analytics','🚀 Landing','📋 Projects'])
	return proceso

@st.cache(allow_output_mutation=True)
def lectura(archivo_las):
	try:
		bytes_data = archivo_las.read()
		str_io = StringIO(bytes_data.decode('Windows-1252'))
		las_file = lasio.read(str_io)
		df = las_file.df()
		df['DEPTH'] = df.index
	except:
		st.sidebar.error("Archivo no admitido ") # Rojo
	return df, las_file

def overview(df, las_file):
	with st.beta_expander('Dataset de registros'):
		st.header("Data Frame")
		st.write(df)

	with st.beta_expander ('Informacion del regitro'):
		nombre_pozo = las_file.header['Well'].WELL.value
		#pais = las_file.header['Well'].COUNT.value
		campo = las_file.header['Well'].FLD.value
		provincia = las_file.header['Well'].PROV.value
		compania = las_file.header['Well'].COMP.value
		unidades_profundidad = las_file.header['Well'].STRT.unit
		profundidad_min = df.index.values[0]
		profundidad_max = df.index.values[-1]
		cola , colb = st.beta_columns(2)
		with cola:
			st.write('Prufundidad inicial: {} {}'.format(profundidad_min,unidades_profundidad))
			st.write('Prufundidad final: {} {}'.format(profundidad_max, unidades_profundidad))
			st.write('Nombre del pozo: {}'.format(nombre_pozo))
			st.write('Nombre del campo: {}'.format(campo))
		with colb:
			#st.write('Pais: {}'.format(pais))
			st.write('Provincia: {}'.format(provincia))
			st.write('Compania: {}'.format(compania))
			st.write('Unidad profundiad: {}'.format(unidades_profundidad))
	with st.beta_expander ('Visualización del registro'):
		
		col1, col2 = st.beta_columns(2)
		with col1:
			lista_registros =list(df.columns)
			registro = st.selectbox('seleccione un regitro', options= lista_registros)
			df_reg=df
			df_reg= df_reg[['DEPTH',registro]]
			df_mask=df_reg[registro] >= 0
			filtered_df = df_reg[df_mask]
			#st.write(filtered_df)

		with col2:
			fig , ax = plt.subplots(figsize=(3,5))
			ax.plot(filtered_df[registro], filtered_df.DEPTH)
			ax.invert_yaxis()
			ax.set_ylabel("Depth")
			ax.set_xlabel(registro)
			ax.grid()
			st.pyplot(fig)

	with st.beta_expander('Clean Dataset'):
		try:
			st.title("Limpieza y selección de datos ")
			selected_columns = ['NPHI','RHOB','GR']
			df_shorter = df[selected_columns]
			df_clean = df_shorter.dropna(subset=['NPHI','RHOB','GR'],axis=0, how='any')
			st.header("DataFrame sin valores nulos")
			st.write(df_clean)
			st.header("Grafico Data Frame Clean")
			f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,8) )
			logs = ['NPHI','RHOB','GR']
			colors = ['green','blue','red','black']
			for i,log,color in zip(range(3),logs,colors):
				ax[i].plot(df_clean[log], df_clean.index, color=color)
				ax[i].invert_yaxis()
			for i,log,color in zip(range(3),logs,colors):
				ax[i].set_xlabel(log)
				ax[0].set_ylabel("Depth(ft)")
				ax[i].grid()
			st.pyplot(f)
		except:
			st.sidebar.error("No fue posible cargar data set limpio y el plot") # Rojo

def performance(df):
	import numpy as np
	from scipy.stats import gaussian_kde
	
	with st.beta_expander('Limpieza de datos'):
		df['DEPT']= df.index
		list_columns = list(df.columns)
		list_columns
		selected_columns = list_columns
		df_shorter = df[selected_columns]
		logs = df_shorter.dropna(subset=list_columns,axis=0, how='any')
		st.write(logs)

	with st.beta_expander('Plot Logs'):

		select_registro = st.selectbox('seleccione un resgitro', options= list_columns)
		top=logs.index.values[0]
		base = logs.index.values[-1]
		tops = ('TOP','BASE')
		tops_depths=(top, base)
		g = plt.figure(figsize=(18,5))
		plt.plot(logs['DEPT'],logs[select_registro], lw=1.25)
		plt.xlim(top, base)
		plt.grid()
		st.pyplot(g)
	with st.beta_expander('Define a Zone to make petrophysic analysis'):

		def GRvsRhob(GR, Rhob,select_registro1,select_registro2):
			x=GR
			y=Rhob
			xy = np.vstack([x,y])
			z = gaussian_kde(xy)(xy)
			fig, ax = plt.subplots()
			ax.scatter(x, y, c=z, s=50, alpha=0.3)
			ax.set_xlabel(select_registro1)
			ax.set_ylabel(select_registro2)
			ax.grid()
			st.pyplot(fig)

		select_registro1 = st.selectbox('Select 1', options= list_columns, index = 0)
		select_registro2 = st.selectbox('select 2', options= list_columns, index = 1)

		top=logs.index.values[0]
		base = logs.index.values[-1]

		Z1 = logs[(logs.DEPT >= top) & (logs.DEPT <= base)]
		
		col3, col4, col5 = st.beta_columns(3)

		with col3:
			Z1_filtered=Z1
			Z1_filtered= Z1_filtered[[select_registro1,select_registro2]]
			st.write(Z1_filtered)
		with col4:
			GRvsRhob(Z1[select_registro1], Z1[select_registro2], select_registro1,select_registro2)

		with col5:
			select_registro3 = st.selectbox('select', options= list_columns, index = 3)
			fig_histograma, ax1 = plt.subplots()
			ax1.set_xlabel(select_registro3)
			ax1.set_ylabel('Frequency')
			ax1.hist(Z1[select_registro3].dropna(),bins=30,color='blue', alpha=0.8)
			# With plot you can make your own guesses of distribution families in data
			# In this case one will be Sand family and the other clay family
			st.pyplot(fig_histograma)

	with st.beta_expander('Inspect GR curve to defime GR min and GR max'):
		col6, col7 = st.beta_columns(2)

		with col6:
			gr_plt = plt.figure(figsize=(15,3))
			plt.plot(Z1.DEPT, Z1.GR)
			st.header('Gamaray')
			st.pyplot(gr_plt)
			GRmax = Z1["GR"].max()
			GRmin = Z1["GR"].min()
			st.write('Gamaray mínimo: {}'.format(GRmin))
			st.write('Gamaray máximo: {}'.format(GRmax))

		with col7:
			st.header('VCL')
			Z1['VCL'] = 1 - ((GRmax - Z1.GR) / (GRmax - GRmin))
			vcl_plt = plt.figure(figsize=(15,3))
			plt.plot(Z1.DEPT, Z1.VCL)
			st.pyplot(vcl_plt)

	def phi(den_log, den_matrix, den_fluid, den_shale):
		phi = (den_log - den_matrix) / (den_fluid - den_matrix)
		return phi
	def phi_shale(den_shale, den_matrix, den_fluid):
		phi_shale = (den_shale - den_matrix) / (den_fluid - den_matrix)
		return phi_shale
	def phi_shale_corrected(den, den_matrix, den_fluid, den_shale, vcl):
		phi = (den - den_matrix) / (den_fluid - den_matrix)
		phi_shale = (den_shale - den_matrix) / (den_fluid - den_matrix)
		phi_sh_corr = phi - vcl * phi_shale
		return phi_sh_corr

	with st.beta_expander('Function to calculate Porosity using only Density Log'):
		col8, col9 = st.beta_columns(2)

		with col8:
			#den_matrix, den_fluid, den_shale = 2.65, 1.1, 2.4
			den_matrix = st.number_input('Input den_matrix ',min_value=0.00, max_value=None, value=2.65,  step=0.01 )
			den_fluid = st.number_input('Input den_fluid ',min_value=0.00, max_value=None, value=1.10,  step=0.01 )
			den_shale = st.number_input('Input den_shale ',min_value=0.00, max_value=None, value=2.40,  step=0.01 )
		with col9:
			Z1['PHID']=phi(Z1.RHOB, den_matrix, den_fluid, den_shale)
			Z1['PHIDshc']=phi_shale_corrected(Z1.RHOB, den_matrix, den_fluid, den_shale, Z1.VCL).clip(0,1)
			PHIDshc_plt = plt.figure(figsize=(15,15))
			plt.plot(Z1['DEPT'],Z1['PHIDshc'], lw=0.7)
			#plt.ylim(-0.1, 0.3)
			#plt.xlim(10000, 10100)
			st.pyplot(PHIDshc_plt)

		
		

	Z1['PHIE']=Z1['PHIDshc']

	def SW_archie(Rw, Rt, Poro, a, m, n):
		F = a / (Poro**m)
		Sw_archie = (F * Rw/Rt)**(1/n)
		return Sw_archie

	def SW_simand(Rw, Rt, Rsh, Poro, a, m, Vsh):
		F = Vsh/Rsh
		G = (a*Rw)/(Poro**m)
		Sw_simand= (G/2)*(((F)**2+ (4/(G*Rt)))**0.5 - F)
		return Sw_simand

	a=1 #turtuosity factor
	m=1.65 #cementation factor
	n=1.7 #saturation exponent

	rwa=0.32
	Rw =rwa
	Rsh=4

	try:

		top=logs.index.values[0]
		base = logs.index.values[-1]
		
		Z1['SWa']=(SW_simand(Rw,Z1.RS,Rsh,Z1.PHIE,a,m,Z1.VCL)).clip(0,1)

		Z1['BVW']=Z1['SWa']*Z1['PHIE']

		Zone=Z1.copy()

		NetPay = Z1[(Z1.DEPT>=top) & (Z1.DEPT<=base)]
		ssb=((NetPay.VCL <= 0.25) & (NetPay.SWa <= 0.65) & (NetPay.PHIE >= 0.05))
		temp_lfc=np.zeros(np.shape(NetPay.VCL))
		temp_lfc[ssb.values]=1
		NetPay.insert(NetPay.columns.size, 'PayFlag', temp_lfc)

	except:
		st.error("El registro requiere observación") # Rojo



	def Petrophysic_plot(top_depth,bottom_depth):
		try:
	    
		    f = Zone[(Zone.DEPT >= top_depth) & (Zone.DEPT <= bottom_depth)]
		    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12,10), sharey=True)
		    fig.suptitle("Well Interpretation", fontsize=22)
		    fig.subplots_adjust(top=0.75,wspace=0.1)

		#General setting for all axis
		    for axes in ax:
		        axes.set_ylim (top_depth,bottom_depth)
		        axes.invert_yaxis()
		        axes.yaxis.grid(True)
		        axes.get_xaxis().set_visible(False) 
		        for (i,j) in zip(tops_depths,tops):
		            if ((i>=top_depth) and (i<=bottom_depth)):
		                axes.axhline(y=i, linewidth=0.5, color='black')
		                axes.text(0.1, i ,j, horizontalalignment='center',verticalalignment='center')
		    
		        
		#1st track: GR, CALI 
		    
		    ax01=ax[0].twiny()
		    ax01.set_xlim(0,120)
		    ax01.plot(logs.GR, logs.DEPT, label='GR[api]', color='green') 
		    ax01.spines['top'].set_position(('outward',0))
		    ax01.set_xlabel('GR[api]',color='green')    
		    ax01.tick_params(axis='x', colors='green')
		    ax01.grid(True)
		    
		    ax02=ax[0].twiny()
		    ax02.set_xlim(0,30)
		    ax02.plot(NetPay.PayFlag, NetPay.DEPT, label='payflag', color='blue', lw=0.5)
		    ax02.spines['top'].set_position(('outward',40))
		    ax02.fill_betweenx(NetPay.DEPT,0,NetPay.PayFlag,color='gray')
		    ax02.set_xlabel('PayFlag',color='black')    
		    ax02.tick_params(axis='x', colors='black')  
		    
		      
		#2nd track: Resistivities

		    ax11=ax[1].twiny()
		    ax11.set_xlim(0.1,50)
		    ax11.set_xscale('linear')
		    ax11.grid(True)
		    ax11.spines['top'].set_position(('outward',40))
		    ax11.set_xlabel('Res[m.ohm]', color='blue')
		    ax11.plot(logs.RS, logs.DEPT, label='Res[m.ohm]', color='blue')
		    ax11.tick_params(axis='x', colors='blue')    
		        
		                # Petrophycal curves

		#4th track: VCL, RHOBC

		    ax21=ax[2].twiny()
		    ax21.set_xlim(0,1)
		    ax21.spines['top'].set_position(('outward',40))
		    ax21.plot(f.VCL, f.DEPT, label='VCL', color='green',linewidth=0.5)
		    ax21.set_xlabel('VCL', color='green')    
		    ax21.tick_params(axis='x', colors='green')
		    ax21.fill_betweenx(f.DEPT,f.VCL,1,color='gold',label= 'Vclay')
		    ax21.fill_betweenx(f.DEPT,f.VCL,0,color='gray',label= 'Shale')
		    ax21.legend(loc='lower left')
		    
		    ax22=ax[2].twiny()
		    ax22.set_xlim(1.95,2.95)
		    ax22.plot(f.RHOB, f.DEPT ,label='RHOB[g/cc]', color='red', linewidth=0.8) 
		    ax22.spines['top'].set_position(('outward',80))
		    ax22.set_xlabel('RHOB[g/cc]',color='red')
		    ax22.tick_params(axis='x', colors='red')    
		    
		#4th track: PHIE, BVW

		    ax31=ax[3].twiny()
		    ax31.grid(True)
		    ax31.set_xlim(0,0.3)
		    ax31.plot(f.PHIE, f.DEPT, label='PHIEQ', color='black', linewidth=0.5)
		    ax31.spines['top'].set_position(('outward',0))
		    ax31.fill_betweenx(f.DEPT,0,f.PHIE,color='green', label='Oil')
		    ax31.set_xlabel('PHIEQ', color='black')    
		    ax31.tick_params(axis='x', colors='black')
		    ax31.legend(loc='lower left')
		    
		    ax32=ax[3].twiny()
		    ax32.set_xlim(0,0.3)
		    ax32.plot(f.BVW, f.DEPT, label='BVW', color='blue', linewidth=0.5)
		    ax32.spines['top'].set_position(('outward',40))
		    ax32.fill_betweenx(f.DEPT,0,f.BVW,color='lightblue', label='Water')
		    ax32.set_xlabel('BVW', color='blue')    
		    ax32.tick_params(axis='x', colors='blue')
		    ax32.legend(loc='lower right')

		    ax33=ax[3].twiny()
		    ax33.set_xlim(-1,1)
		    ax33.plot(f.SWa, f.DEPT, label='SWa', color='lightgreen', linewidth=0.8)
		    ax33.spines['top'].set_position(('outward',80))
		    ax33.fill_betweenx(f.DEPT,1,f.SWa,color='green',alpha=0.5)
		    ax33.set_xlabel('SWa', color='green')    
		    ax33.tick_params(axis='x', colors='green')

		    plt.savefig ('Well.png', dpi=200, format='png')
		    st.pyplot(fig)
		except:
			st.error("No fue posible continuar esta sección") # Rojo
	with st.beta_expander('Plot humble analysis'):
		top=logs.index.values[0]
		base = logs.index.values[-1]


		zona_superior = st.number_input('Limite superior de evaluacion',min_value=top, max_value=base, value= top )
		zona_inferior = st.number_input('Limite inferior de evaluacion',min_value=top, max_value=base, value= base )

		Petrophysic_plot(zona_superior,zona_inferior)

def analytics(df):
	import numpy as np
	import matplotlib.pyplot as plt
	import pandas as pd
	import seaborn as sns

	st.info('Una demostración sobre la predicción de registros P-Sonic mediante el Machine Learning')

	with st.beta_expander('Well Train'):

		df['DEPT']= df.index
		list_columns = list(df.columns)
		list_columns
		selected_columns = list_columns
		df_shorter = df[selected_columns]
		logs = df_shorter.dropna(subset=list_columns,axis=0, how='any')
		well_train = logs
		st.write(well_train.head(10))

	with st.beta_expander('We plot the pairplots using seaborn as the first EDA.'):
		feature_target = ['NPHI', 'RHOB', 'GR', 'RT', 'CALI', 'DT']

		fig1sns = sns.pairplot(well_train, vars=feature_target, diag_kind='kde',
             plot_kws = {'alpha': 0.6, 's': 30, 'edgecolor': 'k'})
		st.pyplot(fig1sns)

	with st.beta_expander('We plot the correlation heatmap as the second EDA.'):
		well_train_only_features = well_train[feature_target]

		# Generate a mask for the upper triangle
		mask = np.zeros_like(well_train_only_features.corr(method = 'spearman') , dtype=np.bool)
		mask[np.triu_indices_from(mask)] = True

		# Generate a custom diverging colormap
		cmap = sns.cubehelix_palette(n_colors=12, start=-2.25, rot=-1.3, as_cmap=True)

		# Draw the heatmap with the mask and correct aspect ratio
		fig= plt.figure(figsize=(12,10))
		sns.heatmap(well_train_only_features.corr(method = 'spearman') ,annot=True,  mask=mask, cmap=cmap, vmax=.3, square=True)

		st.pyplot(fig)

	with st.beta_expander('Now we do normalization. The best method that we will use is the power transform using Yeo-Johnson method.'):
		from sklearn.compose import ColumnTransformer
		from sklearn.preprocessing import PowerTransformer

		# transform the RT to logarithmic
		well_train['RT'] = np.log10(well_train['RT'])

		# normalize using power transform Yeo-Johnson method
		scaler = PowerTransformer(method='yeo-johnson')

		# columns
		colnames = well_train.columns
		only_feature = ['NPHI', 'RHOB', 'GR', 'RT', 'CALI'] # only feature column names
		only_target = 'DT' # only target column names
		feature_target = np.append(only_feature, only_target) # feature and target column names

		## ColumnTransformer
		column_drop = ['WELL', 'DEPTH']
		ct = ColumnTransformer([('transform', scaler, feature_target)], remainder='passthrough')

		## fit and transform
		well_train_norm = ct.fit_transform(well_train)

		## convert to dataframe
		well_train_norm = pd.DataFrame(well_train_norm, columns=colnames)
		well_train_norm['WELL']='LAG-040'

		## up until this step, if we pass df.dtypes, we can see all the results are object. 
		## So, we change the dtypes to solve this.
		x = well_train_norm[feature_target].astype(float)
		y = well_train_norm['WELL'].astype(str)
		z = well_train_norm['DEPTH'].astype(float)

		well_train_norm = pd.concat([x, y, z], axis=1)

		st.write(well_train_norm.head(10))

	with st.beta_expander('Next we remove the outliers. The best method that we´ll use is One-class SVM.'):
		from sklearn.svm import OneClassSVM

		# make copy of well_train_norm, called well_train_dropped
		well_train_drop = well_train_norm.copy()

		# on the well_train_drop, drop WELL and DEPTH column
		well_train_drop = well_train_norm.drop(['WELL', 'DEPTH'], axis=1)

		# removing outliers using One-class SVM
		svm = OneClassSVM(nu=0.1)
		yhat = svm.fit_predict(well_train_drop)
		mask = yhat != -1
		well_train_svm = well_train_norm[mask]

		st.write(well_train_svm.head(10))

	with st.beta_expander('Next we make pairplot to compare the training data before and after normalization and outlier removal.'):

		# visualize the pairplot after normalization and outliers removed
		import seaborn as sns

		fig_pair = sns.pairplot(well_train_svm, vars=feature_target,
		             diag_kind='kde',
		             plot_kws = {'alpha': 0.6, 's': 30, 'edgecolor': 'k'})
		st.pyplot(fig_pair)








def petrofisica():
	st.sidebar.warning("Bienvenido al módulo petrofísico elija el metodo de evaluacion") #Amarillo
	metodo_evaluacion = st.sidebar.selectbox("Escoja un método de evaluación",['Select an option', 'Data demo', 'Cargar archivo LAS'], format_func=lambda x: 'Select an option' if x == '' else x)
		
	if  metodo_evaluacion == 'Data demo':
		st.sidebar.info("La data utilizada es un registro las precargado") # Azul
		proceso_evaluador = menu_evaluador()
		archivo_las = lasio.read("Archivos_las\LGAE-040.las")
		las = archivo_las
		df_demo = archivo_las.df()
		df_demo['DEPTH'] = df_demo.index
		
		if proceso_evaluador == '🏠 Overview':
			overview(df_demo, las)
		elif proceso_evaluador == '⏲️ Performance':
			performance(df_demo)

		elif proceso_evaluador == '📈 Analytics':
			analytics(df_demo)
		
	elif metodo_evaluacion == 'Cargar archivo LAS':
		archivo_las = st.sidebar.file_uploader("Import here las file" ,type=['.las' , '.LAS'] , key=None)
		if archivo_las is not None:
			df_upload, las_file = lectura(archivo_las)
			st.sidebar.success("Archivo las cargado exitosamente")
			proceso_evaluador = menu_evaluador()
			if proceso_evaluador == '🏠 Overview':
				overview(df_upload, las_file)
			elif proceso_evaluador == '⏲️ Performance':
				df_upload2 =df_upload
				performance(df_upload)

			
				





	

