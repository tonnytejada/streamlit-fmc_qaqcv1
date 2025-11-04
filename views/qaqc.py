import streamlit as st
import pandas as pd
import numpy as np
from scrpt import scrptqc
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("ARD and HARD analysis")
# Subida de archivo CSV
uploaded_file = st.file_uploader("Choose a file", type="csv")
# Verificar si el archivo fue cargado
if uploaded_file is not None:
    # Leer el archivo CSV
    df1x = pd.read_csv(uploaded_file)
    #df = df1x[df1x['QC_Category'].str.startswith('y')]
    if 'LimitLower' in df1x.columns:
        ldl = df1x['LimitLower'].min()
    else:
        ldl = 0.01  # o el valor por defecto que desees
    # Mostrar un resumen de las primeras filas del DataFrame
    st.write(df1x.head())
    # Parámetros que el usuario puede modificar
    unique_qc_types = df1x['QC_Category'].unique()
    unique_element = df1x['Element'].unique()
    #st.write(unique_qc_types)
    orginal_col = scrptqc.buscar_columna(df1x, ['Orig', 'OriginalResult', 'Original_Au_ppm', 'Orig_Au'])  
    duplicate_col = scrptqc.buscar_columna(df1x, ['Dup', 'RepeatResult', 'Duplicate_Au_ppm','Au_ppm'])
    with st.sidebar:
        st.subheader("Input Parameters")
        QC = st.multiselect('Value for QC Category', options=unique_qc_types, default=unique_qc_types)

        org = st.text_input("Value for Original", value=orginal_col)# 'org'
        dup = st.text_input("Value for Duplicate", value=duplicate_col) #'dup'
        #Elem = st.text_input("Element", value="Au")
        Elem = st.multiselect('Value for Element', options=unique_element, default=unique_element)
        ldl = st.number_input("Detection limit (Limit Lower)", value=ldl, format="%.3f")
        xLDL = st.number_input("Times detection limit (xLw)", value=10, step=1, format="%i")
        
        ARD = st.radio("Select ARD percentage",
                            options=[0.3, 0.2,0.15, 0.1])
        if ARD == 0.3:
            st.write(":red[You selected ARD for Field Duplicate 30%]")
        elif ARD == 0.2:
            st.write(":red[You selected ARD for Prep. Duplicates or Reject assays 20%]")
        elif ARD == 0.15:       
            st.write(":red[You selected ARD for Check assay 15%]")
        elif ARD == 0.1:
            st.write(":red[You selected ARD for Lab. Duplicates or Duplicate assays 10%]")
            #st.write("This is normal text. This text is :red[red] and this is :blue[blue].")
        Outliers = st.slider("If exists Outliers", 
                            min_value=0.5, 
                            max_value=2.0, 
                            value=2.0, 
                            step=0.1, 
                            format="%.2f", 
                            key="Outliers")

    df = df1x[df1x['QC_Category'].isin(QC)&df1x['Element'].isin(Elem)]
    max_value = max(df[org].max(), df[dup].max())

    rounded_max = 5 * round(max_value / 5)
    with st.sidebar:
        MAX = st.slider("Select Max value", 
                            min_value=0, 
                            max_value=rounded_max+10, 
                            value=rounded_max, 
                            step=1, 
                            format="%.2f", 
                            key="MAX")
    filtered_data = (df
                    .assign(
                    difABS=lambda x: abs(x[org] - x[dup]) / ((x[org] + x[dup]) / 2),
                    difREL=lambda x: (x[org] - x[dup]) / ((x[org] + x[dup]) / 2) * 100,
                    Success='n')
                    )    
    df = filtered_data[filtered_data['difABS'] < Outliers]    
    dfout = filtered_data[filtered_data['difABS'] > Outliers]   
    unique_dfout = len(dfout)
    # Subtítulo para ARD Summary
    st.subheader("ARD Summary", divider="gray")
    # Llamar a la función que procesa los datos (asegúrate de que la función esté definida correctamente)
    if 'scrptqc' in globals():
        filtered_data_ard = scrptqc.filter_calcARD(df, QC, org, dup, Elem, ldl, ARD, xLDL)
        st.write(filtered_data_ard)
    else:
        st.warning("The scrptqc.filter_calcARD function is not defined or imported.")

    # Subtítulo para HARD Summary
    st.subheader("HARD Summary", divider="gray")
    
    # Llamar a la función para HARD Summary
    if 'scrptqc' in globals():
        #df1 = df.copy()
        filtered_data_hard = scrptqc.filter_calcHARD(df, org, dup, Elem, ldl)
        st.write(filtered_data_hard)
    else:
        st.warning("The scrptqc.filter_calcHARD function is not defined or imported.")

    st.subheader("Final Chart", divider="gray")
    # Subtítulo para gráficos
    if  'scrptqc' in globals():
        # Leer los archivos CSV en DataFrames
        #df = pd.read_csv(uploaded_file)
        dataf = filtered_data_ard.copy()
        datafh = filtered_data_hard.copy()

        # Llamar a la función resumen con los parámetros y los datos
        scrptqc.resumen(dataf, datafh, maxx=MAX, org=org, dup=dup, Elem=Elem, ldl=ldl, ARD=ARD)

        # Mostrar mensaje de éxito
        st.success("Chart generated successfully.")
    else:
        st.warning("Please upload CSVs")

    # Subtítulo para Outliers Summary
    st.subheader("Outliers Summary", divider="gray")
    st.write(f'Outlier Count: {unique_dfout}')
    st.write(dfout)
    # 
    st.subheader("Interpretation", divider="gray")
    #
    X = df[org].values#[:, np.newaxis]
    y = df[dup].values
    #st.write(df1x[org].corr(df1x[dup]))
    #st.write(df[org].corr(df[dup]))
    #st.write((df[org].corr(df[dup]))**2)
    scrptqc.calculate_rma(X, y)
    #
else:
    st.warning("Please upload correct CSVs")