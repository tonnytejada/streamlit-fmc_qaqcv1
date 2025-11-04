import streamlit as st
import pandas as pd
import numpy as np
from scrpt import scrptqc
import matplotlib.pyplot as plt
import datetime
# Título de la aplicación
st.title("Blank Analysis")
# Subida de archivo CSV
uploaded_file = st.file_uploader("Choose a file", type="csv")
# Verificar si el archivo fue cargado
# Obtener el mes anterior
hoy = datetime.date.today()
primer_dia_mes = hoy.replace(day=1)
mes_anterior = primer_dia_mes - datetime.timedelta(days=1)
nombre_mes_anterior = mes_anterior.strftime("%B")  
# Si se ha cargado un archivo, procesarlo
if uploaded_file is not None:
    # Leer el archivo CSV
    df1x = pd.read_csv(uploaded_file)
    # Mostrar un resumen de las primeras filas del DataFrame
    st.write(df1x.head())
    unique_qc_types = df1x['StandardID'].unique()
    unique_element = df1x['Element'].unique()
    unique_code = df1x['UnitCode'].unique()
    if 'sysResult' in df1x.columns:
        ldlx = df1x['sysResult'].min()*2
    else:
        ldlx = 0.01  # o el valor por defecto que desees
    
    with st.sidebar:
        st.subheader("Input Parameters")
        QC = st.multiselect('Standard', options=unique_qc_types, default=unique_qc_types)
        text = st.text_input("Title", value=nombre_mes_anterior)
        #ldl = st.number_input("Limit Lower (LOD)", value=ldlx, format="%.3f")
        ldl1 = st.number_input("LOD", value=0.01, step=0.005, format="%.3f")
        ldl2 = st.number_input("Warning Threshold", value=0.02, step=0.005, format="%.3f")
        ldl3 = st.number_input("Action Threshold", value=0.03, step=0.005, format="%.3f")
        Elem = st.multiselect('Value for Element', options=unique_element, default=unique_element)
        Unit = st.text_input("X Title", value=unique_code[0])
        
        
    st.subheader("BLK Summary", divider="gray")
    
    st.write(f"Count: {len(df1x)}")
    if 'sysResult' in df1x.columns:
        min_val = df1x['sysResult'].min()
        max_val = df1x['sysResult'].max()
        mean_val = df1x['sysResult'].mean()
        #st.write(f"Resumen de sysResult:")
        st.write(f"Min: {min_val} and Max: {max_val}")
        #st.write(f"Máximo: {max_val}")
        st.write(f"Media: {round(mean_val,4)}")
        st.write(f"Limit Lower (LOD) used: {ldlx}")
    else:
        st.warning("The 'sysResult' column does not exist in the file.")
    
    st.subheader("Final Chart", divider="gray")
    
    if  'scrptqc' in globals():
        # Leer los archivos CSV en DataFrames
        #df = pd.read_csv(uploaded_file)
        df = df1x.copy()
        # Llamar a la función resumen con los parámetros y los datos
        scrptqc.scatter_sysresult(df, lod=ldl1, 
                                warning_threshold=ldl2, 
                                action_threshold=ldl3,
                                standardID=unique_qc_types[0],
                                Element=unique_element[0],
                                UnitC=Unit,
                                title=text)
        # Mostrar mensaje de éxito
        st.success("Chart generated successfully.")
else:
    st.warning("Please upload CSVs")