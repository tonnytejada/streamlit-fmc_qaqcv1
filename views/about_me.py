import streamlit as st

# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small", vertical_alignment="center")
#with col1:
#    st.image("./assets/profile_image.png", width=230)

with col2:
    st.title("Alex Delgado", anchor=False)
    st.write("**Senior Resource Geologist and Data Analyst**")
    st.markdown(
        '''
        <a href="mailto:geology.modelling@gmail.com" style="text-decoration:none" aria-label="Enviar correo">
            <button type="button" style="
                display:inline-flex;
                align-items:center;
                justify-content:center;
                width:42px;
                height:42px;
                border-radius:8px;
                border:1px solid #ccc;
                background:#f5f5f5;
                cursor:pointer;
                font-size:20px;
            " title="Enviar correo">
                📧
            </button>
        </a>
        ''',
        unsafe_allow_html=True
    )
