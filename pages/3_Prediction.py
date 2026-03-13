
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(page_title="Prediction", layout="wide")
st.title("Page 3 - Prediction de prix")
st.markdown("---")

if not os.path.exists('app/model_gb.pkl'):
    st.warning("Entrainer les modeles dans la page Training.")
    st.stop()

modele        = joblib.load('app/model_gb.pkl')
sc            = joblib.load('app/scaler.pkl')
noms_features = joblib.load('app/features.pkl')
df_encode     = joblib.load('app/df_encoded.pkl')
maisondf      = st.session_state.get('maisondf', df_encode)

st.subheader("Caracteristiques de la maison")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("#### Surface")
    surface_totale = st.slider("Surface totale (ft2)",  500,  6000, 2000, 50)
    superficie     = st.slider("Superficie terrain",    1000, 20000, 8000, 500)
    nb_salles_bain = st.slider("Salles de bain",        1.0,  5.0,  2.0,  0.5)

with c2:
    st.markdown("#### Qualite")
    qualite_gen = st.slider("Qualite generale (1-10)", 1, 10, 5)
    condition   = st.slider("Condition generale (1-10)", 1, 10, 5)
    age         = st.slider("Age du logement (annees)", 0, 150, 20)

with c3:
    st.markdown("#### Equipements")
    surface_garage = st.slider("Surface garage (ft2)",  0, 1500, 400, 50)
    places_garage  = st.slider("Places de garage",      0, 4, 2)
    cheminees      = st.slider("Cheminees",             0, 3, 1)

st.markdown("---")

if st.button("Predire le prix", use_container_width=True):

    X_input = pd.DataFrame(columns=noms_features)
    X_input.loc[0] = df_encode[noms_features].mean()

    valeurs = {
        'SurfaceTotale' : surface_totale,
        'LotArea'       : superficie,
        'NbSallesDeBain': nb_salles_bain,
        'OverallQual'   : qualite_gen,
        'OverallCond'   : condition,
        'AgeLogement'   : age,
        'GarageArea'    : surface_garage,
        'GarageCars'    : places_garage,
        'Fireplaces'    : cheminees,
    }

    for col, val in valeurs.items():
        if col in X_input.columns:
            X_input[col] = val

    prix = modele.predict(sc.transform(X_input.astype(float)))[0]

    c1, c2 = st.columns(2)

    with c1:
        st.success(f"Prix estime : ${prix:,.0f}")
        recap = pd.DataFrame({
            'Caracteristique': list(valeurs.keys()),
            'Valeur'         : list(valeurs.values())
        })
        st.dataframe(recap, use_container_width=True)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(
            ['Min', 'Median', 'Moyen', 'Predit', 'Max'],
            [maisondf['SalePrice'].min(), maisondf['SalePrice'].median(),
             maisondf['SalePrice'].mean(), prix, maisondf['SalePrice'].max()],
            color=['#4ECDC4', '#4ECDC4', '#4ECDC4', '#FF6B6B', '#4ECDC4'],
            alpha=0.8, edgecolor='black'
        )
        ax.set_xlabel('Prix ($)')
        ax.set_title('Position sur le marche')
        ax.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("Ajuster les valeurs puis cliquer sur Predire le prix")
