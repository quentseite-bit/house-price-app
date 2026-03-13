import streamlit as st

if 'authentifie' not in st.session_state:
    st.session_state.authentifie = False

if not st.session_state.authentifie:
    mdp = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if mdp == st.secrets["password"]:
            st.session_state.authentifie = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect")
    st.stop()

st.set_page_config(page_title="House Prices - Seite Quentin", layout="wide")

st.title("House Prices - Application ML")
st.markdown("---")
st.markdown("""
### Navigation
- **Page 1 - Data** : Upload CSV et exploration
- **Page 2 - Training** : Entrainement et performances
- **Page 3 - Prediction** : Predire le prix d'une maison

### Comment utiliser
1. Charger le fichier **train.csv** dans la page Data
2. Entrainer les modeles dans la page Training
3. Predire un prix dans la page Prediction
""")
st.info("Commencer par la page Data.")
