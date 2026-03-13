
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Training", layout="wide")
st.title("Page 2 - Entrainement")
st.markdown("---")

if 'maisondf' not in st.session_state:
    st.warning("Charger les donnees dans la page Data.")
    st.stop()

@st.cache_resource
def entrainer(_maisondf):
    maisondf = _maisondf.copy()
    le = LabelEncoder()
    for col in maisondf.select_dtypes(include='object').columns:
        maisondf[col] = le.fit_transform(maisondf[col].astype(str))

    X = maisondf.drop(columns=['SalePrice', 'Id'])
    y = maisondf['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_sc, y_train)

    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    gb.fit(X_train_sc, y_train)

    joblib.dump(gb,                'app/model_gb.pkl')
    joblib.dump(sc,                'app/scaler.pkl')
    joblib.dump(X.columns.tolist(),'app/features.pkl')
    joblib.dump(maisondf,          'app/df_encoded.pkl')

    resultats = {
        'Regression Lineaire': {'test': lr.predict(X_test_sc)},
        'Random Forest'      : {'test': rf.predict(X_test_sc)},
        'Gradient Boosting'  : {'test': gb.predict(X_test_sc)},
    }
    return resultats, y_test, rf, X.columns.tolist()

with st.spinner("Entrainement en cours..."):
    resultats, y_test, rf, noms_features = entrainer(st.session_state['maisondf'])

st.success("Entrainement termine.")

lignes = []
for nom, pred in resultats.items():
    lignes.append({
        'Modele'     : nom,
        'R2 (Test)'  : round(r2_score(y_test, pred['test']), 4),
        'MAE (Test)' : round(mean_absolute_error(y_test, pred['test']), 0),
        'RMSE (Test)': round(np.sqrt(mean_squared_error(y_test, pred['test'])), 0)
    })

tableau = pd.DataFrame(lignes)
st.dataframe(tableau, use_container_width=True)
st.success(f"Meilleur modele : {tableau.loc[tableau['R2 (Test)'].idxmax(), 'Modele']}")

st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, resultats['Regression Lineaire']['test'], alpha=0.3, s=10, color='steelblue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_title(f"Regression Lineaire - R2={r2_score(y_test, resultats['Regression Lineaire']['test']):.3f}")
    ax.set_xlabel('Prix reel ($)')
    ax.set_ylabel('Prix predit ($)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with c2:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, resultats['Gradient Boosting']['test'], alpha=0.3, s=10, color='forestgreen')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_title(f"Gradient Boosting - R2={r2_score(y_test, resultats['Gradient Boosting']['test']):.3f}")
    ax.set_xlabel('Prix reel ($)')
    ax.set_ylabel('Prix predit ($)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")
importance = pd.DataFrame({
    'Feature'   : noms_features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(importance['Feature'], importance['Importance'], color='steelblue', alpha=0.8)
ax.set_xlabel('Importance')
ax.set_title('Top 10 features')
ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
