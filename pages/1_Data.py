
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Data", layout="wide")
st.title("Page 1 - Exploration")
st.markdown("---")

fichier = st.file_uploader("Charger train.csv", type=["csv"])

if fichier is not None:

    @st.cache_data
    def charger_donnees(f):
        maisondf = pd.read_csv(f)

        cols_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                     'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                     'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                     'BsmtFinType2', 'MasVnrType']
        for col in cols_none:
            if col in maisondf.columns:
                maisondf[col].fillna('None', inplace=True)

        cols_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1',
                     'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
                     'BsmtHalfBath', 'MasVnrArea']
        for col in cols_zero:
            if col in maisondf.columns:
                maisondf[col].fillna(0, inplace=True)

        maisondf['LotFrontage'] = maisondf.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        maisondf['Electrical'].fillna(maisondf['Electrical'].mode()[0], inplace=True)
        maisondf['AgeLogement']    = 2025 - maisondf['YearBuilt']
        maisondf['SurfaceTotale']  = maisondf['GrLivArea'] + maisondf['TotalBsmtSF']
        maisondf['NbSallesDeBain'] = maisondf['FullBath'] + 0.5 * maisondf['HalfBath']
        return maisondf

    maisondf = charger_donnees(fichier)
    st.session_state['maisondf'] = maisondf

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes",             f"{maisondf.shape[0]:,}")
    c2.metric("Colonnes",           f"{maisondf.shape[1]}")
    c3.metric("Prix moyen",         f"${maisondf['SalePrice'].mean():,.0f}")
    c4.metric("Valeurs manquantes", f"{maisondf.isnull().sum().sum()}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Apercu", "Distribution", "Correlations"])

    with tab1:
        st.dataframe(maisondf.head(20), use_container_width=True)
        st.download_button("Telecharger", maisondf.to_csv(index=False).encode('utf-8'), "donnees.csv", "text/csv")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(maisondf['SalePrice'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('SalePrice ($)')
            ax.set_ylabel('Frequence')
            ax.set_title('Distribution des prix')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.boxplot(maisondf['SalePrice'], vert=True)
            ax.set_ylabel('SalePrice ($)')
            ax.set_title('Boxplot des prix')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

    with tab3:
        corr = maisondf.select_dtypes(include='number').corr()['SalePrice'].sort_values(ascending=False).drop('SalePrice').head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        corr.plot(kind='barh', ax=ax, color='steelblue', alpha=0.8)
        ax.set_xlabel('Correlation')
        ax.set_title('Top 10 correlations avec SalePrice')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.warning("Charger le fichier csv pour continuer.")
