import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# --- Titre ---
st.title("💳 Détection de fraudes bancaires")
st.write("Application de détection d'anomalies avec Isolation Forest")

# --- Upload du fichier ---
uploaded_file = st.file_uploader("📂 Uploadez votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    # Lecture des données
    df = pd.read_csv(uploaded_file)

    st.subheader("Aperçu des données")
    st.write(df.head())

    # Prétraitement (comme dans ton notebook)
    
    df['Amount_scaled'] = StandardScaler().fit_transform(df[['Amount']])
    df['Time_scaled'] = StandardScaler().fit_transform(df[['Time']])
    features = df.drop(columns=['Time', 'Amount', 'Class'], errors='ignore')

    # --- Modélisation ---
    contamination_rate = st.slider("Taux d'anomalies estimé", 0.0001, 0.01, 0.0017)
    model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42)
    model.fit(features)
    df['Anomaly'] = model.predict(features)
    df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})  # 1 = anomalie, 0 = normal

    st.subheader("Résultats de la détection")
    st.write(df['Anomaly'].value_counts())

    # --- Graphique interactif ---
    if 'PCA1' not in df.columns:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        df['PCA1'], df['PCA2'] = pca_result[:,0], pca_result[:,1]

    fig = px.scatter(
        df, x='PCA1', y='PCA2',
        color=df['Anomaly'].map({0: "Normal", 1: "Anomalie"}),
        title="Visualisation des anomalies détectées"
    )
    st.plotly_chart(fig)

    # --- Export ---
    st.download_button(
        label="📥 Télécharger les résultats",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="transactions_avec_anomalies.csv",
        mime='text/csv'
    )
else:
    st.info("Veuillez uploader un fichier CSV pour commencer.")

