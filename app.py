# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:13:56 2025

@author: ZARAVITA Haydar
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import requests
from io import BytesIO

# Titre de l'application
st.markdown("""
Cette application est conçue pour l'analyse vibratoire de signaux. Elle permet :
1. **Importation de données** : Chargez un fichier CSV contenant les colonnes "time" et "amplitude".
2. **Visualisation des données** : Affichez les premières lignes du dataset et le signal original.
3. **Traitement BLSD (Bearing Low Speed Detection) du signal** :
   - Filtre passe-haut (fréquence de coupure réglable).
   - Redressement du signal.
   - Filtre passe-bas (fréquence de coupure réglable).
4. **Affichage des résultats** : Visualisez le signal après traitement et son spectre FFT.

Ce projet a été réalisé par **M. A Angelico** et **ZARAVITA** dans le cadre de l'analyse vibratoire.
""")

# Chargement des données des roulements depuis GitHub
@st.cache_data
def load_bearing_data():
    url = "https://github.com/ZARAVITA/analyse_vibratoire_app/raw/main/Bearing%20data%20Base.xlsx"
    try:
        response = requests.get(url)
        response.raise_for_status()
        bearing_data = pd.read_excel(BytesIO(response.content))
        return bearing_data
    except:
        # Données par défaut si le chargement échoue
        default_data = {
            'Manufacturer': ['AMI', 'AMI', 'DODGE', 'DODGE', 'FAFNIR', 'FAFNIR', 'KOYO', 'KOYO', 
                            'SEALMASTER', 'SKF', 'SKF', 'SNR', 'SNR', 'TORRINGTON', 'TORRINGTON'],
            'Name': ['201', '202', 'P2B5__USAF115TTAH (B)', 'P2B5__USAF115TTAH (C)', '206NPP', 
                    '206NPPA1849', '7304B (B)', '7304B (C)', '204', '214 (A)', '205 (A)', 
                    '6316ZZ (B)', 'NU324', '23172B', '23172BW33C08BR'],
            'Number of Rollers': [8, 8, 18, 17, 9, 9, 9, 9, 21, 15, 13, 8, 13, 22, 22],
            'FTF': [0.383, 0.383, 0.42, 0.57, 0.39, 0.39, 0.38, 0.38, 0.4404, 0.41, 0.42, 
                   0.38, 0.4, 0.44, 0.44],
            'BSF': [2.025, 2.025, 3.22, 6.49, 2.31, 2.31, 1.79, 1.79, 7.296, 2.7, 2.36, 
                   2.07, 2.42, 4.16, 4.16],
            'BPFO': [3.066, 3.066, 7.65, 7.24, 3.56, 3.56, 3.47, 3.46, 9.2496, 6.15, 5.47, 
                    3.08, 5.21, 9.71, 9.71],
            'BPFI': [4.934, 4.934, 10.34, 9.75, 5.43, 5.43, 5.53, 5.53, 11.7504, 8.84, 7.52, 
                    4.91, 7.78, 12.28, 12.28]
        }
        return pd.DataFrame(default_data)

bearing_data = load_bearing_data()

# Sidebar - Sélection du roulement
st.sidebar.header("Paramètres du roulement")

# Sélection du fabricant
manufacturers = sorted(bearing_data['Manufacturer'].unique())
selected_manufacturer = st.sidebar.selectbox("Fabricant", manufacturers)

# Filtrage des modèles en fonction du fabricant sélectionné
models = bearing_data[bearing_data['Manufacturer'] == selected_manufacturer]['Name'].unique()
selected_model = st.sidebar.selectbox("Modèle", models)

# Filtrage du nombre de rouleaux en fonction du modèle sélectionné
selected_roller_count = bearing_data[
    (bearing_data['Manufacturer'] == selected_manufacturer) & 
    (bearing_data['Name'] == selected_model)
]['Number of Rollers'].values[0]

st.sidebar.info(f"Nombre de rouleaux: {selected_roller_count}")

# Vitesse de rotation en RPM
rotation_speed_rpm = st.sidebar.number_input("Vitesse de rotation (RPM)", 
                                           min_value=1, 
                                           max_value=10000, 
                                           value=1000)

# Calcul des fréquences caractéristiques
selected_bearing = bearing_data[
    (bearing_data['Manufacturer'] == selected_manufacturer) & 
    (bearing_data['Name'] == selected_model)
].iloc[0]

rotation_speed_hz = rotation_speed_rpm / 60  # Conversion RPM -> Hz

# Calcul des fréquences caractéristiques
ftf_freq = selected_bearing['FTF'] * rotation_speed_hz
bsf_freq = selected_bearing['BSF'] * rotation_speed_hz
bpfo_freq = selected_bearing['BPFO'] * rotation_speed_hz
bpfi_freq = selected_bearing['BPFI'] * rotation_speed_hz

# Affichage des fréquences calculées
st.sidebar.subheader("Fréquences caractéristiques")
st.sidebar.write(f"- FTF: {ftf_freq:.2f} Hz")
st.sidebar.write(f"- BSF: {bsf_freq:.2f} Hz")
st.sidebar.write(f"- BPFO: {bpfo_freq:.2f} Hz")
st.sidebar.write(f"- BPFI: {bpfi_freq:.2f} Hz")

# Options d'affichage des fréquences
st.sidebar.subheader("Options d'affichage")
show_ftf = st.sidebar.checkbox("Afficher FTF", True)
show_bsf = st.sidebar.checkbox("Afficher BSF", True)
show_bpfo = st.sidebar.checkbox("Afficher BPFO", True)
show_bpfi = st.sidebar.checkbox("Afficher BPFI", True)
show_harmonics = st.sidebar.checkbox("Afficher les harmoniques", True)

# [...] (Le reste de votre code existant jusqu'à la partie FFT après traitement)

    # Affichage du spectre FFT interactif
    if st.checkbox("Afficher le spectre FFT du signal après traitement BLSD"):
        n = len(signal_filtre)
        f_min = st.slider("Zoom sur fréquence minimale (Hz)", 1, n//2, 1)
        f_limit = st.slider("Zoom sur fréquence maximale (Hz)", n//1000, n//2, 500)
        valeur_fft = fft(signal_filtre)
        frequencies = fftfreq(n, d=1/fs)[:n//2][f_min:f_limit]
        fft_magnitudes = 2.0/n * np.abs(valeur_fft[0:n//2])[f_min:f_limit]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frequencies,
            y=fft_magnitudes,
            mode='lines',
            name='Spectre FFT',
            hovertemplate='Freq: %{x:.1f} Hz<br>Amplitude: %{y:.1f}<extra></extra>'
        ))
        
        # Couleurs pour les fréquences caractéristiques
        freq_colors = {
            'FTF': 'violet',
            'BSF': 'green',
            'BPFO': 'blue',
            'BPFI': 'red'
        }
        
        # Dictionnaire pour stocker les données du tableau récapitulatif
        summary_data = []
        
        # Ajout des fréquences caractéristiques et de leurs harmoniques
        for freq_type, show, freq in [('FTF', show_ftf, ftf_freq),
                                     ('BSF', show_bsf, bsf_freq),
                                     ('BPFO', show_bpfo, bpfo_freq),
                                     ('BPFI', show_bpfi, bpfi_freq)]:
            if show:
                # Fréquence fondamentale
                idx = np.abs(frequencies - freq).argmin()
                measured_freq = frequencies[idx]
                measured_amp = fft_magnitudes[idx]
                
                # Ajout au graphique
                fig.add_trace(go.Scatter(
                    x=[measured_freq],
                    y=[measured_amp],
                    mode='markers+text',
                    marker=dict(size=10, color=freq_colors[freq_type]),
                    text=freq_type,
                    textposition='top center',
                    name=freq_type,
                    hovertemplate=f'{freq_type}<br>Fréquence: {measured_freq:.1f} Hz<br>Amplitude: {measured_amp:.2f}<extra></extra>'
                ))
                
                # Ajout des harmoniques si activé
                if show_harmonics:
                    harmonics_row = {f'Harmonique': freq_type}
                    for i in range(1, 6):  # 5 harmoniques
                        harmonic_freq = i * freq
                        idx_harm = np.abs(frequencies - harmonic_freq).argmin()
                        harmonic_freq_measured = frequencies[idx_harm]
                        harmonic_amp = fft_magnitudes[idx_harm]
                        
                        # Ajout au graphique
                        fig.add_trace(go.Scatter(
                            x=[harmonic_freq_measured],
                            y=[harmonic_amp],
                            mode='markers+text',
                            marker=dict(size=8, color=freq_colors[freq_type]),
                            text=f'{i}×',
                            textposition='top center',
                            name=f'{freq_type} {i}×',
                            hovertemplate=f'{freq_type} {i}×<br>Fréquence: {harmonic_freq_measured:.1f} Hz<br>Amplitude: {harmonic_amp:.2f}<extra></extra>'
                        ))
                        
                        # Ajout d'une ligne verticale
                        fig.add_shape(
                            type='line',
                            x0=harmonic_freq_measured, y0=0,
                            x1=harmonic_freq_measured, y1=harmonic_amp,
                            line=dict(color=freq_colors[freq_type], width=1, dash='dot')
                        )
                        
                        # Stockage des données pour le tableau
                        harmonics_row[f'{i}× Fréquence (Hz)'] = f'{harmonic_freq_measured:.1f}'
                        harmonics_row[f'{i}× Amplitude'] = f'{harmonic_amp:.2f}'
                    
                    summary_data.append(harmonics_row)
        
        # Affichage du tableau récapitulatif
        if summary_data:
            st.subheader("Tableau récapitulatif des fréquences")
            summary_df = pd.DataFrame(summary_data)
            
            # Réorganisation des colonnes
            columns_order = ['Harmonique']
            for i in range(1, 6):
                columns_order.extend([f'{i}× Fréquence (Hz)', f'{i}× Amplitude'])
            
            summary_df = summary_df[columns_order]
            st.dataframe(summary_df)
            
            # Option de téléchargement
            csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger le tableau récapitulatif",
                data=csv,
                file_name='frequences_caracteristiques.csv',
                mime='text/csv'
            )
        
        # Configuration de l'interactivité
        fig.update_layout(
            title='Spectre FFT du Signal après Traitement BLSD avec Fréquences Caractéristiques',
            xaxis_title='Fréquence (Hz)',
            yaxis_title='Amplitude',
            hovermode='x unified',
            clickmode='event+select'
        )
        
        st.plotly_chart(fig, config={'displayModeBar': True})
else:
    st.info("Veuillez importer un fichier CSV pour commencer l'analyse.")
