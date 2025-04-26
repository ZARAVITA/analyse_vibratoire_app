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
#df=pd.read_excel("Bearing data Base.xlsx")          # ----------------------------------------------------------
#print(len(df))                                      #-----------------------------------------------------------

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

# Vitesse de rotation en Hz
rotation_speed_hz = st.sidebar.number_input("Vitesse de rotation (Hz)", 
                                         min_value=0.1, 
                                         max_value=1000.0, 
                                         value=16.67,  # 1000 RPM ≈ 16.67 Hz
                                         step=0.1,
                                         format="%.2f")

# Calcul des fréquences caractéristiques
selected_bearing = bearing_data[
    (bearing_data['Manufacturer'] == selected_manufacturer) & 
    (bearing_data['Name'] == selected_model)
].iloc[0]

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
harmonics_count = st.sidebar.slider("Nombre d'harmoniques à afficher", 1, 5, 3) if show_harmonics else 0

# Nouvelle option pour les harmoniques de vitesse
st.sidebar.subheader("Options des harmoniques de vitesse")
show_speed_harmonics = st.sidebar.checkbox("Afficher les harmoniques de vitesse", False)
speed_harmonics_count = st.sidebar.slider("Nombre d'harmoniques de vitesse", 1, 5, 3) if show_speed_harmonics else 0
speed_harmonics_color = st.sidebar.color_picker("Couleur des harmoniques de vitesse", "#FFA500")  # Orange par défaut

# Upload du fichier CSV
uploaded_file = st.file_uploader("Importez votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    # Lecture du fichier CSV
    data = pd.read_csv(uploaded_file, sep=";", skiprows=1)
    # Conversion des colonnes en numpy arrays
    time = data.iloc[:, 0].values / 1000  # Conversion en secondes
    amplitude = data.iloc[:, 1].values

    # Aperçu du dataset
    if st.checkbox("Afficher les 5 premières lignes du dataset"):
        st.write(data.head())

    # Fonction pour calculer la FFT
    def calculate_fft(signal, fs):
        n = len(signal)
        yf = fft(signal)
        xf = fftfreq(n, 1/fs)[:n//2]
        return xf, 2.0/n * np.abs(yf[0:n//2])

    # Variables globales pour stocker les points sélectionnés
    if 'selected_points' not in st.session_state:
        st.session_state.selected_points = {'t1': None, 't2': None, 'fig': None}

    # Fonction pour créer un graphique avec sélection de points
    def create_signal_figure(x, y, title, x_title, y_title):
        fig = go.Figure()
        
        # Ajout du signal
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Signal',
            hovertemplate='Time: %{x:.3f} s<br>Amplitude: %{y:.1f}<extra></extra>'
        ))
        
        # Ajout des marqueurs si des points sont sélectionnés
        if st.session_state.selected_points['fig'] == title:
            t1 = st.session_state.selected_points['t1']
            t2 = st.session_state.selected_points['t2']
            
            if t1 is not None:
                # Trouver l'index le plus proche de t1
                idx1 = np.abs(x - t1).argmin()
                y1 = y[idx1]
                
                # Ajouter marqueur et ligne verticale pour t1
                fig.add_trace(go.Scatter(
                    x=[t1],
                    y=[y1],
                    mode='markers+text',
                    marker=dict(size=12, color='red'),
                    text='T1',
                    textposition='top center',
                    name='Point 1',
                    hoverinfo='none'
                ))
                
                fig.add_shape(
                    type='line',
                    x0=t1, y0=min(y), x1=t1, y1=max(y),
                    line=dict(color='red', width=2, dash='dot')
                )
                
            if t2 is not None:
                # Trouver l'index le plus proche de t2
                idx2 = np.abs(x - t2).argmin()
                y2 = y[idx2]
                
                # Ajouter marqueur et ligne verticale pour t2
                fig.add_trace(go.Scatter(
                    x=[t2],
                    y=[y2],
                    mode='markers+text',
                    marker=dict(size=12, color='blue'),
                    text='T2',
                    textposition='top center',
                    name='Point 2',
                    hoverinfo='none'
                ))
                
                fig.add_shape(
                    type='line',
                    x0=t2, y0=min(y), x1=t2, y1=max(y),
                    line=dict(color='blue', width=2, dash='dot')
                )
            
            # Ajouter une ligne horizontale entre les deux points si les deux sont sélectionnés
            if t1 is not None and t2 is not None:
                fig.add_shape(
                    type='line',
                    x0=t1, y0=(y1+y2)/2, x1=t2, y1=(y1+y2)/2,
                    line=dict(color='green', width=2, dash='dash'),
                    name='Intervalle'
                )
                
                # Calculer et afficher la fréquence
                delta_t = abs(t2 - t1)
                if delta_t > 0:
                    frequency = 1 / delta_t
                    fig.add_annotation(
                        x=(t1 + t2)/2,
                        y=(y1 + y2)/2,
                        text=f'F = {frequency:.2f} Hz',
                        showarrow=True,
                        arrowhead=1,
                        font=dict(size=14, color='green'),
                        bgcolor='white',
                        bordercolor='green',
                        borderwidth=1
                    )
        
        # Mise en page
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,
            hovermode='x unified',
            clickmode='event+select',
            dragmode='zoom'
        )
        
        return fig

    # Affichage du signal original avec sélection de points
    if st.checkbox("Afficher le signal original (time vs amplitude)"):
        fig_original = create_signal_figure(
            time, amplitude, 
            'Signal Original', 
            'Time (s)', 'Amplitude'
        )
        
        # Affichage du graphique
        st.plotly_chart(
            fig_original, 
            config={'displayModeBar': True, 'scrollZoom': True},
            key='original_signal'
        )
        
        # Boutons pour sélection manuelle des points
        col1, col2 = st.columns(2)
        with col1:
            t1 = st.number_input("Sélectionnez T1 (s)", min_value=float(time.min()), 
                                max_value=float(time.max()), value=0.0, step=0.001)
        with col2:
            t2 = st.number_input("Sélectionnez T2 (s)", min_value=float(time.min()), 
                                max_value=float(time.max()), value=0.0, step=0.001)
        
        if t1 != 0 or t2 != 0:
            st.session_state.selected_points = {'t1': t1, 't2': t2, 'fig': 'Signal Original'}
            if t1 != 0 and t2 != 0 and t1 != t2:
                delta_t = abs(t2 - t1)
                frequency = 1 / delta_t
                st.success(f"Fréquence calculée: {frequency:.2f} Hz")
        
        # Calcul et affichage de la FFT du signal original
        st.subheader("FFT du signal original")
        xf, yf = calculate_fft(amplitude, 1/np.mean(np.diff(time)))
        
        fft_fig = go.Figure()
        fft_fig.add_trace(go.Scatter(
            x=xf,
            y=yf,
            mode='lines',
            name='FFT originale',
            hovertemplate='Freq: %{x:.1f} Hz<br>Amplitude: %{y:.1f}<extra></extra>'
        ))
        
        # Ajout des harmoniques si une fréquence est sélectionnée
        if st.checkbox("Afficher les harmoniques sur la FFT originale"):
            freq_fond = st.number_input("Fréquence fondamentale (Hz)", 
                                      min_value=0.1, 
                                      max_value=float(xf.max()), 
                                      value=10.0, step=0.1)
            
            if freq_fond > 0:
                for i in range(1, 6):  # 5 harmoniques
                    freq = i * freq_fond
                    idx = np.abs(xf - freq).argmin()
                    freq_harm = xf[idx]
                    amp_harm = yf[idx]
                    
                    fft_fig.add_trace(go.Scatter(
                        x=[freq_harm],
                        y=[amp_harm],
                        mode='markers+text',
                        marker=dict(size=10, color='red'),
                        text=f'H{i}',
                        textposition='top center',
                        name=f'Harmonique {i}',
                        hovertemplate=f'H{i}: {freq_harm:.1f} Hz<br>Amplitude: {amp_harm:.2f}<extra></extra>'
                    ))
                    
                    fft_fig.add_annotation(
                        x=freq_harm,
                        y=amp_harm,
                        text=f'({freq_harm:.1f} Hz, {amp_harm:.2f})',
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-30,
                        font=dict(color='red')
                    )
        
        fft_fig.update_layout(
            title='Spectre FFT du Signal Original',
            xaxis_title='Fréquence (Hz)',
            yaxis_title='Amplitude',
            hovermode='x unified',
            clickmode='event+select'
        )
        
        st.plotly_chart(fft_fig)

    # Fréquence d'échantillonnage
    dt = np.diff(time)
    fs = 1 / np.mean(dt)
    st.write(f"Fréquence d'échantillonnage : {fs:.0f} Hz")

    # Filtre passe-haut
    freq_coupure_haut = st.sidebar.slider("Fréquence de coupure passe-haut (Hz)", 1, 5000, 500)
    def filtre_pass_haut(data, freq_coupure, freq_echantillonnage):
        freq_nyquist = 0.5 * freq_echantillonnage
        freq_normalisee = freq_coupure / freq_nyquist
        b, a = butter(4, freq_normalisee, btype='high', analog=False)
        return filtfilt(b, a, data)
    
    filtre_haut = filtre_pass_haut(amplitude, freq_coupure_haut, fs)
    
    # Redressement
    signal_redresse = np.abs(filtre_haut)
    
    # Filtre passe-bas
    freq_coupure_bas = st.sidebar.slider("Fréquence de coupure passe-bas (Hz)", 1, 1000, 200)
    def filtre_passe_bas(data, freq_coupure, freq_echantillonnage):
        freq_nyquist = 0.5 * freq_echantillonnage
        freq_normalisee = freq_coupure / freq_nyquist
        b, a = butter(4, freq_normalisee, btype='low', analog=False)
        return filtfilt(b, a, data)
    
    signal_filtre = filtre_passe_bas(signal_redresse, freq_coupure_bas, fs)
    
    # Affichage du signal après traitement avec sélection de points
    if st.checkbox("Afficher le signal après traitement BLSD(Bearing Low Speed Detection)"):
        fig_treated = create_signal_figure(
            time, signal_filtre, 
            'Signal après traitement (passe-haut, redressement, passe-bas)', 
            'Time (s)', 'Amplitude'
        )
        
        # Affichage du graphique
        st.plotly_chart(
            fig_treated, 
            config={'displayModeBar': True, 'scrollZoom': True},
            key='treated_signal'
        )
        
        # Boutons pour sélection manuelle des points
        col1, col2 = st.columns(2)
        with col1:
            t1_treated = st.number_input("Sélectionnez T1 (s) - Signal traité", 
                                       min_value=float(time.min()), 
                                       max_value=float(time.max()), 
                                       value=0.0, step=0.001)
        with col2:
            t2_treated = st.number_input("Sélectionnez T2 (s) - Signal traité", 
                                       min_value=float(time.min()), 
                                       max_value=float(time.max()), 
                                       value=0.0, step=0.001)
        
        if t1_treated != 0 or t2_treated != 0:
            st.session_state.selected_points = {'t1': t1_treated, 't2': t2_treated, 'fig': 'Signal traité'}
            if t1_treated != 0 and t2_treated != 0 and t1_treated != t2_treated:
                delta_t = abs(t2_treated - t1_treated)
                frequency = 1 / delta_t
                st.success(f"Fréquence calculée: {frequency:.2f} Hz")

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
            'BPFI': 'red',
            'Vitesse': speed_harmonics_color
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
                
                # Ajout d'une ligne verticale pour la fréquence fondamentale
                fig.add_shape(
                    type='line',
                    x0=measured_freq, y0=0,
                    x1=measured_freq, y1=measured_amp,
                    line=dict(color=freq_colors[freq_type], width=1, dash='dot')
                )
                
                # Ajout des harmoniques si activé
                if show_harmonics:
                    harmonics_row = {f'Harmonique': freq_type}
                    for i in range(1, harmonics_count + 1):  # Harmoniques jusqu'au nombre sélectionné
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
                        
                        # Ajout d'une ligne verticale pour l'harmonique
                        fig.add_shape(
                            type='line',
                            x0=harmonic_freq_measured, y0=0,
                            x1=harmonic_freq_measured, y1=harmonic_amp,
                            line=dict(color=freq_colors[freq_type], width=1, dash='dot')
                        )
                        
                        # Ajout d'une annotation avec la valeur exacte
                        fig.add_annotation(
                            x=harmonic_freq_measured,
                            y=harmonic_amp,
                            text=f'{harmonic_freq_measured:.1f} Hz',
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-30,
                            font=dict(color=freq_colors[freq_type], size=10)
                        )
                        
                        # Stockage des données pour le tableau
                        harmonics_row[f'{i}× Fréquence (Hz)'] = f'{harmonic_freq_measured:.1f}'
                        harmonics_row[f'{i}× Amplitude'] = f'{harmonic_amp:.2f}'
                    
                    summary_data.append(harmonics_row)
        
        # Ajout des harmoniques de vitesse si activé
        if show_speed_harmonics:
            speed_harmonics_row = {'Harmonique': 'Vitesse'}
            for i in range(1, speed_harmonics_count + 1):
                speed_harmonic = i * rotation_speed_hz
                idx_speed = np.abs(frequencies - speed_harmonic).argmin()
                speed_harmonic_measured = frequencies[idx_speed]
                speed_amp = fft_magnitudes[idx_speed]
                
                # Ajout au graphique
                fig.add_trace(go.Scatter(
                    x=[speed_harmonic_measured],
                    y=[speed_amp],
                    mode='markers+text',
                    marker=dict(size=10, color=freq_colors['Vitesse']),
                    text=f'V {i}×',  # Format court "V 1×", "V 2×", etc.
                    textposition='top center',
                    name=f'Vitesse {i}×',
                    hovertemplate=f'Vitesse {i}×<br>Fréquence: {speed_harmonic_measured:.1f} Hz<br>Amplitude: {speed_amp:.2f}<extra></extra>'
                ))
                
                # Ajout d'une ligne verticale pointillée orange
                fig.add_shape(
                    type='line',
                    x0=speed_harmonic_measured, y0=0,
                    x1=speed_harmonic_measured, y1=speed_amp,
                    line=dict(color=freq_colors['Vitesse'], width=1, dash='dot')
                )
                
                # Ajout d'une annotation avec la valeur exacte
                fig.add_annotation(
                    x=speed_harmonic_measured,
                    y=speed_amp,
                    text=f'{speed_harmonic_measured:.1f} Hz',
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-30,
                    font=dict(color=freq_colors['Vitesse'], size=10)
                )
                
                # Stockage des données pour le tableau
                speed_harmonics_row[f'{i}× Fréquence (Hz)'] = f'{speed_harmonic_measured:.1f}'
                speed_harmonics_row[f'{i}× Amplitude'] = f'{speed_amp:.2f}'
            
            summary_data.append(speed_harmonics_row)
        
        # Affichage du tableau récapitulatif
        if summary_data:
            st.subheader("Tableau récapitulatif des fréquences")
            summary_df = pd.DataFrame(summary_data)
            
            # Réorganisation des colonnes
            columns_order = ['Harmonique']
            max_harmonics = max(harmonics_count, speed_harmonics_count) if show_speed_harmonics else harmonics_count
            for i in range(1, max_harmonics + 1):
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
