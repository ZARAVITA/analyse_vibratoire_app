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
4. **Analyse des fréquences caractéristiques** : Sélectionnez un roulement et la vitesse de rotation pour afficher les fréquences caractéristiques (FTF, BSF, BPFO, BPFI) et leurs harmoniques.
5. **Affichage des résultats** : Visualisez le signal après traitement et son spectre FFT avec les fréquences caractéristiques.

Ce projet a été réalisé par **M. A Angelico** et **ZARAVITA** dans le cadre de l'analyse vibratoire.
""")

# Chargement des données des roulements depuis GitHub
@st.cache_data
def load_bearing_data():
    url = "https://raw.githubusercontent.com/[votre_utilisateur]/[votre_repo]/main/Bearing%20data%20Base.xlsx"
    try:
        response = requests.get(url)
        bearing_data = pd.read_excel(BytesIO(response.content))
        return bearing_data
    except:
        st.warning("Impossible de charger les données des roulements depuis GitHub. Utilisation des données par défaut.")
        # Données par défaut si le chargement échoue
        return pd.DataFrame({
            'Manufacturer': ['AMI', 'AMI', 'AMI'],
            'Name': ['201', '202', '203'],
            'Number of Rollers': [8, 8, 8],
            'FTF': [0.383, 0.383, 0.383],
            'BSF': [2.025, 2.025, 2.025],
            'BPFO': [3.066, 3.066, 3.066],
            'BPFI': [4.934, 4.934, 4.934]
        })

bearing_db = load_bearing_data()

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
                        hovertemplate=f'H{i}: {freq_harm:.1f} Hz<br>Amplitude: {harmonic_amp:.2f}<extra></extra>'
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
    
    # Section pour la sélection du roulement et la vitesse de rotation
    st.sidebar.header("Paramètres du roulement")
    
    # Sélection du fabricant
    manufacturers = sorted(bearing_db['Manufacturer'].unique())
    selected_manufacturer = st.sidebar.selectbox("Fabricant", manufacturers)
    
    # Filtrage des noms de roulements en fonction du fabricant sélectionné
    names = sorted(bearing_db[bearing_db['Manufacturer'] == selected_manufacturer]['Name'].unique())
    selected_name = st.sidebar.selectbox("Modèle", names)
    
    # Filtrage du nombre de rouleaux en fonction du modèle sélectionné
    rollers = bearing_db[(bearing_db['Manufacturer'] == selected_manufacturer) & 
                         (bearing_db['Name'] == selected_name)]['Number of Rollers'].unique()
    selected_roller = st.sidebar.selectbox("Nombre de rouleaux", rollers)
    
    # Sélection de la vitesse de rotation
    running_speed = st.sidebar.number_input("Vitesse de rotation (RPM)", min_value=1, max_value=10000, value=1800)
    running_speed_hz = running_speed / 60  # Conversion en Hz
    
    # Récupération des coefficients du roulement sélectionné
    bearing_coeffs = bearing_db[(bearing_db['Manufacturer'] == selected_manufacturer) & 
                               (bearing_db['Name'] == selected_name) & 
                               (bearing_db['Number of Rollers'] == selected_roller)].iloc[0]
    
    # Calcul des fréquences caractéristiques
    ftf = bearing_coeffs['FTF'] * running_speed_hz
    bsf = bearing_coeffs['BSF'] * running_speed_hz
    bpfo = bearing_coeffs['BPFO'] * running_speed_hz
    bpfi = bearing_coeffs['BPFI'] * running_speed_hz
    
    # Affichage des fréquences caractéristiques
    st.sidebar.markdown("### Fréquences caractéristiques calculées")
    st.sidebar.write(f"FTF: {ftf:.2f} Hz")
    st.sidebar.write(f"BSF: {bsf:.2f} Hz")
    st.sidebar.write(f"BPFO: {bpfo:.2f} Hz")
    st.sidebar.write(f"BPFI: {bpfi:.2f} Hz")
    
    # Affichage du spectre FFT interactif avec les fréquences caractéristiques
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
        
        # Options d'affichage des fréquences caractéristiques
        st.subheader("Options d'affichage des fréquences caractéristiques")
        col1, col2, col3, col4 = st.columns(4)
        show_ftf = col1.checkbox("FTF", value=True)
        show_bsf = col2.checkbox("BSF", value=True)
        show_bpfo = col3.checkbox("BPFO", value=True)
        show_bpfi = col4.checkbox("BPFI", value=True)
        
        show_harmonics = st.checkbox("Afficher les harmoniques (jusqu'à 5ème)", value=True)
        
        # Couleurs pour les différentes fréquences
        colors = {
            'FTF': 'purple',
            'BSF': 'green',
            'BPFO': 'blue',
            'BPFI': 'orange'
        }
        
        # Dictionnaire pour stocker les données des fréquences
        freq_data = {
            'FTF': ftf,
            'BSF': bsf,
            'BPFO': bpfo,
            'BPFI': bpfi
        }
        
        # Affichage des fréquences caractéristiques et de leurs harmoniques
        freq_table_data = []
        
        for freq_name, freq_value in freq_data.items():
            if (freq_name == 'FTF' and not show_ftf) or \
               (freq_name == 'BSF' and not show_bsf) or \
               (freq_name == 'BPFO' and not show_bpfo) or \
               (freq_name == 'BPFI' and not show_bpfi):
                continue
            
            # Trouver l'index le plus proche de la fréquence caractéristique
            idx = np.abs(frequencies - freq_value).argmin()
            exact_freq = frequencies[idx]
            amp = fft_magnitudes[idx]
            
            # Ajouter la fréquence fondamentale
            fig.add_trace(go.Scatter(
                x=[exact_freq],
                y=[amp],
                mode='markers+text',
                marker=dict(size=10, color=colors[freq_name]),
                text=freq_name,
                textposition='top center',
                name=freq_name,
                hovertemplate=f'{freq_name}: {exact_freq:.1f} Hz<br>Amplitude: {amp:.2f}<extra></extra>'
            ))
            
            # Ajouter une annotation avec les coordonnées exactes
            fig.add_annotation(
                x=exact_freq,
                y=amp,
                text=f'{exact_freq:.1f} Hz',
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-30,
                font=dict(color=colors[freq_name])
            )
            
            # Stocker les données pour le tableau
            freq_table_data.append({
                'Type': freq_name,
                'Fréquence théorique': f'{freq_value:.2f} Hz',
                'Fréquence mesurée': f'{exact_freq:.1f} Hz',
                'Amplitude': f'{amp:.2f}'
            })
            
            # Ajouter les harmoniques si demandé
            if show_harmonics:
                for i in range(2, 6):  # Harmoniques 2 à 5
                    harmonic_freq = i * freq_value
                    idx_harm = np.abs(frequencies - harmonic_freq).argmin()
                    exact_harm_freq = frequencies[idx_harm]
                    amp_harm = fft_magnitudes[idx_harm]
                    
                    fig.add_trace(go.Scatter(
                        x=[exact_harm_freq],
                        y=[amp_harm],
                        mode='markers+text',
                        marker=dict(size=8, color=colors[freq_name]),
                        text=f'{freq_name} H{i}',
                        textposition='top center',
                        name=f'{freq_name} H{i}',
                        hovertemplate=f'{freq_name} H{i}: {exact_harm_freq:.1f} Hz<br>Amplitude: {amp_harm:.2f}<extra></extra>'
                    ))
                    
                    # Stocker les données des harmoniques pour le tableau
                    freq_table_data.append({
                        'Type': f'{freq_name} H{i}',
                        'Fréquence théorique': f'{harmonic_freq:.2f} Hz',
                        'Fréquence mesurée': f'{exact_harm_freq:.1f} Hz',
                        'Amplitude': f'{amp_harm:.2f}'
                    })
        
        # Afficher un tableau récapitulatif des fréquences
        if freq_table_data:
            st.write("Détails des fréquences caractéristiques:")
            freq_df = pd.DataFrame(freq_table_data)
            st.table(freq_df)
        
        # Configuration de l'interactivité
        fig.update_layout(
            title=f'Spectre FFT du Signal après Traitement BLSD - Vitesse: {running_speed} RPM',
            xaxis_title='Fréquence (Hz)',
            yaxis_title='Amplitude',
            hovermode='x unified',
            clickmode='event+select'
        )
        
        st.plotly_chart(fig, config={'displayModeBar': True})
else:
    st.info("Veuillez importer un fichier CSV pour commencer l'analyse.")
