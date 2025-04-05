# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:13:56 2025

@author: ZARAVITA Haydar
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

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

    # Affichage du signal original
    if st.checkbox("Afficher le signal original (time vs amplitude)"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time,
            y=amplitude,
            mode='lines',
            name='Signal original',
            hovertemplate='Time: %{x:.1f} s<br>Amplitude: %{y:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Signal Original',
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            hovermode='x unified',
            clickmode='event+select'
        )
        
        # Configuration pour la sélection de points
        fig.update_layout(
            dragmode='select',
            selectdirection='h'
        )
        
        st.plotly_chart(fig)
        
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
    
    # Affichage du signal après traitement
    if st.checkbox("Afficher le signal après traitement BLSD(Bearing Low Speed Detection)"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time,
            y=signal_filtre,
            mode='lines',
            name='Signal traité',
            hovertemplate='Time: %{x:.1f} s<br>Amplitude: %{y:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Signal après traitement (passe-haut, redressement, passe-bas)',
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            hovermode='x unified',
            clickmode='event+select'
        )
        
        # Configuration pour la sélection de points
        fig.update_layout(
            dragmode='select',
            selectdirection='h'
        )
        
        st.plotly_chart(fig)
    
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
        
        # Variables pour stocker les clics
        clicked_points = []
        
        # Fonction pour ajouter les harmoniques
        def add_harmonics(fig, fundamental_freq):
            for i in range(1, 6):  # 5 harmoniques
                freq = i * fundamental_freq
                idx = np.abs(frequencies - freq).argmin()
                fig.add_trace(go.Scatter(
                    x=[frequencies[idx]],
                    y=[fft_magnitudes[idx]],
                    mode='markers+text',
                    marker=dict(size=10, color='red'),
                    text=f'H{i}',
                    textposition='top center',
                    name=f'Harmonique {i}',
                    hovertemplate=f'H{i}: {frequencies[idx]:.1f} Hz<br>Amplitude: {fft_magnitudes[idx]:.1f}<extra></extra>'
                ))
        
        # Configuration de l'interactivité
        fig.update_layout(
            title='Spectre FFT du Signal après Traitement BLSD',
            xaxis_title='Fréquence (Hz)',
            yaxis_title='Amplitude',
            hovermode='x unified',
            clickmode='event+select'
        )
        
        # Configuration pour la sélection de points
        fig.update_layout(
            dragmode='select',
            selectdirection='h'
        )
        
        # Affichage du graphique
        st.plotly_chart(fig, config={'displayModeBar': True})
        
        # Gestion des événements de clic
        if st.checkbox("Activer la sélection d'harmoniques (double-cliquez sur le graphique)"):
            st.info("Double-cliquez sur une fréquence fondamentale pour afficher ses harmoniques")
            
            # Cette partie nécessiterait normalement une intégration avec JavaScript
            # Pour Streamlit, nous utilisons une approche simplifiée avec des sliders
            fundamental_freq = st.slider("Sélectionnez la fréquence fondamentale pour afficher ses harmoniques", 
                                       min_value=float(f_min), 
                                       max_value=float(f_limit), 
                                       value=float(f_min))
            
            # Recréer la figure avec les harmoniques
            fig_with_harmonics = go.Figure()
            fig_with_harmonics.add_trace(go.Scatter(
                x=frequencies,
                y=fft_magnitudes,
                mode='lines',
                name='Spectre FFT',
                hovertemplate='Freq: %{x:.1f} Hz<br>Amplitude: %{y:.1f}<extra></extra>'
            ))
            
            add_harmonics(fig_with_harmonics, fundamental_freq)
            
            # Mise à jour de la mise en page
            fig_with_harmonics.update_layout(
                title='Spectre FFT avec Harmoniques',
                xaxis_title='Fréquence (Hz)',
                yaxis_title='Amplitude',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_with_harmonics)
        
        # Calcul de fréquence entre deux points sélectionnés
        st.subheader("Mesure de fréquence entre deux points")
        col1, col2 = st.columns(2)
        with col1:
            t1 = st.number_input("Temps du premier point (s)", value=0.0, step=0.1)
        with col2:
            t2 = st.number_input("Temps du deuxième point (s)", value=0.0, step=0.1)
        
        if t1 != 0 and t2 != 0 and t1 != t2:
            delta_t = abs(t1 - t2)
            frequency = 1 / delta_t
            st.write(f"Fréquence calculée: {frequency:.1f} Hz")
else:
    st.info("Veuillez importer un fichier CSV pour commencer l'analyse.")
