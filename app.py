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
    time = data.iloc[:, 0].values / 1000 # data.iloc[:, 0].values / 1000   si Conversion en secondes
    amplitude = data.iloc[:, 1].values

    # Aperçu du dataset
    if st.checkbox("Afficher les 5 premières lignes du dataset"):
        st.write(data.head())

    # Affichage du signal original
    if st.checkbox("Afficher le signal original (time vs amplitude)"):
        fig, ax = plt.subplots()
        ax.plot(time*1000, amplitude)
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.grid()
        ax.set_title("Signal Original")
        st.pyplot(fig)

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
        fig, ax = plt.subplots()
        ax.plot(time, signal_filtre)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid()
        ax.set_title("Signal après traitement (passe-haut, redressement, passe-bas)")
        st.pyplot(fig)
    
    # Affichage du spectre FFT interactif
    if st.checkbox("Afficher le spectre FFT du signal après traitement BLSD"):
        n = len(signal_filtre)
        f_min = st.slider("Zoom sur fréquence minimale (Hz)", 1, n//2, 1)  # le 1 est la valeur par défaut
        f_limit = st.slider("Zoom sur fréquence maximale (Hz)", n//1000, n//2, 500)
        valeur_fft = fft(signal_filtre)
        frequencies = fftfreq(n, d=1/fs)[f_min:f_limit]
        fft_magnitudes = np.abs(valeur_fft)[f_min:f_limit]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frequencies, y=fft_magnitudes, mode='lines', name='Spectre FFT'))
        fig.update_layout(
            title='Spectre FFT du Signal après Traitement BLSD',
            xaxis_title='Fréquence (Hz)',
            yaxis_title='Amplitude',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig)
else:
    st.info("Veuillez importer un fichier CSV pour commencer l'analyse.")
