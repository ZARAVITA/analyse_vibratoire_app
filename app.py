import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

# Titre de l'application
st.title("Analyse Vibratoire - Projet par Angelico et ZARAVITA")
st.markdown("""
Cette application permet d'analyser des signaux vibratoires en appliquant des filtres passe-haut, redressement, et filtre passe-bas.
Vous pouvez importer un fichier CSV contenant les colonnes "time" et "amplitude", visualiser les données, et appliquer les traitements.
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

    # Affichage du signal original
    if st.checkbox("Afficher le signal original (time vs amplitude)"):
        fig, ax = plt.subplots()
        ax.plot(time, amplitude)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid()
        ax.set_title("Signal Original")
        st.pyplot(fig)

    # Fréquence d'échantillonnage
    dt = np.diff(time)
    fs = 1 / np.mean(dt)
    st.write(f"Fréquence d'échantillonnage : {fs:.0f} Hz")

    # Filtre passe-haut
    st.sidebar.header("Paramètres du filtre passe-haut")
    freq_coupure_haut = st.sidebar.slider("Fréquence de coupure passe-haut (Hz)", min_value=1, max_value=1000, value=500)

    def filtre_pass_haut(data, freq_coupure_haut, freq_echantillonnage):
        freq_nyquist = 0.5 * freq_echantillonnage
        freq_normalisee_wn = freq_coupure_haut / freq_nyquist
        b, a = butter(4, freq_normalisee_wn, btype='high', analog=False)
        return filtfilt(b, a, data)

    filtre_haut = filtre_pass_haut(amplitude, freq_coupure_haut, fs)

    # Redressement
    def redressement(data):
        return np.abs(data)

    signal_redresse = redressement(filtre_haut)

    # Filtre passe-bas
    st.sidebar.header("Paramètres du filtre passe-bas")
    freq_coupure_bas = st.sidebar.slider("Fréquence de coupure passe-bas (Hz)", min_value=1, max_value=1000, value=200)

    def filtre_passe_bas(data, freq_coupure, freq_echantillonnage):
        freq_nyquist = 0.5 * freq_echantillonnage
        freq_normalisee = freq_coupure / freq_nyquist
        b, a = butter(4, freq_normalisee, btype='low', analog=False)
        return filtfilt(b, a, data)

    filtre_basse_du_signal_redresse = filtre_passe_bas(signal_redresse, freq_coupure_bas, fs)

    # Affichage du signal après traitement
    if st.checkbox("Afficher le signal après traitement"):
        fig, ax = plt.subplots()
        ax.plot(time, filtre_basse_du_signal_redresse)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid()
        ax.set_title("Signal après traitement (passe-haut, redressement, passe-bas)")
        st.pyplot(fig)

    # Spectre FFT du signal après traitement
    if st.checkbox("Afficher le spectre FFT du signal après traitement"):
        n = len(filtre_basse_du_signal_redresse)
        valeur_fft = fft(filtre_basse_du_signal_redresse / 10000)
        frequencies = fftfreq(n, d=1/fs)[:n//8]
        fft_magnitudes = np.abs(valeur_fft)[:n//8]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frequencies, fft_magnitudes, label='Spectre FFT')
        ax.set_title('Spectre FFT du Signal après Traitement')
        ax.set_xlabel('Fréquence (Hz)')
        ax.set_ylabel('Amplitude')
        ax.grid()
        ax.legend()
        st.pyplot(fig)
else:
    st.info("Veuillez importer un fichier CSV pour commencer l'analyse.")
