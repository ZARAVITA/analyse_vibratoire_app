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
        selected_point = st.plotly_chart(
            fig_original, 
            config={'displayModeBar': True, 'scrollZoom': True},
            key='original_signal'
        )
        
        # Bouton pour réinitialiser la sélection
        if st.button('Réinitialiser la sélection pour le signal original'):
            st.session_state.selected_points = {'t1': None, 't2': None, 'fig': None}
            st.experimental_rerun()
        
        # Gestion de la sélection de points via un clic
        if st.session_state.get('click_data'):
            point_data = st.session_state.click_data['points'][0]
            x_clicked = point_data['x']
            
            if st.session_state.selected_points['fig'] != 'Signal Original':
                st.session_state.selected_points = {'t1': x_clicked, 't2': None, 'fig': 'Signal Original'}
            else:
                if st.session_state.selected_points['t1'] is None:
                    st.session_state.selected_points['t1'] = x_clicked
                else:
                    st.session_state.selected_points['t2'] = x_clicked
            
            # Réinitialiser les données de clic après traitement
            st.session_state.click_data = None
            st.experimental_rerun()
        
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
    
    # Affichage du signal après traitement avec sélection de points
    if st.checkbox("Afficher le signal après traitement BLSD(Bearing Low Speed Detection)"):
        fig_treated = create_signal_figure(
            time, signal_filtre, 
            'Signal après traitement (passe-haut, redressement, passe-bas)', 
            'Time (s)', 'Amplitude'
        )
        
        # Affichage du graphique
        selected_point_treated = st.plotly_chart(
            fig_treated, 
            config={'displayModeBar': True, 'scrollZoom': True},
            key='treated_signal'
        )
        
        # Bouton pour réinitialiser la sélection
        if st.button('Réinitialiser la sélection pour le signal traité'):
            st.session_state.selected_points = {'t1': None, 't2': None, 'fig': None}
            st.experimental_rerun()
        
        # Gestion de la sélection de points via un clic
        if st.session_state.get('click_data_treated'):
            point_data = st.session_state.click_data_treated['points'][0]
            x_clicked = point_data['x']
            
            if st.session_state.selected_points['fig'] != 'Signal traité':
                st.session_state.selected_points = {'t1': x_clicked, 't2': None, 'fig': 'Signal traité'}
            else:
                if st.session_state.selected_points['t1'] is None:
                    st.session_state.selected_points['t1'] = x_clicked
                else:
                    st.session_state.selected_points['t2'] = x_clicked
            
            # Réinitialiser les données de clic après traitement
            st.session_state.click_data_treated = None
            st.experimental_rerun()
    
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
        
        # Configuration de l'interactivité
        fig.update_layout(
            title='Spectre FFT du Signal après Traitement BLSD',
            xaxis_title='Fréquence (Hz)',
            yaxis_title='Amplitude',
            hovermode='x unified',
            clickmode='event+select'
        )
        
        st.plotly_chart(fig, config={'displayModeBar': True})
        
        # Section pour afficher les harmoniques avec leurs coordonnées
        st.subheader("Analyse des harmoniques")
        fundamental_freq = st.slider(
            "Sélectionnez la fréquence fondamentale pour afficher ses harmoniques", 
            min_value=float(f_min), 
            max_value=float(f_limit), 
            value=float(f_min),
            step=0.1
        )
        
        if fundamental_freq > 0:
            # Créer une nouvelle figure avec les harmoniques
            fig_harmonics = go.Figure()
            fig_harmonics.add_trace(go.Scatter(
                x=frequencies,
                y=fft_magnitudes,
                mode='lines',
                name='Spectre FFT',
                hovertemplate='Freq: %{x:.1f} Hz<br>Amplitude: %{y:.1f}<extra></extra>'
            ))
            
            # Ajouter les harmoniques avec annotations
            harmonics_data = []
            for i in range(1, 6):  # 5 harmoniques
                freq = i * fundamental_freq
                # Trouver l'index le plus proche de la fréquence harmonique
                idx = np.abs(frequencies - freq).argmin()
                harmonic_freq = frequencies[idx]
                harmonic_amp = fft_magnitudes[idx]
                
                # Stocker les données pour le tableau
                harmonics_data.append({
                    'Harmonique': f'H{i}',
                    'Fréquence (Hz)': f'{harmonic_freq:.1f}',
                    'Amplitude': f'{harmonic_amp:.1f}'
                })
                
                # Ajouter le point et l'annotation
                fig_harmonics.add_trace(go.Scatter(
                    x=[harmonic_freq],
                    y=[harmonic_amp],
                    mode='markers+text',
                    marker=dict(size=10, color='red'),
                    text=f'H{i}',
                    textposition='top center',
                    name=f'Harmonique {i}',
                    hovertemplate=f'H{i}<br>Fréquence: {harmonic_freq:.1f} Hz<br>Amplitude: {harmonic_amp:.1f}<extra></extra>'
                ))
                
                # Ajouter une annotation avec les coordonnées exactes
                fig_harmonics.add_annotation(
                    x=harmonic_freq,
                    y=harmonic_amp,
                    text=f'({harmonic_freq:.1f} Hz, {harmonic_amp:.1f})',
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-30,
                    font=dict(color='red')
                )
            
            # Mise à jour de la mise en page
            fig_harmonics.update_layout(
                title=f'Spectre FFT avec Harmoniques de {fundamental_freq:.1f} Hz',
                xaxis_title='Fréquence (Hz)',
                yaxis_title='Amplitude',
                hovermode='x unified',
                showlegend=False
            )
            
            st.plotly_chart(fig_harmonics)
            
            # Afficher un tableau récapitulatif des harmoniques
            st.write("Détails des harmoniques:")
            harmonics_df = pd.DataFrame(harmonics_data)
            st.table(harmonics_df)
else:
    st.info("Veuillez importer un fichier CSV pour commencer l'analyse.")
