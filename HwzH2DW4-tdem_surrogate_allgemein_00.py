# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:31:46 2026

@author: tonih
"""

# =============================================================================
# TDEM Surrogate-Modell - Code-Vorlagen (korrigierte Version)
# =============================================================================
# Dieses File enthaelt Vorlagen fuer alle drei Projektphasen:
#   Phase 1: Trainingsdatengenerierung mit SimPEG
#   Phase 2: Surrogate-Modell-Training mit PyTorch
#   Phase 3: Inversion mit Surrogate
#
# Abhaengigkeiten:
#   pip install simpeg torch numpy scipy matplotlib
#
# Hardware: RTX 4070 Mobile (8 GB VRAM) oder Cloud-GPU
#
# Korrigierte Version mit folgenden Fixes:
#   - Klassenname TDEMSurrogateMLP (kein Leerzeichen)
#   - CircularLoop statt MagDipole fuer Simulation1DLayered
#   - use_inductive-Parameter entfernt (nicht in SimPEG API)
#   - Feste Schichtanzahl fuer einheitliche Modellgroesse
#   - Log-Parameterisierung in Inversion verhindert negative Leitfaehigkeit
#   - kernel_size als int oder tuple in ConvLSTMCell
# =============================================================================

import multiprocessing

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from simpeg import maps
import simpeg.electromagnetics.time_domain as tdem

# --- Sensor-Konfiguration (Empfaengerrauschmodell) ---
# FGM3D/75: 20 pT/sqrt(Hz). Cut-off-Frequenz: Standardversion 2 kHz,
# verwendete Geraeteversion 4 kHz. Wirkt auf die Fruehzeit-Kanaele
# (hoehere Bandbreite = mehr Rauschen, aber auch schnelleres Signal).
SENSOR_BW_HZ = 4000.0

# --- Turn-off-Konfiguration (Totzeit-Schaetzung) ---
# Ramp-off der Sendeschleife ueber TVS + Parallelwiderstand: Beim Oeffnen
# des Schalters klemmt die TVS die induktive Spitze (~300 V), danach klingt
# der Schleifenstrom exponentiell ueber den Parallelwiderstand ab.
# Kanaele frueher als t_excl duerfen nicht invertiert werden (das Surrogate
# ist auf idealen Step-off trainiert).
LOOP_WIRE_RADIUS_M = 0.001        # Drahtradius der Sendeschleife (m)
LOOP_N_TURNS = 1                  # Windungszahl der Sendeschleife
TVS_CLAMP_VOLTAGE = 300.0         # Klemmspannung der TVS (V)
R_PARALLEL_OHM = 2200.0           # Parallelwiderstand zur Schleife (Ohm, 2k2)
COPPER_RESISTIVITY_OHM_M = 1.68e-8
MU_0 = 4.0e-7 * np.pi


# =============================================================================
# KONFIGURATION - ALLE PARAMETER ZENTRAL HIER ANPASSEN
# =============================================================================
# DEV_MODE = True  -> Parameter aus DEV_CONFIG verwenden
# DEV_MODE = False -> Parameter aus FULL_CONFIG verwenden
#
# Geschaetzte Laufzeiten auf RTX 4070 Mobile (8 GB):
#   DEV_MODE  (1000 Proben):  ca. 10-15 min
#   FULL_MODE (10000 Proben): ca. 2-4 h
#
# Hinweise zum Tuning:
#   - Surrogate-RMSE (log10-Raum) unter 0.05 = brauchbar fuer Inversion
#   - Bei konstantem Inversionsergebnis: mehr Proben/Epochen oder groesseres MLP
#   - Regularisierungsgewicht senken (0.001), wenn Kontraste zu stark geglaettet
# =============================================================================

DEV_MODE = True

# Klassische SimPEG-Referenz-Inversion nach den Surrogate-Tests
# (Validierung des Surrogates am Ende der Kette; minutenlang, daher
#  abschaltbar). Laeuft im DEV-Lauf nach den Profil-Tests.
KLASSISCHE_INVERSION = True


# --- Parameter fuer DEV_MODE (Schnelltest, Iterieren beim Entwickeln) -------
# Anlagenkonfiguration: radius/current als BEREICH (min, max) -> das
# Surrogate wird konditioniert trainiert (je Probe log-uniform gezogen)
# und ist fuer alle Werte im Bereich einsetzbar. Skalar = fix (klassisch).
#   Zeiten 1e-5 bis 20e-3 s
DEV_CONFIG = {
    # Phase 0: Schichtraster (fix)
    "z_multiplier": 1.5,                     # Dicken-Wachstumsfaktor
    "first_layer": 1.0,                      # Dicke 1. Schicht (m)
    "z_max": 258.0,                          # max. Tiefe (m) -> 12 Schichten
    "n_layer": 13,                           # 12 Rasterschichten + Halbraum

    # Phase 1: Datengenerierung
    "n_samples": 50000,                      # Trainingsproben
    "n_times": 25,                           # Zeitkanaele 1e-5 bis 20e-3 s
    "sigma_range": (1e-3, 10.0),             # Leitfaehigkeit S/m (log-uniform)
    "n_workers": None,                       # CPU-Kerne (None = automatisch)
    # Klassenmischung der Trainingsmodelle (Summe 1.0):
    # Hintergrund 75 % (keine Floez-Topologie, breite Generalisierung,
    # verhindert Geisterfloez), Floez 15 % (duenne leitende Zone in
    # resistivem Rahmen), harte Faelle 10 % (tiefe/schwache Floez,
    # massive Leitung, extrem resistiv).
    "background_fraction": 0.75,
    "floez_fraction": 0.15,
    "hard_fraction": 0.10,
    "radius": (10.0, 100.0),                 # Schleifenradius-Bereich in m
                                             # (konditioniert; Skalar = fix)
    "current": (1.0, 20.0),                  # Sendestrom-Bereich in A
    "radius_test": 50.0,                     # Geometrie fuer die Phase-3-
    "current_test": 5.0,                     # Tests (Skalar, im Bereich)

    # Phase 2: Surrogate-Training
    "hidden_dims": [256, 512, 512, 256],          # MLP-Architektur
    "epochs": 250,                           # Trainingsepochen
    "batch_size": 128,                        # Batch-Groesse
    "lr": 1e-3,                              # Lernrate

    # Phase 3: Inversion / Multi-Test
    "n_stacks": 1024,                        # Stackings pro Sondierung
                                             # (Feldfall; Rauschen sinkt
                                             #  mit sqrt(n_stacks))
    "n_restarts": 1,                        # Multi-Start-Inversionen je
                                             # Station/Modell (1 = klassisch)
    "n_iterations": 500,                     # Inversions-Iterationen
    "inv_lr": 0.05,                          # Lernrate der Inversion
    "reg_weight": 0.001,                     # Tikhonov-Gewicht

    # Phase 3D: Referenz-Inversion (SimPEG)
    "max_iter_simpeg": 150,                  # Iterationen der klassischen
                                             # Referenz-Inversion; sollte
                                             # bis zum TargetMisfit oder
                                             # Konvergenz laufen, nicht
                                             # ans Limit (STOP!-Grund im
                                             # Log pruefen)
}


# --- Parameter fuer FULL_MODE (Produktionslauf) ------------------------------
FULL_CONFIG = {
    # Phase 0: Schichtraster (fix)
    "z_multiplier": 1.5,
    "first_layer": 1.0,
    "z_max": 258.0,
    "n_layer": 13,

    # Phase 1: Datengenerierung
    "n_samples": 100000,
    "n_times": 25,
    "sigma_range": (1e-3, 10.0),
    "n_workers": None,
    "radius": (10.0, 100.0),                 # konditioniert (Bereich)
    "current": (1.0, 20.0),
    "radius_test": 20.0,                     # Geometrie fuer die Phase-3-
    "current_test": 5.0,                     # Tests (Skalar, im Bereich)

    # Phase 2: Surrogate-Training
    "hidden_dims": [256, 512, 512, 256],
    "epochs": 300,
    "batch_size": 128,
    "lr": 1e-3,

    # Phase 3: Inversion / Multi-Test
    "n_stacks": 1024,                        # wie DEV-validiert (Feldfall)
    "n_restarts": 1,                         # Multi-Start-Inversionen je
                                             # Station/Modell. 1 = klassisch
                                             # (schnell, Zweck des Surrogates:
                                             # rasche Uebersicht). >1 nur fuer
                                             # Berichtslaeufe mit Unsicherheits-
                                             # band (ens_median/p16/p84)

    # Phase 3C: Profil-Test
    "n_stations": 10,                        # Stationen je Einfallswinkel
                                             # (jede Station = 1 Inversion;
                                             # mit KLASSISCHE_INVERSION=True
                                             # zusaetzlich ~30-60 s SimPEG
                                             # je Station!)
    "station_spacing": 5.0,                  # Stationsabstand in m
    "n_iterations": 500,
    "inv_lr": 0.05,
    "reg_weight": 0.001,                     # Tikhonov-Gewicht

    # Phase 3C: Profil-Test
    "n_stations": 10,                        # Stationen je Einfallswinkel
    "station_spacing": 5.0,                  # Stationsabstand in m

    # Phase 3D: Referenz-Inversion (SimPEG)
    "max_iter_simpeg": 150,                  # s. DEV_CONFIG-Kommentar
}


CONFIG = DEV_CONFIG if DEV_MODE else FULL_CONFIG


# =============================================================================
# PHASE 1A: 1D-TDEM-Forward-Modellierung mit SimPEG
# =============================================================================
# Quelle: SimPEG Tutorials (docs.simpeg.xyz)
# Zweck: Generierung von Trainingsdaten fuer 1D-Surrogate-Modelle
# Ausfuehrbar auf: RTX 4070 Mobile (lokal)
# =============================================================================

def make_fixed_layer_mesh(z_multiplier=1.5, first_layer=1.0, z_max=258.0):
    """Erzeugt ein festes Schichtraster fuer die Inversion (Standard bei TDEM).

    Schichtdicken wachsen geometrisch: 1. Schicht first_layer dick,
    jede weitere z_multiplier mal so dick wie die vorige.
    Die 12. Schicht ist endlich; darunter folgt der Halbraum (13. Parameter).

    Bei z_multiplier=1.5, first_layer=1.0, z_max=258:
        Grenzen [m]: 1.0, 2.5, 4.75, 8.13, 13.19, 20.78, 32.17, 49.26,
                    74.89, 113.34, 171.01, 257.51
        -> 12 endliche Schichten + Halbraum = 13 Leitfaehigkeiten

    Rueckgabe:
        thicknesses: Array der 12 endlichen Schichtdicken
    """
    boundaries = [0.0]
    t = first_layer
    while boundaries[-1] + t <= z_max:
        boundaries.append(boundaries[-1] + t)
        t *= z_multiplier
    thicknesses = np.diff(boundaries)
    return thicknesses


def generate_random_1d_model(n_layer,
                              sigma_range=(1e-3, 10.0)):
    """Generiert ein zufaelliges Leitfaehigkeitsmodell auf festem Raster.

    Die Schichtdicken kommen NICHT als Parameter - sie sind durch das
    feste Schichtraster (make_fixed_layer_mesh) fuer alle Modelle gleich.

    Parameter:
        n_layer: Anzahl Schichten (aus make_fixed_layer_mesh)
        sigma_range: (min, max) Leitfaehigkeit in S/m (log-uniform)

    Rueckgabe:
        sigma_model: Array der Leitfaehigkeiten pro Schicht (n_layer Werte)
    """
    sigma_model = 10 ** np.random.uniform(
        np.log10(sigma_range[0]), np.log10(sigma_range[1]), n_layer
    )
    return sigma_model


def loop_inductance(radius_m, wire_radius_m=LOOP_WIRE_RADIUS_M,
                    n_turns=LOOP_N_TURNS):
    """Induktivitaet einer kreisfoermigen Sendeschleife (Henry).

    Einturnig: L = mu0 * a * (ln(8a/r_w) - 2). Mehrturnig naehert sich
    L ~ n_turns^2 * L_einturnig (enge Kopplung der Windungen).
    """
    l_single = MU_0 * radius_m * (np.log(8.0 * radius_m / wire_radius_m)
                                  - 2.0)
    return l_single * n_turns ** 2


def loop_resistance(radius_m, wire_radius_m=LOOP_WIRE_RADIUS_M,
                    n_turns=LOOP_N_TURNS):
    """Ohm'scher Widerstand der Sendeschleife (Kupferdraht, rund)."""
    wire_length = 2.0 * np.pi * radius_m * n_turns
    cross_section = np.pi * wire_radius_m ** 2
    return COPPER_RESISTIVITY_OHM_M * wire_length / cross_section


def turnoff_exclusion_time(radius_m, current_a,
                           clamp_voltage=TVS_CLAMP_VOLTAGE,
                           r_parallel=R_PARALLEL_OHM,
                           wire_radius_m=LOOP_WIRE_RADIUS_M,
                           n_turns=LOOP_N_TURNS,
                           residual_fraction=0.01):
    """Totzeit nach dem Ramp-off: auszuschliessende Fruehzeit (Sekunden).

    Zwei Phasen beim Abschalten (Schleife L, R_loop; TVS klemmt auf V_c;
    Parallelwiderstand R_p):
      1. Klempphase (TVS leitend): Strom faellt von I0 auf
         I* = V_c/R_p (danach sperrt die TVS), mit
         L*dI/dt = -(V_c + R_loop*I) geloest:
             t_clamp = (L/R_loop) * ln((V_c + R_loop*I0)
                                       / (V_c + R_loop*I*))
      2. Exponentialer Abfall ueber R_loop + R_p mit tau = L/(R_loop+R_p)
         bis auf residual_fraction von I*:
             t_tail = tau * ln(1/residual_fraction)

    Rueckgabe:
        dict mit L, R_loop, tau, i_star, t_clamp, t_excl (= t_clamp+t_tail),
        alle Zeiten in Sekunden. Faustformel: t_excl ~ L*I0/V_c, da
        tau << t_clamp (Totzeit wird von der Klempphase dominiert).
    """
    L = loop_inductance(radius_m, wire_radius_m, n_turns)
    R_loop = loop_resistance(radius_m, wire_radius_m, n_turns)
    tau = L / (R_loop + r_parallel)
    i_star = clamp_voltage / r_parallel
    if current_a > i_star and R_loop > 0.0:
        t_clamp = (L / R_loop) * np.log(
            (clamp_voltage + R_loop * current_a)
            / (clamp_voltage + R_loop * i_star)
        )
    else:
        t_clamp = 0.0
    t_excl = t_clamp + tau * np.log(1.0 / residual_fraction)
    return {"L": L, "R_loop": R_loop, "tau": tau, "i_star": i_star,
            "t_clamp": t_clamp, "t_excl": t_excl}


def run_1d_tdem_forward(thicknesses, sigma_model, times,
                         src_location, rx_location,
                         radius=56.4, current=10.0):
    """Fuehrt eine 1D-TDEM-Vorwaertsmodellierung mit SimPEG durch.

    Parameter:
        thicknesses: Schichtdicken (n_layer - 1 Werte)
        sigma_model: Leitfaehigkeiten pro Schicht (n_layer Werte)
        times: Zeitkanaele in Sekunden
        src_location: Quellposition [x, y, z]
        rx_location: Empfaengerposition [x, y, z]
        radius: Schleifenradius in m (flaechengleiches Kreis-Equivalent)
        current: Sendestrom in A

    Rueckgabe:
        dpred: Vorhergesagte EM-Antwort (dB/dt) pro Zeitkanal
    """
    # Step-off Wellenform
    waveform = tdem.sources.StepOffWaveform(off_time=0.0)

    # Empfaenger (dB/dt z-Komponente)
    # use_inductive ist KEIN gueltiger Parameter und wurde entfernt
    rx_list = [
        tdem.receivers.PointMagneticFluxTimeDerivative(
            rx_location, times, orientation="z"
        )
    ]

    # Quelle: CircularLoop (nicht MagDipole) fuer Simulation1DLayered
    # MagDipole ist nur fuer 3D-Simulationen gueltig
    src_list = [
        tdem.sources.CircularLoop(
            receiver_list=rx_list,
            location=src_location,
            waveform=waveform,
            radius=radius,
            current=current,
        )
    ]

    # Survey
    survey = tdem.Survey(src_list)

    # Mapping
    model_mapping = maps.IdentityMap(nP=len(sigma_model))

    # Simulation
    simulation = tdem.Simulation1DLayered(
        survey=survey,
        thicknesses=thicknesses,
        sigmaMap=model_mapping,
    )

    # Vorwaertsmodellierung
    dpred = simulation.dpred(sigma_model)
    return dpred


# --- Worker fuer Multi-Core-Datengenerierung (Modulebene, Windows-sicher) ---
_WORKER_ARGS = None


def _draw_model_class(rng, n_layer=13, sigma_range=(1e-3, 10.0),
                      background_fraction=0.75,
                      floez_fraction=0.15,
                      hard_fraction=0.10):
    """Zieht ein Trainingsmodell aus der gewichteten Klassenmischung.

    Klassen (Training-Prior, NICHT die Testverteilung):
      - background (75 %): allgemeiner Untergrund. Alle Schichten
        unabhaengig log-uniform aus sigma_range - inkl. zufaelliger
        Kontraste, aber OHNE konstruierte Floez-Topologie. Verhindert
        Geisterfloez-Uebererkennung im homogenen Untergrund.
      - floez (15 %): duenne leitende Zone (1-3 Schichten, 0.5-10 S/m)
        in resistivem Rahmen (1e-3 bis 0.1 S/m), zufaellige Tiefe.
        Entspricht der make_test_model-Floeze-Signatur.
      - hard (10 %): Randfaelle an der Aufloesungsgrenze:
        sehr tiefe Floez (> 2/3 der Tiefenskala), sehr geringer
        Kontrast (Faktor ~10), oder massiv leitfaehiger Hintergrund
        (dike-artig, mehrere Schichten 1-10 S/m) und sehr resistive
        Modelle nahe sigma_min.

    Die drei Fraktionen muessen sich zu 1.0 summieren (assert).
    Rueckgabe: (sigma_model, class_label) mit class_label in
    {"background", "floez", "hard"}.
    """
    assert abs(background_fraction + floez_fraction + hard_fraction
               - 1.0) < 1e-9, "Klassenfraktionen muessen 1.0 ergeben"
    n_finite = n_layer - 1  # endliche Schichten + Halbraum
    lo = np.log10(sigma_range[0])
    hi = np.log10(sigma_range[1])

    u = rng.uniform()
    if u < background_fraction:
        # --- Hintergrund: alles log-uniform, keine Floez-Topologie ---
        sigma = 10 ** rng.uniform(lo, hi, n_layer)
        label = "background"
    elif u < background_fraction + floez_fraction:
        # --- Floez: leitende Zone in resistivem Rahmen ---
        sigma = 10 ** rng.uniform(lo, np.log10(0.1), n_layer)
        zone_width = rng.integers(1, 4)  # 1-3 Schichten
        zone_start = rng.integers(0, n_layer)
        zone_end = min(zone_start + zone_width, n_layer)
        sigma[zone_start:zone_end] = 10 ** rng.uniform(
            np.log10(0.5), hi, zone_end - zone_start
        )
        label = "floez"
    else:
        # --- Harte Faelle: Aufloesungsgrenzen und Extreme ---
        case = rng.integers(0, 3)
        if case == 0:
            # sehr tiefes Floez (unteres Drittel des Rasters)
            sigma = 10 ** rng.uniform(lo, np.log10(0.1), n_layer)
            start = int(2 * n_layer / 3)
            sigma[start:] = 10 ** rng.uniform(np.log10(0.5), hi,
                                               n_layer - start)
            # geringer Kontrast mit ~50 % Wahrscheinlichkeit
            if rng.uniform() < 0.5:
                sigma[start:] *= 10 ** rng.uniform(-0.5, 0.0,
                                                    n_layer - start)
        elif case == 1:
            # massiv leitfaehiger Hintergrund (dike-artig)
            sigma = 10 ** rng.uniform(np.log10(0.1), hi, n_layer)
        else:
            # sehr resistives Modell nahe sigma_min
            sigma = 10 ** rng.uniform(lo, lo + 0.5, n_layer)
        label = "hard"
    return sigma, label


def _forward_worker(args):
    """Berechnet einen Block von Vorwaertsmodellierungen in einem Worker-Prozess.

    Wird von multiprocessing.Pool aufgerufen; alle noetigen Parameter
    kommen ueber args (Global _WORKER_ARGS wird von jedem Worker gesetzt).

    radius_range/current_range: wenn als Bereich (min, max) gegeben,
    wird JE PROBE ein Radius/Strom log-uniform gezogen (parametrisiertes
    Surrogate); bei einem Skalar bleibt der Wert fix.
    """
    import numpy as _np

    idx_block, seed = args
    if _WORKER_ARGS is None:
        raise RuntimeError("Worker-Argumente nicht initialisiert")
    (thicknesses, times, src_location, rx_location,
     sigma_range, radius_arg, current_arg, n_layer, block_size,
     fractions) = _WORKER_ARGS
    background_fraction, floez_fraction, hard_fraction = fractions

    def _draw_model_class_local(rng):
        return _draw_model_class(
            rng, n_layer=n_layer, sigma_range=sigma_range,
            background_fraction=background_fraction,
            floez_fraction=floez_fraction,
            hard_fraction=hard_fraction,
        )

    def _draw(param, rng):
        """Skalar -> fix; Tupel -> log-uniform aus dem Bereich."""
        if isinstance(param, (tuple, list)):
            return 10 ** rng.uniform(_np.log10(param[0]),
                                     _np.log10(param[1]))
        return float(param)

    # Jeder Worker hat eigenen Zufallsgenerator (reproduzierbar)
    rng = _np.random.default_rng(seed)

    models_block = []
    responses_block = []
    radii_block = []
    currents_block = []
    classes_block = []
    for _ in range(block_size):
        sigma_model, model_class = _draw_model_class_local(rng)
        radius = _draw(radius_arg, rng)
        current = _draw(current_arg, rng)
        try:
            dpred = run_1d_tdem_forward(
                thicknesses, sigma_model, times,
                src_location, rx_location,
                radius=radius, current=current
            )
            if _np.all(_np.isfinite(dpred)) and _np.all(_np.abs(dpred) > 0):
                models_block.append(sigma_model)
                responses_block.append(dpred)
                radii_block.append(radius)
                currents_block.append(current)
                classes_block.append(model_class)
        except Exception:
            continue
    return models_block, responses_block, radii_block, currents_block, classes_block


def _init_worker(worker_args):
    """Initialisiert globale Parameter in jedem Worker-Prozess."""
    global _WORKER_ARGS
    _WORKER_ARGS = worker_args


def generate_training_data_1d(n_samples, times, src_location, rx_location,
                               thicknesses,
                               sigma_range=(1e-3, 10.0),
                               radius=20.0, current=5.0,
                               n_workers=None,
                               fractions=(0.75, 0.15, 0.10),
                               output_file="tdem_1d_training_data.npz"):
    """Generiert einen 1D-Trainingsdatensatz auf festem Raster (multi-core).

    Parameter:
        n_samples: Anzahl der Trainingsproben
        times: Zeitkanaele
        src_location: Quellposition
        rx_location: Empfaengerposition
        thicknesses: Feste Schichtdicken aus make_fixed_layer_mesh
        sigma_range: (min, max) Leitfaehigkeit in S/m (log-uniform)
        radius: Schleifenradius in m — Skalar (fix) ODER (min, max)
            fuer log-uniform gezogene Radien (parametrisiertes
            Surrogate)
        current: Sendestrom in A — Skalar ODER (min, max)
        n_workers: Anzahl CPU-Kerne (None = automatisch, max. logische Kerne)
        fractions: (background, floez, hard) Klassenmischung der
            Trainingsmodelle; Summe 1.0. background = allgemeiner
            Untergrund (keine Floez-Topologie), floez = duenne
            leitende Zone in resistivem Rahmen, hard = Randfaelle
            (tiefe/schwache Floez, massive Leitung, extrem resistiv)
        output_file: Dateiname fuer Speicherung

    Rueckgabe:
        models, responses, radii, currents: Arrays; radii/currents
        enthalten den je Probe verwendeten Radius/Strom (bei fester
        Konfiguration konstante Arrays).
        Speichert zusaetzlich alles als .npz-Datei
    """
    import multiprocessing as mp
    import os

    n_layer = len(thicknesses) + 1  # Rasterschichten + Halbraum

    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)  # 1 Kern fuer System frei

    # Bloecke bilden: grob 50 Bloecke pro Worker fuer gleichmaessige Last
    n_blocks = max(n_workers * 50, 100)
    block_size = int(np.ceil(n_samples / n_blocks))
    actual_blocks = int(np.ceil(n_samples / block_size))

    worker_args = (thicknesses, times, src_location, rx_location,
                   sigma_range, radius, current, n_layer, block_size,
                   fractions)

    seeds = np.random.SeedSequence(42).spawn(actual_blocks)
    block_args = [(i, int(seed.generate_state(1)[0])) for i, seed
                  in enumerate(seeds)]

    print(f"Datengenerierung: {n_samples} Proben auf "
          f"{n_workers} Worker-Prozessen "
          f"({actual_blocks} Bloecke)")

    if n_workers <= 1:
        _init_worker(worker_args)
        results = [_forward_worker(ba) for ba in block_args]
    else:
        ctx = mp.get_context("spawn")  # Windows/pytest-sicher
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(worker_args,),
        ) as pool:
            results = pool.map(_forward_worker, block_args)

    all_models = []
    all_responses = []
    all_radii = []
    all_currents = []
    all_classes = []
    class_counts = {"background": 0, "floez": 0, "hard": 0}
    for models_block, responses_block, radii_block, currents_block, \
            classes_block in results:
        all_models.extend(models_block)
        all_responses.extend(responses_block)
        all_radii.extend(radii_block)
        all_currents.extend(currents_block)
        all_classes.extend(classes_block)
        for c in classes_block:
            class_counts[c] += 1

    # Auf Zielanzahl kuerzen (Bloecke koennen mehr liefern)
    all_models = np.array(all_models[:n_samples])
    all_responses = np.array(all_responses[:n_samples])
    all_radii = np.array(all_radii[:n_samples])
    all_currents = np.array(all_currents[:n_samples])
    all_classes = all_classes[:n_samples]

    print(f"Klassenmischung (vor Kuerzung): {class_counts}")

    np.savez(output_file,
             models=all_models,
             responses=all_responses,
             radii=all_radii,
             currents=all_currents,
             classes=np.array(all_classes),
             thicknesses=thicknesses,
             times=times)
    print(f"Datensatz gespeichert: {len(all_models)} Proben in {output_file}")
    return all_models, all_responses, all_radii, all_currents


# =============================================================================
# PHASE 1B: 3D-TDEM-Forward-Modellierung (Cloud-GPU)
# =============================================================================
# Fuer 3D-Forward-Modellierung wird TorchTEM3D oder SimPEG 3D verwendet.
# TorchTEM3D: PyTorch-basiert, GPU-beschleunigt, automatische Differentiation.
#
# Alternative: SimPEG 3D Simulation
#   simulation = tdem.simulation.Simulation3DMagneticFluxDensity(
#       mesh, survey=survey, sigmaMap=model_map, solver=Solver
#   )
#   simulation.time_steps = time_steps
#
# WICHTIG: 3D-Forward-Modellierung erfordert 24-80 GB VRAM.
# Auf der 4070 Mobile (8 GB) nur sehr kleine Netze moeglich.
# Empfehlung: Cloud-GPU (A100 80GB) mieten fuer Phase 1B.
# =============================================================================


# =============================================================================
# PHASE 2A: 1D-Surrogate-Modell (MLP) mit PyTorch
# =============================================================================
# Zweck: Surrogate fuer 1D-TDEM-Vorwaertsmodellierung
# Input:  Leitfaehigkeitsmodell (sigma_model, log10-transformiert)
# Output: EM-Antwort (dB/dt pro Zeitkanal, log10-transformiert)
# Hardware: RTX 4070 Mobile (8 GB VRAM) - vollstaendig ausreichend
# =============================================================================

class TDEMSurrogateMLP(nn.Module):
    """Multilayer-Perceptron als Surrogate-Modell fuer 1D-TDEM-Forward.

    Input:  Leitfaehigkeitsmodell (log10-transformiert) plus optional
            2 Konditionierungseingaben (log10 Radius, log10 Strom),
            jeweils wie die Modellparameter normalisiert
    Output: EM-Antwort (log10-transformiert)
    """

    def __init__(self, n_model_params, n_time_channels,
                 hidden_dims=[256, 512, 512, 256], n_cond=2):
        super().__init__()
        self.n_cond = n_cond
        layers = []
        in_dim = n_model_params + n_cond
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_time_channels))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TDEMDataset(Dataset):
    """PyTorch Dataset fuer TDEM-Trainingsdaten auf festem Schichtraster.

    Eingabe: log10(sigma) je Schicht (z-Score-normalisiert)
    Ziel:    log10(|dB/dt|) je Zeitkanal (z-Score-normalisiert)

    Die Schichtdicken sind NICHT Teil der Eingabe - sie sind durch das
    feste Schichtraster (make_fixed_layer_mesh) fuer alle Modelle gleich.
    """

    def __init__(self, models, responses,
                 x_stats=None, y_stats=None):
        """Parameter:
            models: Leitfaehigkeiten (n_samples, n_layer), linearer Raum
            responses: Antworten (n_samples, n_times), linearer Raum
            x_stats, y_stats: optional vorgegebene Normalisierungsparameter
        """
        x = np.log10(models)
        y = np.log10(np.abs(responses) + 1e-30)

        if x_stats is None:
            self.mu_x, self.std_x = x.mean(axis=0), x.std(axis=0) + 1e-12
        else:
            self.mu_x, self.std_x = x_stats
        if y_stats is None:
            self.mu_y, self.std_y = y.mean(axis=0), y.std(axis=0) + 1e-12
        else:
            self.mu_y, self.std_y = y_stats

        self.x = (x - self.mu_x) / self.std_x
        self.y = (y - self.mu_y) / self.std_y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx]), torch.FloatTensor(self.y[idx])


def save_checkpoint(model, optimizer, scheduler, epoch, train_losses,
                    checkpoint_file="tdem_surrogate_checkpoint.pth"):
    """Speichert einen vollstaendigen Trainings-Checkpoint zum Fortsetzen.

    Enthaelt Modellgewichte, Optimizer- und Scheduler-Zustand sowie die
    bisherige Epochennummer und den Loss-Verlauf, sodass das Training
    exakt an derselben Stelle fortgesetzt werden kann.
    """
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "train_losses": train_losses,
    }, checkpoint_file)
    print(f"Checkpoint gespeichert: {checkpoint_file} (Epoche {epoch})")


def load_checkpoint(checkpoint_file, model, optimizer=None, scheduler=None,
                    device="cuda"):
    """Laedt einen Checkpoint und gibt die Startepode zurueck.

    Parameter:
        checkpoint_file: Pfad zur .pth-Datei
        model: TDEMSurrogateMLP-Instanz (Architektur muss identisch sein)
        optimizer: Optional, wird mit gespeichertem Zustand ueberschrieben
        scheduler: Optional, wird mit gespeichertem Zustand ueberschrieben
        device: "cuda" oder "cpu"

    Rueckgabe:
        (model, optimizer, scheduler, start_epoch, train_losses)
    """
    ckpt = torch.load(checkpoint_file, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    train_losses = ckpt.get("train_losses", [])
    print(f"Checkpoint geladen: {checkpoint_file} "
          f"(weiter ab Epoche {start_epoch})")
    return model, optimizer, scheduler, start_epoch, train_losses


def train_surrogate_mlp(models, responses, n_model_params, n_time_channels,
                         epochs=200, batch_size=64, lr=1e-3, device="cuda",
                         hidden_dims=[256, 512, 512, 256],
                         radii=None, currents=None,
                         resume_file=None,
                         checkpoint_every=25,
                         checkpoint_file="tdem_surrogate_checkpoint.pth"):
    """Trainiert das 1D-Surrogate-MLP.

    Parameter:
        models: Trainingsmodelle (n_samples, n_model_params)
        responses: Trainingsantworten (n_samples, n_time_channels)
        n_model_params: Anzahl der Modellparameter
        n_time_channels: Anzahl der Zeitkanaele
        epochs: Anzahl der Trainingsepochen
        batch_size: Batch-Groesse
        lr: Lernrate
        device: "cuda" oder "cpu"
        hidden_dims: Liste der verdeckten Schichtgroessen
        radii, currents: Optional (n_samples,) Arrays mit je Probe
            verwendetem Schleifenradius/Sendestrom. Werden sie
            uebergeben, lernt das Netz sie als zusaetzliche
            Konditionierungseingaben (parametrisiertes Surrogate);
            andernfalls bleibt das Verhalten wie vorher (n_cond=0
            aus Sicht der Daten; das Netz behaelt 2 Reserve-Eingaben
            nur bei radii/currents).
        resume_file: Optionaler Pfad zu einem Checkpoint (.pth); das Training
            wird dann mit gespeicherten Gewichten, Optimizer- und
            Scheduler-Zustand fortgesetzt
        checkpoint_every: Alle N Epochen wird ein Checkpoint gespeichert
        checkpoint_file: Dateiname fuer die Checkpoints

    Rueckgabe:
        model: Trainiertes Surrogate-Modell
        train_losses: Verlustverlauf
        norm_stats: Normalisierungsparameter (mu/std fuer Eingang und
            Ziel; bei Konditionierung um mu/std fuer log10(radius)
            und log10(current) als zusaetzliche Spalten erweitert,
            Keys "mu_cond"/"std_cond")
    """
    conditioned = radii is not None and currents is not None
    if conditioned:
        # Eingabematrix im LINEAREN Raum: sigma | radius | current.
        # TDEMDataset wendet selbst log10 an (siehe Klasse) - auch auf
        # die Konditionierungsspalten, was genau gewuenscht ist.
        x_raw = np.hstack([
            models,
            np.asarray(radii, dtype=np.float64)[:, None],
            np.asarray(currents, dtype=np.float64)[:, None],
        ])
        n_cond = 2
    else:
        x_raw = models
        n_cond = 0

    dataset = TDEMDataset(x_raw, responses)

    # Normalisierungsparameter fuer spaetere Anwendung aufbewahren
    if conditioned:
        norm_stats = {
            "mu_x": dataset.mu_x[:-2], "std_x": dataset.std_x[:-2],
            "mu_cond": dataset.mu_x[-2:], "std_cond": dataset.std_x[-2:],
            "mu_y": dataset.mu_y, "std_y": dataset.std_y,
        }
    else:
        norm_stats = {
            "mu_x": dataset.mu_x, "std_x": dataset.std_x,
            "mu_y": dataset.mu_y, "std_y": dataset.std_y,
        }

    # Kompletten Datensatz als Tensoren auf die GPU legen.
    # Der Datensatz (n_samples x Dimensionen) ist klein genug fuer den VRAM;
    # das ist deutlich schneller als ein DataLoader mit einzelnen Items
    # (besonders auf Windows, wo Worker-Prozesse teuer sind).
    x_all = torch.FloatTensor(dataset.x).to(device)
    y_all = torch.FloatTensor(dataset.y).to(device)
    n_total = x_all.shape[0]

    model = TDEMSurrogateMLP(n_model_params, n_time_channels,
                             hidden_dims=hidden_dims,
                             n_cond=n_cond).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.MSELoss()

    # Mixed Precision fuer VRAM-Ersparnis
    # torch.amp ist die aktuelle API (torch.cuda.amp ist deprecated)
    scaler = GradScaler("cuda")

    train_losses = []
    model.train()

    # Fortsetzen eines abgebrochenen Trainings
    start_epoch = 0
    if resume_file is not None:
        _, _, _, start_epoch, train_losses = load_checkpoint(
            resume_file, model, optimizer, scheduler, device=device
        )
        model.train()
        if start_epoch >= epochs:
            print("Checkpoint liegt bereits hinter der Zielepoche - "
                  "epochs erhoehen, um weiterzutrainieren.")

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        # Zufaellige Batch-Indizes direkt auf der GPU ziehen
        perm = torch.randperm(n_total, device=device)
        n_batches = (n_total + batch_size - 1) // batch_size
        for batch_idx in range(n_batches):
            idx = perm[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            model_input = x_all[idx]
            target = y_all[idx]

            optimizer.zero_grad()

            with autocast("cuda"):
                prediction = model(model_input)
                loss = criterion(prediction, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6e} - "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Periodischer Checkpoint (Absturzsicherung, Fortsetzen moeglich)
        if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs:
            save_checkpoint(model, optimizer, scheduler, epoch,
                            train_losses, checkpoint_file)

    # Normalisierungsparameter zusammen mit Checkpoint speichern
    torch.save(norm_stats, os.path.join(
        os.path.dirname(checkpoint_file), "norm_stats.pth"))

    return model, train_losses, norm_stats


# =============================================================================
# PHASE 2B: 3D-U-Net Surrogate (fuer 3D-TDEM-Inversion)
# =============================================================================
# Zweck: Surrogate fuer 3D-TDEM-Forward-Modellierung oder End-to-End-Inversion
# Architektur: 3D-U-Net (Encoder-Decoder mit Skip-Connections)
# Hardware: Cloud-GPU (A100 80GB empfohlen) oder 4070 Mobile mit Einschraenkungen
#
# Vorlagen:
#   - wolny/pytorch-3dunet (GitHub): fertige 3D-U-Net Implementierung
#   - fepegar/unet (GitHub): pip install unet, 1D/2D/3D U-Net
#   - cosmic-cortex/pytorch-UNet: 2D und 3D UNet
#
# Fuer 8 GB VRAM (4070 Mobile):
#   - Mixed Precision (FP16) zwingend
#   - Gradient-Checkpointing zwingend
#   - Batch-Size 1 oder 2
#   - Patch-basiertes Training empfohlen
# =============================================================================

class ConvBlock3D(nn.Module):
    """3D-Convolution-Block mit zwei Conv3d-Schichten und BatchNorm."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """3D-U-Net fuer TDEM-Surrogate-Modellierung.

    Architektur: Encoder-Decoder mit Skip-Connections.
    Input:  (batch, in_channels, D, H, W)
    Output: (batch, out_channels, D, H, W)

    Anpassungen fuer 8 GB VRAM:
        - Reduzierte Kanalzahlen (16, 32, 64, 128 statt 64, 128, 256, 512)
        - Mixed Precision Training
        - Gradient-Checkpointing
    """

    def __init__(self, in_channels=1, out_channels=1,
                 features=[16, 32, 64, 128]):
        super().__init__()
        self.features = features

        # Encoder (Downsampling)
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        for f in features:
            self.encoder_blocks.append(ConvBlock3D(in_channels, f))
            in_channels = f

        # Bottleneck
        self.bottleneck = ConvBlock3D(features[-1], features[-1] * 2)

        # Decoder (Upsampling)
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(f * 2, f, 2, stride=2))
            self.decoder_blocks.append(ConvBlock3D(f * 2, f))

        # Output
        self.output_conv = nn.Conv3d(features[0], out_channels, 1)

    def forward(self, x):
        skip_connections = []

        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(len(self.features)):
            x = self.upconvs[i](x)
            skip = skip_connections[i]
            # Padding falls Dimensionen nicht uebereinstimmen
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.pad(
                    x,
                    [0, skip.shape[4] - x.shape[4],
                     0, skip.shape[3] - x.shape[3],
                     0, skip.shape[2] - x.shape[2]]
                )
            x = torch.cat([skip, x], dim=1)
            x = self.decoder_blocks[i](x)

        return self.output_conv(x)


# =============================================================================
# PHASE 2C: ConvLSTM-Modul (fuer ATEM3D-Net Stil)
# =============================================================================
# Zweck: Erfassung der zeitlichen Abhaengigkeiten in TDEM-Daten
# Kombination: 3D-U-Net (raeumlich) + ConvLSTM (zeitlich)
#
# Vorlagen:
#   - ndrplz/ConvLSTM_pytorch (GitHub): referenzimplementierung
#   - KL4805/ConvLSTM-Pytorch: alternative Implementierung
# =============================================================================

class ConvLSTMCell(nn.Module):
    """Convolutional LSTM Cell nach Shi et al. (2015).

    Quelle: ndrplz/ConvLSTM_pytorch (GitHub)
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        # kernel_size kann int oder tuple sein
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1
        )
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(
                batch_size, self.hidden_dim, height, width, device=device
            ),
            torch.zeros(
                batch_size, self.hidden_dim, height, width, device=device
            ),
        )


class ConvLSTM(nn.Module):
    """ConvLSTM-Modul fuer sequenzielle TDEM-Datenverarbeitung.

    Parameter:
        input_dim: Anzahl der Eingabekanaele (int oder list)
        hidden_dim: Anzahl der verdeckten Kanaele (int oder list)
        kernel_size: Faltungs-Kernelgroesse (int oder tuple)
        num_layers: Anzahl der ConvLSTM-Schichten
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * num_layers

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        layers = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            layers.append(
                ConvLSTMCell(cur_input_dim, hidden_dim[i], kernel_size, bias)
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = [
                self.layers[i].init_hidden(b, h, w, input_tensor.device)
                for i in range(self.num_layers)
            ]

        last_state = []
        seq_len = input_tensor.size(1)

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.layers[layer_idx](
                    input_tensor[:, t, ...], (h, c)
                )
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            input_tensor = layer_output
            last_state.append((h, c))

        if not self.return_all_layers:
            last_state = last_state[-1]

        return layer_output, last_state


# =============================================================================
# PHASE 3: Inversion mit Surrogate-Modell
# =============================================================================
# Variante A: Surrogate als Forward-Ersatz in klassischer Inversion
# Variante B: End-to-End DL-Inversion (direkte Anwendung auf Felddaten)
# =============================================================================

class SurrogateInversion:
    """Klassische Inversion mit Surrogate als Forward-Ersatz (Variante A).

    Verwendet das trainierte Surrogate-Modell anstelle der teuren
    physikalischen Vorwaertsmodellierung innerhalb der Inversionsschleife.

    WICHTIG: Die Inversion arbeitet im log10-Raum, um negative Leitfaehigkeiten
    zu verhindern. Das Surrogate-Modell wurde mit log10-transformierten Daten
    trainiert, daher ist diese Parameterisierung konsistent.
    """

    def __init__(self, surrogate_model, times, device="cuda",
                 norm_stats=None, sigma_bounds=(1e-3, 10.0),
                 radius=None, current=None, dead_time=None):
        self.surrogate = surrogate_model.to(device)
        self.surrogate.eval()
        self.times = times
        self.device = device
        # Zulaessiger Leitfaehigkeitsbereich (Trainingsbereich).
        # Die Inversion wird darauf projiziert - das Surrogate darf
        # nicht in seine Extrapolationszone getrieben werden.
        self.sigma_bounds = sigma_bounds
        # Normalisierungsparameter des Trainings (z-Score im log10-Raum)
        if norm_stats is None:
            raise ValueError(
                "norm_stats ist erforderlich (Rueckgabe von "
                "train_surrogate_mlp). Das Surrogate arbeitet mit "
                "normalisierten Ein-/Ausgaben."
            )
        self.norm_stats = norm_stats
        # Konditionierung (parametrisiertes Surrogate): radius/current
        # werden als fixe zusaetzliche Netzeingaben angehaengt.
        # Nur aktiv, wenn das Training konditioniert war (mu_cond
        # in norm_stats vorhanden) UND hier Werte uebergeben wurden.
        self.conditioned = (
            "mu_cond" in norm_stats
            and radius is not None and current is not None
        )
        if self.conditioned:
            cond_raw = np.array([np.log10(radius), np.log10(current)])
            self.cond_norm = (
                (cond_raw - norm_stats["mu_cond"])
                / norm_stats["std_cond"]
            ).astype(np.float32)
        else:
            self.cond_norm = None
        # Totzeit (Ramp-off via TVS + R_parallel): Kanaele mit t < dead_time
        # werden in invert() hart aus der Misfit-Gewichtung genommen
        # (Surrogate ist auf idealen Step-off trainiert; die Ramp-off-
        # Kontamination ist systematisch, keine Rauschgewichtung).
        # dead_time=None -> automatisch aus radius/current geschaetzt;
        # dead_time=0 -> deaktiviert.
        if dead_time is None and radius is not None and current is not None:
            dead_time = turnoff_exclusion_time(radius, current)["t_excl"]
        self.dead_time = 0.0 if dead_time is None else float(dead_time)
        self.n_excluded = int(np.sum(
            np.asarray(times, dtype=np.float64) < self.dead_time))
        if self.dead_time > 0.0 and self.n_excluded > 0:
            print(f"Totzeit {self.dead_time * 1e6:.1f} us: "
                  f"{self.n_excluded}/{len(times)} Kanaele ausgeschlossen")

    def _normalize_input(self, sigma_model):
        """Leitfaehigkeiten (S/m) -> normalisierte Netzeingabe."""
        x = np.log10(sigma_model)
        return (x - self.norm_stats["mu_x"]) / self.norm_stats["std_x"]

    def _append_cond(self, x_norm_tensor):
        """Haengt die normierte Konditionierung (radius, current) an
        die normalisierten Modelleingaben an (falls konditioniert)."""
        if self.cond_norm is None:
            return x_norm_tensor
        cond = torch.tensor(
            self.cond_norm, dtype=torch.float32, device=x_norm_tensor.device
        )
        # x_norm_tensor: (..., n_layer) -> (..., n_layer + 2)
        cond_batch = cond.expand(*x_norm_tensor.shape[:-1], 2)
        return torch.cat([x_norm_tensor, cond_batch], dim=-1)

    def _denormalize_output(self, pred_norm):
        """Normalisierte Netzausgabe -> log10-Antwort."""
        return (pred_norm * self.norm_stats["std_y"]
                + self.norm_stats["mu_y"])

    def forward_surrogate(self, sigma_model):
        """Surrogate-Vorwaertsaufruf: Modell -> EM-Antwort.

        Input:  sigma_model (Leitfaehigkeiten in S/m), auf festem Raster
        Output: EM-Antwort (numpy array, dB/dt)
        """
        with torch.no_grad():
            x_norm = torch.tensor(
                self._normalize_input(sigma_model),
                dtype=torch.float32
            ).to(self.device)
            pred_norm = self.surrogate(self._append_cond(x_norm.unsqueeze(0)))
            log_resp = self._denormalize_output(
                pred_norm.squeeze(0).cpu().numpy()
            )
            response = 10 ** log_resp
        return response

    def invert(self, observed_data, initial_model,
               n_iterations=100, lr=0.01, regularization_weight=0.1,
               noise_dt=None, verbose=True, n_restarts=1, seed=None):
        """Gradientenbasierte Inversion mit Surrogate auf festem Raster.

        Optimiert ausschliesslich die Leitfaehigkeiten; die Schichtdicken
        sind durch das feste Raster vorgegeben.

        Parameter:
            observed_data: Gemessene Felddaten (numpy array)
            initial_model: Startmodell (Leitfaehigkeiten, numpy array)
            n_iterations: Maximale Anzahl Iterationen
            lr: Lernrate / Schrittweite
            regularization_weight: Tikhonov-Regularisierungsgewicht
            noise_dt: dB/dt-Rauschen (1 Sigma) je Kanal (z. B. aus
                compute_noise_dt). Bei Angabe wird der Misfit mit der
                inversen Rauschvarianz gewichtet (Chi-Quadrat): Kanaele,
                deren Rauschen das Signal uebersteigt, tragen kaum bei
                und koennen nicht mehr durch tiefe Leiter "erklaert"
                werden. Ohne Angabe: ungewichtete Mittelung (bisheriges
                Verhalten, fuer rauschfreie Daten korrekt).
            verbose: Iterationsprotokoll ausgeben
            n_restarts: Multi-Start-Inversion. 1 = klassisch (nur das
                uebergebene Startmodell). >1 = zusaetzlich n_restarts-1
                zufaellige Startmodelle (log-uniform im Trainingsbereich);
                das Modell mit dem besten Misfit gewinnt. Das Ensemble
                aller Restart-Modelle steht danach in self.last_ensemble
                (n_restarts, n_layer) fuer Unsicherheitsstatistik
                (Median, Perzentile).
            seed: Zufallsseed fuer die Startmodelle (reproduzierbar)

        Rueckgabe:
            best_sigma: invertiertes Leitfaehigkeitsmodell (S/m)
        """
        n_layer = len(initial_model)
        rng = np.random.default_rng(seed)

        def _random_start():
            """Log-uniformes Zufalls-Startmodell im Trainingsbereich."""
            return 10 ** rng.uniform(
                np.log10(self.sigma_bounds[0]),
                np.log10(self.sigma_bounds[1]), n_layer,
            )

        starts = [np.array(initial_model, dtype=np.float64)]
        for _ in range(max(0, n_restarts - 1)):
            starts.append(_random_start())

        # Beobachtete Daten: log10-Raum -> normalisierter Zielraum
        observed = torch.tensor(
            (np.log10(np.abs(observed_data) + 1e-30)
             - self.norm_stats["mu_y"]) / self.norm_stats["std_y"],
            dtype=torch.float32, device=self.device
        )

        # Kanalgewichte aus Rauschmodell (Chi-Quadrat-Misfit):
        # dB/dt-Rauschen -> Rauschen im log10-Raum -> normalisiert.
        # Kanaele mit Signal unter dem Rauschfloor erhalten ~ Gewicht 0.
        if noise_dt is not None:
            noise_dt = np.asarray(noise_dt, dtype=np.float64)
            sigma_log = noise_dt / (np.log(10.0)
                                    * (np.abs(observed_data) + 1e-30))
            # normalisiertes Rauschen je Kanal, mit Untergrenze gegen
            # Division durch 0 bei extrem gutem SNR
            sigma_norm = np.maximum(
                sigma_log / self.norm_stats["std_y"], 1e-3
            )
            inv_var = torch.tensor(
                1.0 / sigma_norm ** 2,
                dtype=torch.float32, device=self.device,
            )
            # Normierung: Gewichte summieren auf Kanalanzahl
            inv_var = inv_var * (len(observed_data)
                                 / inv_var.sum().item())
        else:
            inv_var = None

        # Totzeit-Maske: Kanaele vor self.dead_time haben Gewicht 0
        if self.dead_time > 0.0:
            mask = torch.tensor(
                (np.asarray(self.times, dtype=np.float64) >= self.dead_time)
                .astype(np.float32),
                dtype=torch.float32, device=self.device,
            )
            inv_var = mask if inv_var is None else inv_var * mask

        # Begrenzung auf den Trainingsbereich des Surrogates:
        # log10(sigma) in [log10(sigma_min), log10(sigma_max)].
        # Ohne Begrenzung treibt Rauschen die Optimierung in die
        # Extrapolationszone des Netzes (unzulaessige Werte > 10 S/m).
        x_lo = torch.tensor(
            (np.log10(self.sigma_bounds[0]) - self.norm_stats["mu_x"])
            / self.norm_stats["std_x"],
            dtype=torch.float32, device=self.device,
        )
        x_hi = torch.tensor(
            (np.log10(self.sigma_bounds[1]) - self.norm_stats["mu_x"])
            / self.norm_stats["std_x"],
            dtype=torch.float32, device=self.device,
        )

        best_misfit = float("inf")
        best_sigma = np.array(initial_model, dtype=np.float64).copy()

        def _denorm(x):
            """normalisierter Vektor -> sigma."""
            x_lin = x * self.norm_stats["std_x"] + self.norm_stats["mu_x"]
            return 10 ** x_lin[:n_layer]

        # --- Multi-Start: ein Optimierungslauf je Startmodell ---
        ensemble = []
        for run_idx, start_model in enumerate(starts, start=1):
            x_opt = torch.tensor(
                self._normalize_input(start_model),
                dtype=torch.float32,
                requires_grad=True, device=self.device,
            )
            optimizer = torch.optim.Adam([x_opt], lr=lr)
            best_misfit_run = float("inf")
            best_sigma_run = start_model.copy()

            for iteration in range(n_iterations):
                optimizer.zero_grad()

                # Surrogate arbeitet vollstaendig im normalisierten Raum
                prediction = self.surrogate(
                    self._append_cond(x_opt.unsqueeze(0))
                ).squeeze(0)

                # Daten-Misfit (im normalisierten Zielraum)
                if inv_var is not None:
                    # Chi-Quadrat: gewichtete mittlere quadratische Abw.,
                    # normiert auf die Summe der Gewichte (bei Totzeit-Maske
                    # also Mittel ueber die verbleibenden Kanaele)
                    data_misfit = torch.sum(
                        inv_var * (prediction - observed) ** 2
                    ) / inv_var.sum().clamp_min(1e-12)
                else:
                    data_misfit = nn.functional.mse_loss(prediction,
                                                         observed)

                # Tikhonov-Regularisierung (Glattheit im log-Raum)
                reg = torch.mean(torch.diff(x_opt) ** 2)
                total_loss = data_misfit + regularization_weight * reg

                total_loss.backward()
                optimizer.step()

                # Projektion auf den Trainingsbereich (n. jedem Schritt)
                with torch.no_grad():
                    x_opt.clamp_(min=x_lo, max=x_hi)

                current_misfit = data_misfit.item()
                if current_misfit < best_misfit_run:
                    best_misfit_run = current_misfit
                    best_sigma_run = _denorm(
                        x_opt.detach().cpu().numpy()
                    )

                if verbose and (iteration + 1) % 10 == 0:
                    print(f"  Restart {run_idx}/{len(starts)} "
                          f"Iteration {iteration+1}/{n_iterations} - "
                          f"Misfit: {current_misfit:.6e}")

            ensemble.append(best_sigma_run)
            if best_misfit_run < best_misfit:
                best_misfit = best_misfit_run
                best_sigma = best_sigma_run

        # Ensemble fuer Unsicherheitsstatistik bereitstellen
        # (Median/P16/P84 je Schicht in den Testfunktionen)
        self.last_ensemble = np.array(ensemble)
        self.last_best_misfit = best_misfit

        if verbose:
            print(f"Inversion abgeschlossen ({len(starts)} Start(s)). "
                  f"Bester Misfit: {best_misfit:.6e}")
        return best_sigma


# =============================================================================
# ERGEBNIS-EXPORT: kurze Ergebnisdatei fuer Berichte
# =============================================================================
import json
import os
from datetime import datetime


def make_run_dir(base="tdem_results"):
    """Erstellt einen zeitstempel-benannten Ordner fuer einen Lauf.

    Beispiel: tdem_results/20260920_174500/
    Alle Ergebnisse eines Laufes (JSON, Grafiken, Checkpoints, Trainingsdaten)
    landen dort und werden nie ueberschrieben.

    Rueckgabe:
        Pfad des Laufordners
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Laufordner: {run_dir}")
    return run_dir


def save_results(results, output_file="tdem_results.json"):
    """Speichert die wichtigsten Ergebnisse als JSON-Datei fuer Berichte.

    Parameter:
        results: Dictionary mit Ergebniswerten (siehe run_pipeline)
        output_file: Dateiname der JSON-Ausgabedatei
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Ergebnisse gespeichert: {output_file}")


# =============================================================================
# VISUALISIERUNG: Vergleich wahres Modell / Surrogate-Inversion
# =============================================================================

def _model_to_depth_profile(sigma_model, thicknesses, z_max=None):
    """Wandelt Schichtmodell in Punktlisten fuer ein Stufen-Tiefenprofil um.

    Rueckgabe:
        z_vals: Tiefen der Polygonpunkte (negativ = nach unten)
        s_vals: Leitfaehigkeit an diesen Punkten
    """
    # Schichtgrenzen: 0, Summen der Dicken, Abschluss mit z_max
    boundaries = np.concatenate((
        [0.0], np.cumsum(thicknesses), [np.inf]
    ))
    if z_max is None:
        z_max = boundaries[-2] + 50.0
    boundaries[-1] = z_max

    # Pro Schicht zwei Punkte (Ober-/Unterkante) -> exakte Stufengeometrie
    n_layer = len(sigma_model)
    z_vals, s_vals = [], []
    for i in range(n_layer):
        z_vals.extend([-boundaries[i], -boundaries[i + 1]])
        s_vals.extend([sigma_model[i], sigma_model[i]])
    return np.array(z_vals), np.array(s_vals)


def plot_results(true_model, true_thicknesses, inverted_model,
                 inverted_thicknesses,
                 observed_data, surrogate_prediction, times,
                 train_losses=None, rmse=None,
                 output_prefix="tdem"):
    """Erstellt Vergleichsgrafiken fuer Berichte.

    Erzeugt drei Abbildungen in einer Figur:
      (1) Trainingsverlust des Surrogates
      (2) Wahres vs. invertiertes Leitfaehigkeitsmodell (Tiefenprofil)
      (3) Beobachtete Daten vs. Surrogate-Vorhersage (log-log)

    Speichert die Figur als PNG und PDF und zeigt sie an.

    Parameter:
        true_model: Wahres Leitfaehigkeitsmodell (S/m)
        true_thicknesses: Wahre Schichtdicken (m)
        inverted_model: Invertiertes Leitfaehigkeitsmodell (S/m)
        observed_data: Beobachtete / wahre EM-Antwort (dB/dt)
        surrogate_prediction: Surrogate-Vorhersage fuer invertiertes Modell
        times: Zeitkanaele (s)
        train_losses: Verlustverlauf (optional, None = Plot wird ausgelassen)
        rmse: Surrogate-RMSE im log10-Raum (optional, fuer Titel)
        output_prefix: Dateinamen-Praefix fuer PNG/PDF
    """
    n_panels = 3 if train_losses is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    axes = np.atleast_1d(axes)

    title_suffix = f" (RMSE log10: {rmse:.4f})" if rmse is not None else ""

    panel = 0

    # --- (1) Trainingsverlust ---
    if train_losses is not None:
        ax = axes[panel]
        ax.plot(train_losses, color="tab:blue")
        ax.set_yscale("log")
        ax.set_xlabel("Epoche")
        ax.set_ylabel("Loss (MSE, log10-Raum)")
        ax.set_title("Surrogate-Training" + title_suffix)
        ax.grid(True, alpha=0.3)
        panel += 1

    # --- (2) Modellvergleich: Tiefenprofil ---
    ax = axes[panel]
    z_true, s_true = _model_to_depth_profile(true_model, true_thicknesses)
    # Fuer invertiertes Modell: gleiche Schichtgrenzen annehmen
    z_inv, s_inv = _model_to_depth_profile(inverted_model,
                                           inverted_thicknesses)

    ax.plot(s_true, z_true, label="Wahres Modell",
            color="black", linewidth=2)
    ax.plot(s_inv, z_inv, label="Inversion (Surrogate)",
            color="tab:red", linewidth=2, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"Leitfaehigkeit $\sigma$ (S/m)")
    ax.set_ylabel("Tiefe (m)")
    ax.set_title("Modellvergleich" + title_suffix)
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)
    panel += 1

    # --- (3) Datenvergleich: gemessen vs. vorhergesagt ---
    ax = axes[panel]
    ax.loglog(times, np.abs(observed_data), "o-", label="Beobachtete Daten",
              color="black", linewidth=1.5)
    ax.loglog(times, np.abs(surrogate_prediction), "s--",
              label="Surrogate-Vorhersage", color="tab:red", linewidth=1.5)
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("|dB/dt| (T/s)")
    ax.set_title("Datenvergleich" + title_suffix)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{output_prefix}_results.png", dpi=300)
    fig.savefig(f"{output_prefix}_results.pdf")
    print(f"Grafiken gespeichert: {output_prefix}_results.png / .pdf")
    plt.show()


# =============================================================================
# MAIN: Beispiel-Aufruf fuer alle Phasen
# =============================================================================
# Zwei Modi:
#   DEV_MODE = True  -> run_quick_test():  ca. 5 min auf RTX 4070 Mobile
#   DEV_MODE = False -> run_full():        Produktionslauf (Cloud-GPU empfohlen)
# =============================================================================

def make_test_model(thicknesses, seed=None):
    """Erzeugt ein 1D-Proxy-Testmodell fuer steilstehende Graphitfloeze.

    Nebengestein: 0.001 - 0.1 S/m (widerstaendig)
    Graphitzone:  1.0 - 10.0 S/m (leitfaehig, starker Kontrast)

    Varianz: 1 oder 2 leitfaehige Zonen auf 1-3 benachbarten Rasterschichten,
    in zufaelliger Tiefe (flach bis tief, alle Tiefenbereiche).

    Rueckgabe:
        sigma_model: Leitfaehigkeiten (12 Rasterschichten + Halbraum)
    """
    rng = np.random.default_rng(seed)
    n_layer = len(thicknesses) + 1  # inkl. Halbraum

    # Nebengestein
    sigma_background = 10 ** rng.uniform(-3, -1)
    sigma_model = np.full(n_layer, sigma_background)

    # Halbraum = Nebengestein (Graphitfloez reicht "sehr tief", aber 1D-Proxy:
    # leitfaehige Zonen nur in endlichen Schichten)
    # 1 oder 2 leitfaehige Zonen
    n_zones = rng.integers(1, 3)  # 1 oder 2
    used = set()
    for _ in range(n_zones):
        zone_len = int(rng.integers(1, 4))  # 1-3 Schichten pro Zone
        start = int(rng.integers(0, len(thicknesses) - zone_len + 1))
        # Ueberlappung mit bereits belegten Schichten vermeiden
        if any(s in used for s in range(start, start + zone_len)):
            continue
        sigma_zone = 10 ** rng.uniform(0, 1)  # 1-10 S/m
        sigma_model[start:start + zone_len] = sigma_zone
        used.update(range(start, start + zone_len))

    return sigma_model


def make_background_model(thicknesses, seed=None):
    """Erzeugt ein Hintergrund-Testmodell OHNE leitfaehige Zone.

    Gegenstueck zu make_test_model fuer die Geisterfloez-Diagnose:
    Gradation oder leichte Zufallsstruktur aus dem vollen
    Hintergrundbereich (1e-3 bis 0.1 S/m), aber keine Graphit-
    Signatur. Die Multi-Test-Auswertung muss hier den invertierten
    Peak als Fehlalarm zaehlen (Geisterpeak-Rate).

    Rueckgabe:
        sigma_model: Leitfaehigkeiten (n_layer Werte)
    """
    rng = np.random.default_rng(seed)
    n_layer = len(thicknesses) + 1

    # Variante 1 (50 %): glatte Gradation mit Zufallsknick
    # Variante 2 (50 %): unabhaengige Schichtwerte im Hintergrundbereich
    if rng.uniform() < 0.5:
        sigma_top = 10 ** rng.uniform(-3, -1)
        sigma_bottom = 10 ** rng.uniform(-3, -1)
        sigma_model = np.logspace(np.log10(sigma_top),
                                  np.log10(sigma_bottom), n_layer)
    else:
        sigma_model = 10 ** rng.uniform(-3, -1, n_layer)
    return sigma_model


def add_receiver_noise(responses, times, n_stacks=32,
                        noise_density_pt=20.0, sensor_bw_hz=None,
                        gain_error=0.002, seed=None):
    """Fuegt Empfaengerrauschen auf dB/dt-Daten hinzu (FGM3D/75-Modell).

    Physikalisches Modell (Fluxgate misst absolutes B-Feld):
        - Der FGM3D misst B inkl. Erdmagnetfeld (~50.000 nT); das TDEM-
          Sekundaersignal wird durch Differenzbildung gegen den stabilen
          Untergrund-Level gewonnen. Offset-Drift und DC-Feld fallen dabei
          heraus -> KEIN signalproportionaler Zufallsterm.
        - Relevant ist das absolute Sensorrauschen im Tor-Bandpass:
          Rauschdichte ~20 pT/sqrt(Hz) (FGM3D Standard).
        - Torbreite = t_i / 4 (log-gated); effektive Bandbreite
          min(1/(2*Torbreite), Sensorbandbreite).
        - Rauschen pro Tor = Dichte * sqrt(Bandbreite) / sqrt(n_stacks).
        - dB/dt-Rauschen = B-Rauschen / Torbreite.
        - Zusaetzlich ein SYSTEMATISCHER Kalibrierfehler der Gesamt-
          empfindlichkeit (0.2%), pro Sondierung konstant (kein Rauschen):
          wird als fester Skalenfaktor auf alle Kanaele angewendet.

    Parameter:
        responses: (n_times,) oder (n, n_times) dB/dt-Daten
        times: Zeitkanaele
        n_stacks: Anzahl Stackings
        noise_density_pt: Rauschdichte in pT/sqrt(Hz)
        sensor_bw_hz: Sensorbandbreite in Hz (FGM3D: 2-6 kHz)
        gain_error: systematische Kalibrierunsicherheit (relativ)
        seed: Zufallsseed (reproduzierbar)

    Rueckgabe:
        Verrauschte Kopie von responses
    """
    rng = np.random.default_rng(seed)
    responses = np.atleast_2d(np.asarray(responses, dtype=np.float64))
    times = np.asarray(times, dtype=np.float64)

    # Torbreiten und effektive Bandbreiten pro Kanal
    # sensor_bw_hz=None -> globale Konstante SENSOR_BW_HZ (4 kHz)
    if sensor_bw_hz is None:
        sensor_bw_hz = SENSOR_BW_HZ
    gate = times / 4.0
    bw = np.minimum(1.0 / (2.0 * gate), sensor_bw_hz)

    # B-Rauschen pro Tor nach Stacking (pT -> T)
    sigma_b = (noise_density_pt * 1e-12) * np.sqrt(bw) / np.sqrt(n_stacks)

    # dB/dt-Rauschen pro Kanal
    noise_dt = sigma_b / gate

    # Zufallsrauschen (absolut, kanalabhaengig)
    noise = rng.normal(0.0, 1.0, responses.shape)
    noisy = responses + noise * noise_dt[None, :]

    # Systematischer Kalibrierfehler: fester Faktor pro Sondierung
    scale = 1.0 + rng.normal(0.0, gain_error)
    noisy = noisy * scale

    return noisy if noisy.shape[0] > 1 else noisy[0]


def compute_noise_dt(times, n_stacks=32, noise_density_pt=20.0,
                     sensor_bw_hz=None):
    """dB/dt-Rauschen (1 Sigma) pro Kanal fuer das FGM3D/75-Modell.

    Identische Rechnung wie in add_receiver_noise, getrennt exportiert,
    damit die Inversion die Kanalgewichtung daraus ableiten kann.
    sensor_bw_hz=None -> globale Konstante SENSOR_BW_HZ (4000 Hz,
    Geraeteversion mit 4-kHz-Cut-off; Normalversion: 2000 Hz).
    """
    if sensor_bw_hz is None:
        sensor_bw_hz = SENSOR_BW_HZ
    gate = np.asarray(times) / 4.0
    if sensor_bw_hz is None:
        sensor_bw_hz = SENSOR_BW_HZ
    bw = np.minimum(1.0 / (2.0 * gate), sensor_bw_hz)
    sigma_b = (noise_density_pt * 1e-12) * np.sqrt(bw) / np.sqrt(n_stacks)
    return sigma_b / gate


def run_sensitivity_suite(device="cuda", output_prefix="sensitivity",
                          cfg=None):
    """Sensitivitaetsstudie: Strom, Radius, Floez- und Nebengesteins-
    parameter, Stationsraster (one-at-a-time um eine Basiskonfiguration).

    Basis: radius=20 m, current=5 A, sigma_zone=1 S/m,
           zone_thickness=2.5 m, sigma_background=0.01 S/m,
           dip=45 Grad, 20 Stationen im 2-m-Abstand, 1024 Stacks.

    Sweeps:
        1. Sendestrom I in {1, 5, 20} A
        2. Schleifenradius R in {10, 30, 100} m
        3. Floezleitfaehigkeit in {0.5, 1, 10} S/m
        4. Floezmaechtigkeit in {0.5, 1, 2.5, 5} m
        5. Nebengesteinsleitfaehigkeit in {0.001, 0.01, 0.1} S/m

    Strom und Radius werden NICHT als eigene Surrogates trainiert:
    es wird ein einziges konditioniertes Surrogate gelernt, das
    log10(radius) und log10(current) als zusaetzliche Netzeingaben
    erhoeht (Trainingsbereich: R in 10-100 m, I in 1-20 A, je Probe
    log-uniform gezogen). Alle Sweeps verwenden dasselbe Netz.

    Kennzahlen je Konfiguration (JSON + Konsole):
        - mittlere/max. Tiefenabweichung der Floezverfolgung
        - Detektionsquote (invertierter Peak > 3x Hintergrundsigma
          und innerhalb der wahren Zone +-1 Schicht)
        - mittlere Peak-Leitfaehigkeit ueber alle Stationen
    """
    if cfg is None:
        cfg = DEV_CONFIG
    n_stacks = cfg.get("n_stacks", 1024)
    thicknesses = make_fixed_layer_mesh(
        z_multiplier=cfg["z_multiplier"],
        first_layer=cfg["first_layer"],
        z_max=cfg["z_max"],
    )
    times = np.logspace(-5, np.log10(0.02), cfg["n_times"])
    src_location = np.array([0.0, 0.0, 0.5])
    rx_location = np.array([0.0, 0.0, 0.5])
    n_layer = len(thicknesses) + 1
    boundaries = np.concatenate(([0.0], np.cumsum(thicknesses)))

    # Basis-Geometrie des Floezes
    BASE = dict(dip_deg=45.0, zone_thickness=2.5, sigma_zone=1.0,
                sigma_background=0.01, z_top=2.0)

    # --- Ein einziges konditioniertes Surrogate (radius & current als
    #     Netzeingaben); deckt alle Sweeps ab: R in [10, 100] m,
    #     I in [1, 20] A, je Probe log-uniform gezogen.
    print("[Surrogate-Training: parametrisiert (R in 10-100 m, "
          "I in 1-20 A) ...]")
    models, responses, radii, currents = generate_training_data_1d(
        n_samples=cfg["n_samples"], times=times,
        src_location=src_location, rx_location=rx_location,
        thicknesses=thicknesses,
        sigma_range=cfg["sigma_range"],
        radius=cfg["radius"], current=cfg["current"],
        n_workers=cfg["n_workers"],
        fractions=(cfg.get("background_fraction", 0.75),
                   cfg.get("floez_fraction", 0.15),
                   cfg.get("hard_fraction", 0.10)),
    )
    model, _, norm_stats = train_surrogate_mlp(
        models=models, responses=responses,
        n_model_params=n_layer, n_time_channels=cfg["n_times"],
        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
        lr=cfg["lr"], device=device,
        hidden_dims=cfg["hidden_dims"],
        radii=radii, currents=currents,
    )
    base_m, base_ns = model, norm_stats

    def _run_config(model, norm_stats, radius, current, zone_sigma,
                    zone_thickness, sigma_bg, label, z_top=None):
        """Fuehrt den Profil-Test fuer eine Konfiguration aus und
        liefert die Kennzahlen zurueck.

        z_top: Tiefe der Floezoberkante. None -> Basiswert (2 m).
            Der fenster-faire Sweep uebergibt einen radius-abhaengigen
            Wert, damit jedes R im Zentrum seines Erkundungsfensters
            getestet wird.
        """
        if z_top is None:
            z_top = BASE["z_top"]
        inverter = SurrogateInversion(model, times, device=device,
                                      norm_stats=norm_stats,
                                      radius=radius, current=current)
        stations_x, true_cols = make_dipping_zone_columns(
            thicknesses, dip_deg=BASE["dip_deg"], n_stations=20,
            station_spacing=2.0, zone_thickness=zone_thickness,
            sigma_zone=zone_sigma, sigma_background=sigma_bg,
            z_top=z_top,
        )
        noise_dt = compute_noise_dt(times, n_stacks=n_stacks)
        inv_cols = np.zeros_like(true_cols)
        detected = 0
        for i in range(len(stations_x)):
            field_data = run_1d_tdem_forward(
                thicknesses, true_cols[i], times,
                src_location, rx_location, radius=radius,
                current=current,
            )
            field_data = add_receiver_noise(
                field_data, times, n_stacks=n_stacks, seed=7000 + i
            )
            inv_cols[i] = inverter.invert(
                observed_data=field_data,
                initial_model=np.array([1e-2] * n_layer),
                n_iterations=cfg["n_iterations"], lr=cfg["inv_lr"],
                regularization_weight=cfg["reg_weight"],
                noise_dt=noise_dt, verbose=False,
            )
            peak_l = int(np.argmax(inv_cols[i]))
            true_l = int(np.argmax(true_cols[i]))
            if (inv_cols[i][peak_l] > 3.0 * sigma_bg
                    and abs(peak_l - true_l) <= 1):
                detected += 1
        inv_peaks = boundaries[np.argmax(inv_cols, axis=1)]
        true_peaks = boundaries[np.argmax(true_cols, axis=1)]
        err = np.abs(inv_peaks - true_peaks)
        result = {
            "label": label, "radius_m": radius, "current_A": current,
            "sigma_zone": zone_sigma, "zone_thickness_m": zone_thickness,
            "sigma_background": sigma_bg, "dip_deg": BASE["dip_deg"],
            "n_stations": 20, "station_spacing_m": 2.0,
            "n_stacks": n_stacks,
            "mean_depth_err_m": float(err.mean()),
            "max_depth_err_m": float(err.max()),
            "detection_rate_percent": 100.0 * detected / len(stations_x),
            "mean_peak_sigma": float(
                np.max(inv_cols, axis=1).mean()),
        }
        print(f"  {label}: Abw. {err.mean():.1f} m (max {err.max():.1f}), "
              f"Detektion {result['detection_rate_percent']:.0f}%, "
              f"Peak-Sigma {result['mean_peak_sigma']:.2f} S/m")
        return result

    results = []

    print("\n=== Sweep 1: Sendestrom (Radius 20 m) ===")
    for current in [1.0, 5.0, 20.0]:
        results.append(_run_config(
            base_m, base_ns, 20.0, current, BASE["sigma_zone"],
            BASE["zone_thickness"], BASE["sigma_background"],
            f"I={current:.0f}A"))

    print("\n=== Sweep 2: Schleifenradius (Strom 5 A) ===")
    for radius in [10.0, 30.0, 100.0]:
        results.append(_run_config(
            base_m, base_ns, radius, 5.0, BASE["sigma_zone"],
            BASE["zone_thickness"], BASE["sigma_background"],
            f"R={radius:.0f}m"))

    print("\n=== Sweep 3: Floezleitfaehigkeit (R=20 m, I=5 A) ===")
    for zone_sigma in [0.5, 1.0, 10.0]:
        results.append(_run_config(
            base_m, base_ns, 20.0, 5.0, zone_sigma,
            BASE["zone_thickness"], BASE["sigma_background"],
            f"sigma_zone={zone_sigma}S/m"))

    print("\n=== Sweep 4: Floezmaechtigkeit (R=20 m, I=5 A) ===")
    for zone_thickness in [0.5, 1.0, 2.5, 5.0]:
        results.append(_run_config(
            base_m, base_ns, 20.0, 5.0, BASE["sigma_zone"],
            zone_thickness, BASE["sigma_background"],
            f"t={zone_thickness}m"))

    print("\n=== Sweep 5: Nebengestein (R=20 m, I=5 A) ===")
    for sigma_bg in [0.001, 0.01, 0.1]:
        results.append(_run_config(
            base_m, base_ns, 20.0, 5.0, BASE["sigma_zone"],
            BASE["zone_thickness"], sigma_bg,
            f"sigma_bg={sigma_bg}S/m"))

    # --- Sweep 6: fenster-fair ---
    # Jeder Radius wird gegen ein Floez getestet, das im Zentrum
    # seines Erkundungsfensters liegt (Faustregel ~0.3 * R fuer die
    # Floezoberkante; das Fenster selbst reicht grob bis ~1 * R).
    # Damit misst der Sweep die Faehigkeit im jeweils gueltigen
    # Regime statt der Fehlanpassung Fenster vs. Ziel.
    print("\n=== Sweep 6: Schleifenradius fenster-fair "
          "(Floezteufe skaliert mit R) ===")
    for radius, z_top_ff in [(10.0, 3.0), (30.0, 10.0), (100.0, 50.0)]:
        results.append(_run_config(
            base_m, base_ns, radius, 5.0, BASE["sigma_zone"],
            BASE["zone_thickness"], BASE["sigma_background"],
            f"R={radius:.0f}m fenster-fair (z_top={z_top_ff:.0f}m)",
            z_top=z_top_ff,
        ))
    # Referenz: kleines R gegen tiefes Floez (Erwartung: schlecht)
    results.append(_run_config(
        base_m, base_ns, 20.0, 5.0, BASE["sigma_zone"],
        BASE["zone_thickness"], BASE["sigma_background"],
        "R=20m Referenz (z_top=50m)", z_top=50.0,
    ))

    with open(f"{output_prefix}_results.json", "w") as f:
        json.dump({"base": BASE, "results": results}, f, indent=2)
    print(f"\nSensitivitaetsstudie gespeichert: "
          f"{output_prefix}_results.json")
    return results


def run_multi_test(model, norm_stats, thicknesses, times,
                   src_location, rx_location, radius, current,
                   device="cuda", n_tests=20, seed_start=100,
                   n_iterations=500, lr=0.05, regularization_weight=0.002,
                   noise=True, n_stacks=32, output_file=None,
                   n_restarts=1, n_background_tests=100,
                   plot_file=None):
    """Mehrfach-Test: n_tests Flöz- + n_background_tests Hintergrundmodelle.

    fuer jedes Testmodell:
        1. make_test_model mit eigenem Seed (Tiefen-/Leitfaehigkeitsvarianz)
        1b. zusaetzlich n_background_tests Hintergrundmodelle
            (make_background_model, OHNE Zone) - Geisterfloez-Diagnose:
            hier ist ein "Treffer" des invertierten Peaks ein
            FEHLALARM (false positive) und wird separat gezaehlt.
        2. Synthetische Daten + FGM3D/75-Rauschen
        3. Inversion
        4. Automatische Auswertung (Zonen-Lokalisierung, Kontrast)

    Rueckgabe:
        Zusammenfassungsdict, zusätzlich als JSON gespeichert
    """
    inverter = SurrogateInversion(model, times, device=device,
                                  norm_stats=norm_stats,
                                  radius=radius, current=current)
    n_layer = len(thicknesses) + 1
    boundaries = np.concatenate(([0.0], np.cumsum(thicknesses)))

    results = []
    # Hintergrundmodelle zuerst (eigene Seed-Serie, keine Kollision
    # mit den Floez-Seeds der Referenz-Inversion)
    for k in range(n_background_tests):
        seed = seed_start + 500 + k
        true_model = make_background_model(thicknesses, seed=seed)

        field_data = run_1d_tdem_forward(
            thicknesses, true_model, times,
            src_location, rx_location, radius=radius, current=current,
        )
        if noise:
            field_data = add_receiver_noise(
                field_data, times, n_stacks=n_stacks, seed=seed + 10000
            )
            noise_dt = compute_noise_dt(times, n_stacks=n_stacks)
        else:
            noise_dt = None

        initial_model = np.array([1e-2] * n_layer)
        inv_model = inverter.invert(
            observed_data=field_data,
            initial_model=initial_model,
            n_iterations=n_iterations,
            lr=lr,
            regularization_weight=regularization_weight,
            noise_dt=noise_dt,
            verbose=False,
            n_restarts=n_restarts, seed=seed,
        )
        peak_layer = int(np.argmax(inv_model))
        peak_sigma = float(inv_model[peak_layer])
        # Rekonstruktionsgüte: Median-Faktor über das Gesamtraster
        log_err = np.abs(np.log10(inv_model) - np.log10(true_model))
        median_factor = float(10 ** np.median(log_err))
        # Geisterfloez: leitfaehiger Peak OHNE wahre Zone.
        # Fehlalarm, wenn invertierter Peak deutlich ueber dem
        # wahren Hintergrundmaximum liegt (Faktor 5).
        true_max = float(np.max(true_model))
        false_positive = peak_sigma > 5.0 * true_max

        results.append({
            "seed": seed,
            "model_class": "background",
            "true_peak_sigma": true_max,
            "inv_peak_layer": peak_layer,
            "inv_peak_depth_m": float(boundaries[peak_layer])
            if peak_layer < len(boundaries) else float(boundaries[-1]),
            "inv_peak_sigma": peak_sigma,
            "median_log_err": float(np.median(log_err)),
            "median_factor": median_factor,
            "inv_model": inv_model.tolist(),
            "ghost_floez_false_positive": bool(false_positive),
        })
        # Fortschrittsanzeige nur jeden 10. (bei 100 Modellen)
        if (k + 1) % 10 == 0 or k == n_background_tests - 1:
            print(f"BG {k+1}/{n_background_tests}: Median-Faktor "
                  f"{median_factor:.2f} "
                  f"({'GEIST' if false_positive else 'ok'})")

    for k in range(n_tests):
        seed = seed_start + k
        true_model = make_test_model(thicknesses, seed=seed)

        # Zone(n) im wahren Modell identifizieren (Schichten > 1 S/m)
        zone_mask = true_model > 1.0
        zone_layers = np.where(zone_mask)[0]
        zone_depth = boundaries[zone_layers[0]] if zone_mask.any() \
            else None

        # Synthetische Daten + optional Rauschen
        field_data = run_1d_tdem_forward(
            thicknesses, true_model, times,
            src_location, rx_location, radius=radius, current=current,
        )
        if noise:
            field_data = add_receiver_noise(
                field_data, times, n_stacks=n_stacks, seed=seed + 10000
            )
            # Kanalgewichte aus demselben Rauschmodell (Chi-Quadrat-Misfit)
            noise_dt = compute_noise_dt(times, n_stacks=n_stacks)
        else:
            noise_dt = None

        # Inversion (Startmodell: widerstaendiger Halbraum)
        # Multi-Start: n_restarts-1 zusaetzliche Zufallsstartmodelle;
        # bestes Modell nach Misfit, Ensemble fuer Unsicherheit.
        initial_model = np.array([1e-2] * n_layer)
        inv_model = inverter.invert(
            observed_data=field_data,
            initial_model=initial_model,
            n_iterations=n_iterations,
            lr=lr,
            regularization_weight=regularization_weight,
            noise_dt=noise_dt,
            verbose=False,
            n_restarts=n_restarts, seed=seed,
        )
        ens = inverter.last_ensemble
        ens_median = np.median(ens, axis=0)
        ens_p16 = np.percentile(ens, 16, axis=0)
        ens_p84 = np.percentile(ens, 84, axis=0)

        # Auswertung: Leitfaehigkeitsmaximum = lokalisierte Zone
        peak_layer = int(np.argmax(inv_model))
        peak_depth = boundaries[peak_layer] if peak_layer < len(boundaries) \
            else boundaries[-1]
        peak_sigma = inv_model[peak_layer]

        # Treffer: globales Kriterium (Peak vs. Peak)
        true_peak_layer = int(np.argmax(true_model))
        hit_global = abs(peak_layer - true_peak_layer) <= 1
        # any-zone-Kriterium: irgendeine wahre Zone getroffen?
        hit_any = bool(any(
            abs(peak_layer - zl) <= 1 for zl in zone_layers
        ))

        # --- Rekonstruktionsgüte über das GESAMTE Profil (Hauptkennzahl,
        # seit Umstellung des Projektkriteriums auf Untergrundtreue) ---
        # Median-Log-Fehler = symmetrischer Faktorfehler über alle
        # Schichten; 10**Wert = typische Abweichung als Faktor.
        log_err = np.abs(np.log10(inv_model) - np.log10(true_model))
        median_factor = float(10 ** np.median(log_err))

        results.append({
            "seed": seed,
            "model_class": "floez",
            "true_zone_layers": zone_layers.tolist(),
            "true_zone_depth_m": zone_depth,
            "true_peak_sigma": float(true_model[true_peak_layer]),
            "inv_peak_layer": peak_layer,
            "inv_peak_depth_m": float(peak_depth),
            "inv_peak_sigma": float(peak_sigma),
            "hit_within_1_layer": bool(hit_global),
            "hit_any_zone": hit_any,
            # Hauptkennzahl Rekonstruktion (Profil über alle Schichten)
            "median_log_err": float(np.median(log_err)),
            "median_factor": median_factor,
            "inv_model": inv_model.tolist(),
            # Ensemble-Streuung der Peak-Leitfaehigkeit (1-Sigma-Band
            # aus P16/P84) als Unsicherheitsproxy
            "ens_peak_sigma_p16": float(np.percentile(
                np.max(ens, axis=1), 16)),
            "ens_peak_sigma_p84": float(np.percentile(
                np.max(ens, axis=1), 84)),
        })
        print(f"{k+1:2d}/20: {zone_layers.tolist()} -> "
              f"{peak_layer} {'OK' if hit_any else '--'}")

    # --- Zusammenfassung mit any-zone-Quote und Tiefengruppen ---
    # Flöz- und Hintergrundmodelle getrennt auswerten
    floez_results = [r for r in results
                     if r.get("model_class", "floez") == "floez"]
    bg_results = [r for r in results
                  if r.get("model_class") == "background"]
    hits_global = [r["hit_within_1_layer"] for r in floez_results]
    hits_any = [r["hit_any_zone"] for r in floez_results]
    ghost_rate = (100.0
                  * sum(r["ghost_floez_false_positive"] for r in bg_results)
                  / len(bg_results)) if bg_results else None

    # Tiefengruppen nach oberster wahrer Zonenschicht
    depth_groups = {}
    for r in floez_results:
        first_zone = r["true_zone_layers"][0]
        if first_zone <= 4:
            g = "flach_0bis8m"
        elif first_zone <= 8:
            g = "mittel_8bis49m"
        else:
            g = "tief_49m_plus"
        depth_groups.setdefault(g, []).append(r)

    group_stats = {}
    for g, rs in depth_groups.items():
        group_stats[g] = {
            "n_tests": len(rs),
            "hit_rate_global_percent": 100.0
            * sum(r["hit_within_1_layer"] for r in rs) / len(rs),
            "hit_rate_any_zone_percent": 100.0
            * sum(r["hit_any_zone"] for r in rs) / len(rs),
        }

    # Verteilung der invertierten Maxima (Bias-Diagnose, nur Flöze)
    peak_counts = {}
    for r in floez_results:
        peak_counts[r["inv_peak_layer"]] = \
            peak_counts.get(r["inv_peak_layer"], 0) + 1

    summary = {
        "n_tests": n_tests,
        "n_background_tests": len(bg_results),
        "ghost_floez_false_positive_percent": ghost_rate,
        # --- Hauptkennzahlen Rekonstruktionsgüte (Projektkriterium:
        #     Untergrundtreue, nicht Flözerkennung) ---
        "reconstruction_median_factor": {
            "floez_median": float(np.median(
                [r["median_factor"] for r in floez_results]))
            if floez_results else None,
            "floez_p16": float(np.percentile(
                [r["median_factor"] for r in floez_results], 16))
            if floez_results else None,
            "floez_p84": float(np.percentile(
                [r["median_factor"] for r in floez_results], 84))
            if floez_results else None,
            "background_median": float(np.median(
                [r["median_factor"] for r in bg_results]))
            if bg_results else None,
            "background_p16": float(np.percentile(
                [r["median_factor"] for r in bg_results], 16))
            if bg_results else None,
            "background_p84": float(np.percentile(
                [r["median_factor"] for r in bg_results], 84))
            if bg_results else None,
        },
        "noise": noise,
        "n_stacks": n_stacks,
        "n_iterations": n_iterations,
        "hit_rate_global_percent": 100.0 * sum(hits_global) / len(hits_global),
        "hit_rate_any_zone_percent": 100.0 * sum(hits_any) / len(hits_any),
        "depth_groups": group_stats,
        "inv_peak_layer_distribution": peak_counts,
        "n_stacks": n_stacks,
        "sensor": f"Sensys FGM3D/75 (20 pT/sqrt(Hz), {n_stacks} Stacks, "
                  "0.2% systematische Kalibrierunsicherheit)",
        "details": results,
    }

    if output_file:
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"Trefferquote global: {summary['hit_rate_global_percent']:.0f}% "
          f"| any-zone: {summary['hit_rate_any_zone_percent']:.0f}% "
          f"| Geisterfloez-Rate: {ghost_rate:.0f}% "
          f"({len(bg_results)} Hintergrundmodelle)"
          f" | Maxima: {peak_counts}")
    # Hauptkennzahl: Rekonstruktionsgüte (Median-Faktor mit Bändern)
    rc = summary["reconstruction_median_factor"]
    if floez_results:
        print(f"Rekonstruktion Flloeze:    Median-Faktor "
              f"{rc['floez_median']:.2f} "
              f"[{rc['floez_p16']:.2f}-{rc['floez_p84']:.2f}]")
    if bg_results:
        print(f"Rekonstruktion Hintergrund: Median-Faktor "
              f"{rc['background_median']:.2f} "
              f"[{rc['background_p16']:.2f}-{rc['background_p84']:.2f}]")

    # --- Visualisierung: Multi-Test-Grafik ---
    # Panel 1: Flloezmodelle - wahre gegen invertierte Peak-Tiefe
    #          (alle 20 Punkte; Diagonale = perfekt, +-1 Schicht Band)
    # Panel 2: Hintergrundmodelle - wahres gegen invertiertes Profil
    #          (Geisterfloez-Diagnose: Abweichung nach rechts = Uebertreibung)
    if plot_file:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2,
                                           figsize=(14, 6))
            # Panel 1: Peak-Tiefen-Vergleich (Flloeze)
            for r in floez_results:
                hit = r["hit_any_zone"]
                ax1.plot(r["true_zone_depth_m"], r["inv_peak_depth_m"],
                         "g^" if hit else "rx", markersize=9,
                         label="_nolegend_")
            lims = [0, max(boundaries[-1], 1)]
            ax1.plot(lims, lims, "k--", alpha=0.5, label="ideal")
            # +-1-Schicht-Toleranzband (grob: 50 % Tiefe)
            ax1.fill_between(lims, [0, lims[1] * 0.5],
                             [lims[1] * 0.5, lims[1]],
                             color="g", alpha=0.08,
                             label="Toleranzband (ca. +-1 Schicht)")
            ax1.set_xlabel("wahre Floeztiefe (m)")
            ax1.set_ylabel("invertierte Peak-Tiefe (m)")
            ax1.set_title(f"Floeze: {summary['hit_rate_global_percent']:.0f}%"
                          f" global / "
                          f"{summary['hit_rate_any_zone_percent']:.0f}%"
                          f" any-zone (n={len(floez_results)})")
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale("symlog")
            ax1.set_yscale("symlog")

            # Panel 2: Hintergrundmodelle (Geisterfloez)
            if bg_results:
                n_bg = len(bg_results)
                for i, r in enumerate(bg_results):
                    inv = np.array(r.get("inv_model", []))
                    if inv.size == 0:
                        continue
                    ax2.step(inv, boundaries, where="post",
                             color=plt.cm.Reds(0.3 + 0.7 * i / n_bg),
                             label=f"inv seed {r['seed']}"
                                   f"{' GEIST' if r['ghost_floez_false_positive'] else ''}")
                ax2.set_xscale("log")
                ax2.set_xlabel("Leitfaehigkeit (S/m)")
                ax2.set_ylabel("Tiefe (m)")
                ax2.invert_yaxis()
                ax2.set_title(f"Hintergrundmodelle: Geisterfloez-Rate "
                              f"{ghost_rate:.0f}% (n={n_bg})")
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_file, dpi=200)
            plt.savefig(plot_file.replace(".png", ".pdf"))
            print(f"Multi-Test-Grafik gespeichert: {plot_file}")
        except Exception as e:
            print(f"Multi-Test-Grafik fehlgeschlagen: {e}")
    return summary


def make_dipping_zone_columns(thicknesses, dip_deg=90.0, n_stations=10,
                              station_spacing=5.0, zone_thickness=3.0,
                              sigma_zone=5.0, sigma_background=0.02,
                              z_top=2.0, x_dip_dir=1.0):
    """Erzeugt 1D-Saeulen fuer ein einfallendes Leitfaehigkeitsfloez.

    Das Floez wird als schmale, geneigte Ebene modelliert (2D-Proxy):
    an jeder Station x_i liegt die Floezmitte in der Tiefe
        z_c(x_i) = z_top + x_dip_dir * (i * spacing) * tan(dip_deg)
    Bei dip=0 (horizontal) bleibt das Floez auf z_top.
    Alle Rasterschichten, deren Mitte in [z_c - t/2, z_c + t/2] liegt,
    erhalten sigma_zone (Scheinfloeze -> meist 1-2 Schichten pro Saeule).

    Parameter:
        thicknesses: Feste Schichtdicken aus make_fixed_layer_mesh
        dip_deg: Einfallen des Floezes in Grad (0 = saigerecht liegend,
            90 = saiger/steilstehend). Konvention Geologie: 0 = horizontal.
        n_stations: Anzahl Sondierungspunkte
        station_spacing: Abstand der Sondierungspunkte in m
        zone_thickness: maechtigkeit des Floezes in m (quer zum Einfallen)
        sigma_zone: Leitfaehigkeit des Floezes (S/m)
        sigma_background: Leitfaehigkeit des Nebengesteins (S/m)
        z_top: Tiefe der Floezoberkante unter Station 0 (m)
        x_dip_dir: +1 = Floez faellt zu steigenden Stationsnummern ein

    Rueckgabe:
        stations_x: (n_stations,) Stationspositionen in m
        true_columns: (n_stations, n_layer) Leitfaehigkeitsmodelle je Station
    """
    boundaries = np.concatenate(([0.0], np.cumsum(thicknesses)))
    layer_centers = 0.5 * (boundaries[:-1] + boundaries[1:])
    n_layer = len(thicknesses) + 1

    # Geologisches Einfallen -> mathematischer Winkel
    # dip=0 (horizontal): keine Tiefenaenderung; dip nahe 90 (saiger):
    # tan -> unendlich, deshalb Spezialfall: saiger stehendes Floez
    # erscheint in der 1D-Saeule als Leiter, der von der Oberkante
    # abwaerts durch ALLE Tiefen reicht (vertikaler "Dike").
    vertical = dip_deg >= 89.0
    slope = np.tan(np.radians(min(dip_deg, 89.0)))

    stations_x = np.arange(n_stations) * station_spacing
    true_columns = np.full((n_stations, n_layer), sigma_background)
    for i, x in enumerate(stations_x):
        z_c = z_top + x_dip_dir * x * slope
        lo, hi = z_c - zone_thickness / 2.0, z_c + zone_thickness / 2.0
        if vertical:
            # Saiger: alle Rasterschichten unterhalb der Floezoberkante
            # leitfaehig (1D-Sicht auf eine vertikale Leiterebene)
            mask = layer_centers >= z_top - zone_thickness / 2.0
            true_columns[i, :len(layer_centers)] = np.where(
                mask, sigma_zone, sigma_background
            )
            true_columns[i, -1] = sigma_zone  # Halbraum mit leitfaehig
        else:
            # Gewichtete Zuordnung: Ueberlappungsanteil des Floez-
            # intervalls [lo, hi] mit jeder Schicht bestimmt deren
            # Leitfaehigkeit (Aequivalenz-Mittelung). Verhindert
            # "Loecher", wenn das Floez zwischen Schichtmitten faellt
            # (bei duennem Floez und dicken Tiefenschichten).
            for j in range(len(thicknesses)):
                z0, z1 = boundaries[j], boundaries[j + 1]
                overlap = max(0.0, min(hi, z1) - max(lo, z0))
                frac = overlap / (z1 - z0)
                if frac > 0.0:
                    # arithmetische Mittelung in sigma (Leitfaehigkeits-
                    # Mittel), ueberlappt der Graphitanteil die ganze
                    # Schicht: frac=1 -> sigma_zone
                    true_columns[i, j] = (
                        frac * sigma_zone
                        + (1.0 - frac) * sigma_background
                    )
            # Floez liegt vollstaendig unterhalb des Rasters:
            # Halbraum mit reduzierter Aequivalenzleitfaehigkeit
            if lo > boundaries[-1]:
                true_columns[i, -1] = sigma_zone
    return stations_x, true_columns


def run_profile_test(model, norm_stats, thicknesses, times,
                     src_location, rx_location, radius, current,
                     device="cuda", dip_deg=90.0, n_stations=10,
                     station_spacing=5.0, n_iterations=500, lr=0.05,
                     regularization_weight=0.001, noise=True, n_stacks=1024,
                     output_prefix="profile", cfg=None, n_restarts=1,
                     klassisch=False, max_iter_simpeg=50):
    """Profil-Simulation: einfallendes Floez, 10 Stationen, Pseudosektion.

    Ablauf:
        1. Wahres 2D-Proxy-Modell (einfallendes Floez) in 1D-Saeulen
        2. Je Station: Vorwaertsmodellierung + FGM3D/75-Rauschen
        3. Je Station: 1D-Inversion (unabhaengig, wie im Feld)
        4. Pseudosektion: Stationsprofil x Tiefe, log10(sigma)

    Rueckgabe:
        (stations_x, true_columns, inv_columns), Grafiken + JSON werden
        im Laufordner gespeichert.
    """
    inverter = SurrogateInversion(model, times, device=device,
                                  norm_stats=norm_stats,
                                  radius=radius, current=current)
    n_layer = len(thicknesses) + 1
    boundaries = np.concatenate(([0.0], np.cumsum(thicknesses)))

    # Wahres Modell: Floez mit 5 S/m in widerstaendigem Nebengestein
    stations_x, true_columns = make_dipping_zone_columns(
        thicknesses, dip_deg=dip_deg, n_stations=n_stations,
        station_spacing=station_spacing, zone_thickness=3.0,
        sigma_zone=5.0, sigma_background=0.02, z_top=2.0,
    )

    # Je Station: Daten + Rauschen + Inversion
    inv_columns = np.zeros_like(true_columns)
    # Ensemble-Statistik (Multi-Start): Median und 16/84-Perzentile
    # je Station als Unsicherheitsband der Leitfaehigkeiten
    ens_median = np.zeros_like(true_columns)
    ens_p16 = np.zeros_like(true_columns)
    ens_p84 = np.zeros_like(true_columns)
    noise_dt = compute_noise_dt(times, n_stacks=n_stacks) if noise else None
    for i in range(n_stations):
        field_data = run_1d_tdem_forward(
            thicknesses, true_columns[i], times,
            src_location, rx_location, radius=radius, current=current,
        )
        if noise:
            field_data = add_receiver_noise(
                field_data, times, n_stacks=n_stacks, seed=5000 + i
            )
        inv_columns[i] = inverter.invert(
            observed_data=field_data,
            initial_model=np.array([1e-2] * n_layer),
            n_iterations=n_iterations,
            lr=lr,
            regularization_weight=regularization_weight,
            noise_dt=noise_dt,
            verbose=False,
            n_restarts=n_restarts, seed=6000 + i,
        )
        # Ensemble-Statistik je Station (Unsicherheitsbaender)
        ens = inverter.last_ensemble
        ens_median[i] = np.median(ens, axis=0)
        ens_p16[i] = np.percentile(ens, 16, axis=0)
        ens_p84[i] = np.percentile(ens, 84, axis=0)
        print(f"Station {i+1}/{n_stations} (x={stations_x[i]:.0f} m): "
              f"Peak {inv_columns[i].max():.2f} S/m in Schicht "
              f"{int(np.argmax(inv_columns[i]))}")

    # --- Optional: klassische SimPEG-Inversion je Station ---
    # (Referenz fuer die Pseudosektion/Flözverfolgung; ca. 30 s je
    #  Station -> nur fuer Berichtslaeufe aktivieren)
    simpeg_columns = None
    if klassisch:
        print("  Klassische SimPEG-Inversion je Station ...")
        simpeg_columns = np.zeros_like(true_columns)
        for i in range(n_stations):
            field_data_i = run_1d_tdem_forward(
                thicknesses, true_columns[i], times,
                src_location, rx_location, radius=radius, current=current,
            )
            if noise:
                field_data_i = add_receiver_noise(
                    field_data_i, times, n_stacks=n_stacks,
                    seed=5000 + i,
                )
            simpeg_model_i, _ = invert_simpeg_1d(
                thicknesses, times, field_data_i,
                radius=radius, current=current,
                noise_dt=noise_dt, noise=noise,
                max_iter=max_iter_simpeg,
            )
            simpeg_columns[i] = (simpeg_model_i if simpeg_model_i
                                 is not None else np.nan)
            print(f"  Station {i+1}/{n_stations} klassisch: Peak "
                  f"{np.nanmax(simpeg_columns[i]):.2f} S/m in Schicht "
                  f"{int(np.nanargmax(simpeg_columns[i]))}")

    # --- Pseudosektion (wahr, Surrogate, optional SimPEG) ---
    panels = [(true_columns, "Wahres Modell"),
              (inv_columns, "Invertiert (1D je Station)")]
    if simpeg_columns is not None:
        panels.append((simpeg_columns, "Klassisch (SimPEG, je Station)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 8),
                             sharey=True)
    for ax, (cols, title) in zip(axes, panels):
        log_sigma = np.log10(cols)
        # Schichtgrenzen als Zeilenraster
        im = ax.pcolormesh(
            stations_x, boundaries, log_sigma.T,
            shading="auto", cmap="viridis",
            vmin=-3, vmax=1,
        )
        ax.invert_yaxis()
        ax.set_xlabel("Position (m)")
        ax.set_title(f"{title} - Floez {dip_deg:.0f} Grad einfallend")
        plt.colorbar(im, ax=ax, label="log10 sigma (S/m)")
    axes[0].set_ylabel("Tiefe (m)")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_section_{int(dip_deg)}deg.png", dpi=200)
    plt.savefig(f"{output_prefix}_section_{int(dip_deg)}deg.pdf")

    # --- Floezverfolgung: Peak-Tiefe je Station ---
    peak_layers = np.argmax(inv_columns, axis=1)
    peak_depths = boundaries[peak_layers]
    true_peaks = np.argmax(true_columns, axis=1)
    true_depths = boundaries[true_peaks]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(stations_x, true_depths, "ko-", label="wahre Floezposition")
    ax2.plot(stations_x, peak_depths, "rs--",
             label="invertierte Floezposition (Surrogate)")
    if simpeg_columns is not None:
        simpeg_peaks = np.nanargmax(simpeg_columns, axis=1)
        ax2.plot(stations_x, boundaries[simpeg_peaks], "b^--",
                 label="invertierte Floezposition (SimPEG)")
    ax2.invert_yaxis()
    ax2.set_xlabel("Position (m)")
    ax2.set_ylabel("Tiefe (m)")
    ax2.set_title(f"Floezverfolgung, {dip_deg:.0f} Grad einfallend")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_trace_{int(dip_deg)}deg.png", dpi=200)

    # JSON-Export
    results = {
        "dip_deg": dip_deg,
        "n_stations": n_stations,
        "station_spacing": station_spacing,
        "n_stacks": n_stacks if noise else None,
        "noise": noise,
        "stations_x": stations_x.tolist(),
        "layer_boundaries": boundaries.tolist(),
        "true_columns": true_columns.tolist(),
        "inv_columns": inv_columns.tolist(),
        # Ensemble-Statistik (nur bei Multi-Start aussagekraeftig;
        # bei n_restarts=1 identisch mit inv_columns)
        "ens_median": ens_median.tolist(),
        "ens_p16": ens_p16.tolist(),
        "ens_p84": ens_p84.tolist(),
        "n_restarts": n_restarts,
        "true_peak_depths": true_depths.tolist(),
        "inv_peak_depths": peak_depths.tolist(),
        "klassisch": klassisch,
        "simpeg_columns": (simpeg_columns.tolist()
                           if simpeg_columns is not None else None),
    }
    with open(f"{output_prefix}_results_{int(dip_deg)}deg.json", "w") as f:
        json.dump(results, f, indent=2)

    # Kennzahl: mittlere Tiefenabweichung der Floezposition
    depth_err = np.abs(peak_depths - true_depths)
    print(f"Floezverfolgung: mittlere Tiefenabweichung "
          f"{depth_err.mean():.1f} m, max {depth_err.max():.1f} m")
    return stations_x, true_columns, inv_columns


# =============================================================================
# PHASE 3E: Referenz-Inversion mit SimPEG (Validierung des Surrogates)
# =============================================================================
# Beantwortet die Frage: Wie viel Genauigkeit verliert man durch das
# Surrogate? Dazu wird dieselbe Sondung (a) klassisch mit SimPEG
# (L-BFGS, sensibilitaetsbasiert, teuer) und (b) mit dem Surrogate
# invertiert; die Differenz der Loesungen ist der reine Surrogate-
# fehler, gemessen am Ende der gesamten Kette - nicht nur am Forward.
# Bewusst als STANDALONE-Funktion: nicht in run_quick_test/run_full
# integriert, sondern bei Bedarf manuell aufzurufen, z. B.:
#   run_reference_inversion(model=model, norm_stats=norm_stats,
#       thicknesses=thicknesses, times=times, run_dir=run_dir)
# nach einem abgeschlossenen DEV/FULL-Lauf (model/norm_stats liegen
# dann im Namensraum der Konsole).
# =============================================================================

def invert_simpeg_1d(thicknesses, times, observed_data,
                     radius=20.0, current=5.0, noise_dt=None,
                     noise=True, max_iter=50):
    """Klassische SimPEG-Inversion einer 1D-TDEM-Sondung.

    Wird von der Referenz-Inversion UND dem Profil-Test verwendet
    (identische Einstellungen, damit beide vergleichbar bleiben).

    Rueckgabe:
        (simpeg_model, simpeg_dpred): invertierte Leitfaehigkeiten
        (n_layer,) und die dazugehoerige Vorwaertsantwort (dpred);
        (None, None), wenn die Inversion numerisch scheitert.
    """
    from simpeg import (
        data as sp_data, maps as sp_maps, regularization, data_misfit,
        optimization, inversion, inverse_problem, directives,
    )
    from discretize import TensorMesh

    src_location = np.array([0.0, 0.0, 0.5])
    rx_location = np.array([0.0, 0.0, 0.5])
    n_layer = len(thicknesses) + 1

    waveform = tdem.sources.StepOffWaveform(off_time=0.0)
    rx_list = [tdem.receivers.PointMagneticFluxTimeDerivative(
        rx_location, times, orientation="z"
    )]
    src_list = [tdem.sources.CircularLoop(
        receiver_list=rx_list, location=src_location,
        waveform=waveform, radius=radius, current=current,
    )]
    survey = tdem.Survey(src_list)
    sim = tdem.Simulation1DLayered(
        survey=survey, thicknesses=thicknesses,
        sigmaMap=sp_maps.ExpMap(nP=n_layer),
    )
    data_obj = sp_data.Data(
        survey, dobs=observed_data,
        standard_deviation=np.maximum(
            noise_dt, 1e-2 * np.abs(observed_data)
        ) if noise else 1e-4 * np.abs(observed_data),
    )
    # Tikhonov/Smoothness auf log-Sigma; echtes 1D-TensorMesh aus den
    # Schichtdicken (+ zusaetzliche unterste Zelle fuer den Halbraum)
    h = list(np.diff(np.concatenate(([0.0],
                                     np.cumsum(thicknesses)))))
    h.append(h[-1])
    mesh = TensorMesh([h])
    m_start = np.log(np.array([1e-2] * n_layer))
    reg = regularization.WeightedLeastSquares(
        mesh, mapping=sp_maps.ExpMap(nP=n_layer),
        alpha_s=1e-3, alpha_x=1.0, reference_model=np.exp(m_start),
    )
    dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=sim)
    # Beschraenkter Optimierer mit Modellgrenzen im log-Raum:
    # sigma in [1e-3, 10] (Trainingsbereich des Surrogates).
    # SimPEG bietet KEINEN L-BFGS-B; ProjectedGradient ist der
    # einzige bound-constrained Optimierer. Das fruehere Bound-
    # Zickzack (Checkerboard 10/0.001) kam vom unregularisierten
    # Regime (beta0_ratio=1e-2 ohne Schedule) - behoben durch
    # beta0=1.0 plus BetaSchedule unten, NICHT durch den Optimierer.
    opt = optimization.ProjectedGradient(
        maxIter=max_iter, maxIterLS=20,
        lower=np.log(1e-3), upper=np.log(10.0),
    )
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    # beta0=1 (statt 1e-2): geglaettet starten, dann per Schedule
    # schrittweise lockern - verhindert das Startschiessen in die
    # Bounds im unregularisierten Regime.
    directives_list = [
        directives.BetaEstimate_ByEig(beta0_ratio=1.0),
        directives.BetaSchedule(coolingFactor=2.0, coolingRate=1),
        directives.TargetMisfit(chifact=1.0),
    ]
    inv = inversion.BaseInversion(inv_prob, directives_list)
    try:
        m_rec = inv.run(m_start)
        simpeg_model = np.exp(m_rec)
        simpeg_dpred = np.asarray(sim.dpred(m_rec))
        return simpeg_model, simpeg_dpred
    except Exception as e:
        print(f"  SimPEG-Inversion fehlgeschlagen "
              f"({type(e).__name__}: {e})")
        return None, None


def run_reference_inversion(model, norm_stats, thicknesses, times,
                            radius=20.0, current=5.0, device="cuda",
                            n_reference_models=3, seed_start=100,
                            n_stacks=1024, noise=True,
                            n_iterations_surrogate=500, inv_lr=0.05,
                            reg_weight=0.001, n_restarts=1,
                            max_iter_simpeg=50, output_file=None,
                            plot=True):
    """Vergleichsinversion: klassische SimPEG- vs. Surrogate-Inversion.

    Fuer n_reference_models Testmodelle (aus make_test_model, gleiche
    Seeds wie im Multi-Test) wird jeweils:
      1. mit SimPEG eine synthetische Sondung gerechnet (Vorwaerts),
      2. darauf das Empfaengerrauschmodell gelegt (Feldsimulation),
      3. einmal klassisch invertiert: SimPEG Inversion (L-BFGS) auf
         demselben festen Schichtraster, log-mapping, Tikhonov-
         Regularisierung (Smoothness in log10-Sigma), Chi-Quadrat-
         Datenmisfit mit demselben Rauschmodell,
      4. einmal mit dem Surrogate invertiert (SurrogateInversion,
         identisches Raster, identische Rauschgewichtung),
      5. beide Loesungen gegen das wahre Modell ausgewertet.

    Kennzahlen je Modell (JSON + Konsole):
      - Tiefenlage des Leitfaehigkeitspeaks (wahr/SimPEG/Surrogate)
      - Peak-Leitfaehigkeit (wahr/SimPEG/Surrogate)
      - relativer Fehler beider Loesungen je Schicht (Median)
      - Differenz SimPEG vs. Surrogate (der reine Surrogate-Anteil)

    Hinweise:
      - Die SimPEG-Inversion ist MINUTEN schnell je Modell (nicht
        Millisekunden wie das Surrogate) - das ist ja der Punkt.
        max_iter_simpeg begrenzt die Kosten.
      - Beide Inversionen nutzen dasselbe feste Raster und dieselben
        Rauschgewichte; Unterschiede sind damit zunaechst auf den
        Forward-Ersatz (Netz vs. Physik) zurueckzufuehren, plus
        unterschiedliche Optimierer (L-BFGS vs. Adam).
      - SimPEG-Inversion via Inversion-Objekt mit directives;
        Regularisierung auf log10-Sigma (exp-mapping auf Werte um 1),
        damit positive Leitfaehigkeiten garantiert sind.

    Rueckgabe:
        summary: dict mit allen Kennzahlen (auch als JSON gespeichert)
    """
    import os
    import time
    from simpeg import (
        data as sp_data, maps as sp_maps, regularization, data_misfit,
        optimization, inversion, inverse_problem, directives,
    )

    src_location = np.array([0.0, 0.0, 0.5])
    rx_location = np.array([0.0, 0.0, 0.5])
    n_layer = len(thicknesses) + 1
    boundaries = np.concatenate(([0.0], np.cumsum(thicknesses)))
    noise_dt = compute_noise_dt(times, n_stacks=n_stacks) if noise else None

    # Surrogate-Inverter (konditioniert, falls mu_cond vorliegt)
    inverter = SurrogateInversion(
        model, times, device=device, norm_stats=norm_stats,
        radius=radius, current=current,
    )

    summary = {"n_reference_models": n_reference_models,
               "radius_m": radius, "current_A": current,
               "n_stacks": n_stacks, "noise": noise,
               "hinweis": "erste Haelfte Flloezmodelle, zweite Haelfte "
                          "Hintergrundmodelle (Geisterfloez-Diagnose)",
               "models": []}

    for k in range(n_reference_models):
        seed = seed_start + k
        # Erste Haelfte Flözmodelle, zweite Haelfte Hintergrundmodelle
        # (Geisterfloez-Diagnose auch im Dreiervergleich wahr/klassisch/
        # Surrogate). n_reference_models=3 -> 2 Flöze + 1 Hintergrund.
        n_floez = (n_reference_models + 1) // 2
        if k < n_floez:
            true_model = make_test_model(thicknesses, seed=seed)
            model_class = "floez"
        else:
            true_model = make_background_model(thicknesses, seed=seed)
            model_class = "background"
        t0 = time.time()

        # --- Synthetische Sondung + Rauschen (identisch fuer beide) ---
        field_data = run_1d_tdem_forward(
            thicknesses, true_model, times,
            src_location, rx_location, radius=radius, current=current,
        )
        observed_data = field_data.copy()
        if noise:
            observed_data = add_receiver_noise(
                observed_data, times, n_stacks=n_stacks, seed=seed + 10000
            )

        # --- 1) Klassische SimPEG-Inversion (Referenz) ---
        simpeg_model, simpeg_dpred = invert_simpeg_1d(
            thicknesses, times, observed_data,
            radius=radius, current=current,
            noise_dt=noise_dt, noise=noise,
            max_iter=max_iter_simpeg,
        )

        # --- 2) Surrogate-Inversion (identische Daten/Gewichte) ---
        inv_model = inverter.invert(
            observed_data=observed_data,
            initial_model=np.array([1e-2] * n_layer),
            n_iterations=n_iterations_surrogate, lr=inv_lr,
            regularization_weight=reg_weight,
            noise_dt=noise_dt, verbose=False,
            n_restarts=n_restarts, seed=seed,
        )
        t_el = time.time() - t0

        # --- Auswertung (Modelle + Datenkurven fuer die Grafik) ---
        true_peak_l = int(np.argmax(true_model))
        res_k = {"seed": seed, "runtime_s": round(t_el, 1),
                 "model_class": model_class}
        res_k["true_model"] = true_model.tolist()
        if simpeg_model is not None:
            res_k["simpeg_model"] = simpeg_model.tolist()
            res_k["simpeg_dpred"] = simpeg_dpred.tolist()
        res_k["surrogate_model"] = inv_model.tolist()
        res_k["observed_data"] = observed_data.tolist()
        res_k["true_dpred"] = field_data.tolist()
        res_k["surrogate_dpred"] = inverter.forward_surrogate(
            inv_model).tolist()
        res_k["true_peak_layer"] = true_peak_l
        res_k["true_peak_depth_m"] = float(boundaries[true_peak_l])
        res_k["true_peak_sigma"] = float(true_model[true_peak_l])

        for name, sol in [("simpeg", simpeg_model),
                          ("surrogate", inv_model)]:
            if sol is None:
                res_k[f"{name}_peak_layer"] = None
                res_k[f"{name}_peak_depth_m"] = None
                res_k[f"{name}_peak_sigma"] = None
                res_k[f"{name}_median_rel_err_pct"] = None
                continue
            pl = int(np.argmax(sol))
            res_k[f"{name}_peak_layer"] = pl
            res_k[f"{name}_peak_depth_m"] = float(boundaries[pl])
            res_k[f"{name}_peak_sigma"] = float(sol[pl])
            # Mittlerer relativer Fehler IM LOG-RAUM (Faktorfehler,
            # symmetrisch: Faktor 2 nach oben = Faktor 2 nach unten).
            # Fuer Hintergrundmodelle die Haupt-Kennzahl: keine Peak-
            # Lokalisierung noetig, sondern wie gut der gesammte
            # Untergrund (alle Schichten) rekonstruiert wird.
            log_err = np.abs(np.log10(sol) - np.log10(true_model))
            res_k[f"{name}_median_log_err"] = float(np.median(log_err))
            res_k[f"{name}_mean_log_err"] = float(np.mean(log_err))
            res_k[f"{name}_median_factor"] = float(
                10 ** np.median(log_err))
            res_k[f"{name}_median_rel_err_pct"] = float(np.median(
                np.abs(sol - true_model) / true_model * 100
            ))

        # Differenz SimPEG vs. Surrogate (reiner Surrogate-Anteil)
        if simpeg_model is not None:
            res_k["surrogate_vs_simpeg_peak_layer_diff"] = (
                res_k["surrogate_peak_layer"]
                - res_k["simpeg_peak_layer"]
            )
            res_k["surrogate_vs_simpeg_median_rel_diff_pct"] = float(
                np.median(np.abs(inv_model - simpeg_model)
                          / simpeg_model * 100)
            )
            # Faire Gegenueberstellung auf dem Gesamtraster:
            # Median-Log-Fehler beider Loesungen gegeneinander
            # (Wert < 1 -> Surrogate naeher am wahren Modell).
            res_k["surrogate_vs_simpeg_log_err_ratio"] = float(
                res_k["surrogate_median_log_err"]
                / res_k["simpeg_median_log_err"]
            ) if res_k.get("simpeg_median_log_err") else None
        summary["models"].append(res_k)

        # --- Konsolausgabe ---
        print(f"Modell {k+1}/{n_reference_models} (seed {seed}, "
              f"{model_class}):")
        if model_class == "floez":
            print(f"  wahr:      Peak {res_k['true_peak_sigma']:.2f} S/m "
                  f"in Schicht {true_peak_l} "
                  f"({res_k['true_peak_depth_m']:.1f} m)")
            if simpeg_model is not None:
                print(f"  SimPEG:    Peak {res_k['simpeg_peak_sigma']:.2f} "
                      f"S/m in Schicht {res_k['simpeg_peak_layer']} "
                      f"({res_k['simpeg_peak_depth_m']:.1f} m)")
            print(f"  Surrogate: Peak {res_k['surrogate_peak_sigma']:.2f} "
                  f"S/m in Schicht {res_k['surrogate_peak_layer']} "
                  f"({res_k['surrogate_peak_depth_m']:.1f} m)")
        else:
            # Hintergrundmodell: Untergrund-Rekonstruktion bewerten
            print(f"  wahr:      Untergrund "
                  f"{true_model.min():.4f}-{true_model.max():.4f} S/m")
            if simpeg_model is not None:
                print(f"  SimPEG:    Median-Faktor "
                      f"{res_k['simpeg_median_factor']:.2f} "
                      f"(Log-Fehler {res_k['simpeg_median_log_err']:.3f})")
            print(f"  Surrogate: Median-Faktor "
                  f"{res_k['surrogate_median_factor']:.2f} "
                  f"(Log-Fehler {res_k['surrogate_median_log_err']:.3f})")
            if res_k.get("surrogate_vs_simpeg_log_err_ratio"):
                v = res_k["surrogate_vs_simpeg_log_err_ratio"]
                print(f"  Fair: Surrogate/SimPEG-Log-Fehler-Verhaeltnis "
                      f"{v:.2f} "
                      f"({'Surrogate besser' if v < 1 else 'SimPEG besser'})")
        print(f"  Laufzeit: {t_el:.1f} s")

    # --- Gesamtstatistik ---
    both = [m for m in summary["models"]
            if m["simpeg_peak_layer"] is not None]
    floez_models = [m for m in both if m.get("model_class") == "floez"]
    bg_models = [m for m in both if m.get("model_class") == "background"]

    if floez_models:
        layer_agree = sum(
            1 for m in floez_models
            if abs(m["simpeg_peak_layer"] - m["surrogate_peak_layer"])
            <= 1
        )
        summary["floez_layer_agreement_within_1_percent"] = round(
            100.0 * layer_agree / len(floez_models), 1
        )

    if bg_models:
        # Fairer Vergleich auf Hintergrundmodellen: mittlerer
        # Log-Fehler je Verfahren (Faktorfehler, symmetrisch) und
        # wie oft das Surrogate naeher am wahren Untergrund lag.
        sim_errs = [m["simpeg_median_log_err"] for m in bg_models]
        surr_errs = [m["surrogate_median_log_err"] for m in bg_models]
        summary["background"] = {
            "n_models": len(bg_models),
            "simpeg_median_log_err": float(np.median(sim_errs)),
            "surrogate_median_log_err": float(np.median(surr_errs)),
            "simpeg_mean_log_err": float(np.mean(sim_errs)),
            "surrogate_mean_log_err": float(np.mean(surr_errs)),
            # Faktor-Interpretation: 10**log_err = mittlerer Faktor,
            # um den die Rekonstruktion vom wahren Wert abweicht
            "simpeg_median_factor": float(10 ** np.median(sim_errs)),
            "surrogate_median_factor": float(10 ** np.median(surr_errs)),
            "surrogate_better_count": sum(
                1 for m in bg_models
                if m["surrogate_median_log_err"]
                < m["simpeg_median_log_err"]
            ),
        }
        print("\n--- Hintergrund-Rekonstruktion (fair, Log-Fehler) ---")
        b = summary["background"]
        print(f"  SimPEG:    Median-Faktor {b['simpeg_median_factor']:.2f} "
              f"(Log {b['simpeg_median_log_err']:.3f})")
        print(f"  Surrogate: Median-Faktor {b['surrogate_median_factor']:.2f} "
              f"(Log {b['surrogate_median_log_err']:.3f})")
        print(f"  Surrogate besser in {b['surrogate_better_count']}"
              f"/{b['n_models']} Faellen")

    if both:
        summary["median_surrogate_vs_simpeg_rel_diff_pct"] = float(
            np.median([m["surrogate_vs_simpeg_median_rel_diff_pct"]
                       for m in both])
        )
        print("\n--- Gesamtstatistik ---")
        if floez_models:
            print(f"  Schicht-Uebereinstimmung (+-1, Flöze): "
                  f"{summary['floez_layer_agreement_within_1_percent']}%")
        print(f"  Median Differenz Surrogate vs. SimPEG: "
              f"{summary['median_surrogate_vs_simpeg_rel_diff_pct']:.1f}%")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False,
                      default=str)
        print(f"Referenz-Inversion gespeichert: {output_file}")

    # --- Visualisierung: je Modell 3 Zeilen ---
    # Zeile 1: Modellvergleich wahr/SimPEG/Surrogate (Tiefenprofil)
    # Zeile 2: Datenvergleich (beobachtete Daten + Vorwaertsantworten
    #          beider Loesungen gegen die wahre Antwort)
    # Zeile 3: Floezverfolgung (Peak-Tiefe wahr/SimPEG/Surrogate)
    if plot:
        n_cols = len(summary["models"])
        fig, axes = plt.subplots(3, n_cols,
                                 figsize=(4.5 * n_cols, 12),
                                 sharey="row")
        if n_cols == 1:
            axes = axes.reshape(3, 1)
        for j, m_entry in enumerate(summary["models"]):
            ax_m, ax_d, ax_t = axes[0, j], axes[1, j], axes[2, j]
            seed = m_entry["seed"]
            m_class = m_entry.get("model_class", "floez")

            # Zeile 1: Modellvergleich
            ax_m.step(m_entry["true_model"], boundaries, "k-",
                      where="post", label="wahr")
            if "simpeg_model" in m_entry:
                ax_m.step(m_entry["simpeg_model"], boundaries, "b--",
                          where="post", label="SimPEG (klassisch)")
            ax_m.step(m_entry["surrogate_model"], boundaries, "r-",
                      where="post", label="Surrogate")
            ax_m.set_xscale("log")
            ax_m.set_xlabel("Leitfaehigkeit (S/m)")
            cls_label = ("Flloezmodell" if m_class == "floez"
                         else "Hintergrundmodell")
            # Median-Faktor je Verfahren im Titel (faire Kennzahl)
            title = (f"{cls_label} - seed {seed}\n"
                     f"Median-Faktor: SimPEG "
                     f"{m_entry.get('simpeg_median_factor', float('nan')):.2f}"
                     f" | Surrogate "
                     f"{m_entry.get('surrogate_median_factor', float('nan')):.2f}")
            ax_m.set_title(title, fontsize=10)
            ax_m.legend(fontsize=8)
            ax_m.grid(True, alpha=0.3)

            # Zeile 2: Datenvergleich
            obs = np.abs(np.array(m_entry["observed_data"]))
            ax_d.loglog(times * 1e3, obs, "k.", label="beobachtet")
            ax_d.loglog(times * 1e3,
                        np.abs(m_entry["true_dpred"]), "k-",
                        alpha=0.4, label="wahres Modell")
            if "simpeg_dpred" in m_entry:
                ax_d.loglog(times * 1e3,
                            np.abs(m_entry["simpeg_dpred"]), "b--",
                            label="SimPEG dpred")
            ax_d.loglog(times * 1e3,
                        np.abs(m_entry["surrogate_dpred"]), "r-",
                        label="Surrogate dpred")
            ax_d.set_xlabel("Zeit (ms)")
            ax_d.set_title(f"Datenvergleich - seed {seed}")
            ax_d.legend(fontsize=8)
            ax_d.grid(True, which="both", alpha=0.3)

            # Zeile 3: Flöz-/Peakverfolgung — nur für Flözmodelle
            # sinnvoll; bei Hintergrundmodellen stattdessen
            # Untergrundprofil-Verstärkung (Geist-Anzeige)
            if m_class == "floez":
                ax_t.plot([0], [m_entry["true_peak_depth_m"]], "ko",
                          label="wahr")
                if m_entry.get("simpeg_peak_depth_m") is not None:
                    ax_t.plot([1], [m_entry["simpeg_peak_depth_m"]], "bs",
                              label="SimPEG")
                ax_t.plot([2], [m_entry["surrogate_peak_depth_m"]], "r^",
                          label="Surrogate")
                ax_t.set_xticks([0, 1, 2])
                ax_t.set_xticklabels(["wahr", "SimPEG", "Surrogate"])
                ax_t.set_title(f"Floezposition - seed {seed}",
                               fontsize=10)
            else:
                # Geisterfloez-Diagnose im Profil: wahres gegen
                # invertiertes Hintergrundmodell (Log-Skalierung
                # zeigt Uebertreibungen als Abweichung nach rechts)
                inv = np.array(m_entry["surrogate_model"])
                true = np.array(m_entry["true_model"])
                ax_t.step(true, boundaries, "k-", where="post",
                          label="wahr")
                ax_t.step(inv, boundaries, "r-", where="post",
                          label="Surrogate")
                if "simpeg_model" in m_entry:
                    ax_t.step(m_entry["simpeg_model"], boundaries,
                              "b--", where="post", label="SimPEG")
                ax_t.set_xscale("log")
                ax_t.set_xlabel("Leitfaehigkeit (S/m)")
                ax_t.set_title(f"Untergrund-Rekonstruktion - seed {seed}"
                               f"\n(kein Floez erwartet)", fontsize=10)
                ax_t.legend(fontsize=8)
            ax_t.grid(True, alpha=0.3)
        axes[0, 0].set_ylabel("Tiefe (m)")
        axes[0, 0].invert_yaxis()
        axes[1, 0].set_ylabel("|dB/dt| (T/s)")
        # Zeile 3: bei Flloezmodellen Peak-Tiefe (Kategorien), bei
        # Hintergrundmodellen Leitfaehigkeitsprofil - einheitliche
        # Tiefenachse, Beschriftung neutral halten
        axes[2, 0].set_ylabel("Tiefe (m)")
        axes[2, 0].invert_yaxis()
        for j, m_entry in enumerate(summary["models"]):
            if m_entry.get("model_class", "floez") != "floez":
                axes[2, j].set_ylabel("Tiefe (m)")
                axes[2, j].invert_yaxis()
                axes[2, j].tick_params(axis="x", labelsize=8)
        plt.suptitle("Referenz-Inversion: wahr vs. klassisch (SimPEG) "
                     "vs. Surrogate", fontsize=12)
        plt.tight_layout()
        # Grafik neben die JSON ins Laufverzeichnis speichern
        if output_file:
            png_path = output_file.replace(".json",
                                           "_models.png")
            pdf_path = png_path.replace(".png", ".pdf")
        else:
            png_path = "reference_inversion_models.png"
            pdf_path = "reference_inversion_models.pdf"
        plt.savefig(png_path, dpi=150)
        plt.savefig(pdf_path)
        print(f"Grafik gespeichert: {png_path}")

    return summary


def run_pipeline(device="cuda"):
    """Kompletter Lauf (DEV oder FULL), gesteuert ueber CONFIG.

    DEV_MODE = True  -> Parameter aus DEV_CONFIG  (Schnelltest)
    DEV_MODE = False -> Parameter aus FULL_CONFIG (Produktionslauf)

    Der Ablauf ist identisch; nur die Parametergroessen (Proben,
    Epochen, Iterationen, Restarts) unterscheiden sich. Frueher waren
    dies zwei Funktionen (run_quick_test/run_full) mit dupliziertem
    Code - zusammengefuehrt, da run_full ohnehin dieselben Phasen
    durchlief.
    """
    import time
    t0 = time.time()

    cfg = CONFIG
    mode = "DEV MODE - Schnelltest" if DEV_MODE else "FULL MODE - Produktionslauf"

    # Eigener zeitstempel-benannter Ordner fuer diesen Lauf
    run_dir = make_run_dir()

    print("=" * 60)
    print(f"{mode} ({cfg['n_samples']} Proben, "
          f"{cfg['epochs']} Epochen)")
    print("=" * 60)

    # --- Gemeinsame Parameter aus CONFIG ---
    n_layer = cfg["n_layer"]
    # Zeiten bis 20 ms
    times = np.logspace(-5, np.log10(0.02), cfg["n_times"])
    src_location = np.array([0.0, 0.0, 0.5])
    rx_location = np.array([0.0, 0.0, 0.5])

    # Festes Schichtraster: 12 endliche Schichten + Halbraum
    thicknesses = make_fixed_layer_mesh(
        z_multiplier=cfg["z_multiplier"],
        first_layer=cfg["first_layer"],
        z_max=cfg["z_max"],
    )
    assert len(thicknesses) + 1 == n_layer, (
        f"Raster ({len(thicknesses)}+1) passt nicht zu n_layer={n_layer}"
    )

    # --- Phase 1: Trainingsdaten generieren (multi-core) ---
    print(f"\n--- Phase 1: Datengenerierung ({cfg['n_samples']} Proben) ---")
    data_file = os.path.join(run_dir, "tdem_1d_dev_data.npz")
    models, responses, radii, currents = generate_training_data_1d(
        n_samples=cfg["n_samples"],
        times=times,
        src_location=src_location,
        rx_location=rx_location,
        thicknesses=thicknesses,
        sigma_range=cfg["sigma_range"],
        radius=cfg["radius"],                 # Tupel = konditioniert, Skalar = fix
        current=cfg["current"],
        n_workers=cfg["n_workers"],
        fractions=(cfg.get("background_fraction", 0.75),
                   cfg.get("floez_fraction", 0.15),
                   cfg.get("hard_fraction", 0.10)),
        output_file=data_file,
    )
    print(f"Phase 1 abgeschlossen ({time.time() - t0:.1f}s)")

    # --- Phase 2: Surrogate trainieren ---
    print(f"\n--- Phase 2: Surrogate-Training ({cfg['epochs']} Epochen) ---")
    data = np.load(data_file, allow_pickle=True)
    models = data["models"]
    responses = data["responses"]
    radii = data["radii"] if "radii" in data else None
    currents = data["currents"] if "currents" in data else None
    n_model_params = models.shape[1]
    n_time_channels = len(times)

    model, losses, norm_stats = train_surrogate_mlp(
        models=models,
        responses=responses,
        n_model_params=n_model_params,
        n_time_channels=n_time_channels,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        device=device,
        hidden_dims=cfg["hidden_dims"],
        radii=radii, currents=currents,   # konditioniert, falls vorhanden
        # Zum Fortsetzen eines Trainings Checkpoint-Datei angeben:
        # resume_file=os.path.join("tdem_results", "<lauf-id>",
        #                          "tdem_surrogate_checkpoint.pth"),
        checkpoint_file=os.path.join(run_dir,
                                     "tdem_surrogate_checkpoint.pth"),
    )
    print(f"Phase 2 abgeschlossen ({time.time() - t0:.1f}s)")
    print(f"  Finaler Loss: {losses[-1]:.6e}")

    # Finales Modell separat speichern (nur Gewichte, fuer Inferenz)
    torch.save(model.state_dict(),
               os.path.join(run_dir, "surrogate_mlp_final.pth"))

    # --- Surrogate-Qualitaet evaluieren ---
    # Faustregel: RMSE (log10-Raum) < 0.05 = brauchbar fuer Inversion
    # (bei konditioniertem Surrogate muessen radius/current-Spalten
    #  angehaengt werden - das Netz hat 15 Eingaben, nicht 13)
    model.eval()
    with torch.no_grad():
        x_np = (np.log10(models) - norm_stats["mu_x"]) / norm_stats["std_x"]
        if radii is not None and currents is not None:
            cond_np = (
                (np.log10(np.stack([radii, currents], axis=1))
                 - norm_stats["mu_cond"]) / norm_stats["std_cond"]
            )
            x_np = np.hstack([x_np, cond_np])
        x = torch.FloatTensor(x_np).to(device)
        pred_log = (model(x).cpu().numpy() * norm_stats["std_y"]
                    + norm_stats["mu_y"])
        true = np.log10(np.abs(responses) + 1e-30)
    rmse = np.sqrt(np.mean((pred_log - true) ** 2))
    print(f"  Surrogate-RMSE (log10-Raum): {rmse:.4f} "
          f"{'(OK)' if rmse < 0.05 else '(zu hoch - Parameter erhoehen!)'}")

    # --- RMSE je Modellklasse (Trainingsmischung diagnosizierbar) ---
    # Der Gesamtmittel verschluckt, ob Anstiege aus der harten Klasse
    # kommen (erwartbar) oder aus Floez-Schwaechen (kritisch).
    # Klassenlabels liegen seit der Mischungs-Generierung im npz.
    rmse_by_class = {}
    try:
        classes = data["classes"] if "classes" in data else None
        if classes is not None:
            classes = np.array(classes, dtype=object)
            for cls in ["background", "floez", "hard"]:
                mask = classes == cls
                if mask.any():
                    rmse_c = float(np.sqrt(
                        np.mean((pred_log[mask] - true[mask]) ** 2)))
                    rmse_by_class[cls] = rmse_c
                    print(f"    RMSE {cls:12s}: {rmse_c:.4f} "
                          f"(n={int(mask.sum())})")
            # Erwartung: background am niedrigsten, hard am hoechsten;
            # floez sollte unter dem Gesamtwert bleiben, sonst
            # ist die Spezialisierung geschaedigt.
    except Exception as e:
        print(f"  (Klassen-RMSE nicht verfuegbar: {e})")

    # --- Phase 3: Inversion ---
    print(f"\n--- Phase 3: Inversion ({cfg['n_iterations']} Iterationen) ---")

    # Totzeit (Ramp-off) fuer die Test-Geometrie; SurrogateInversion
    # rechnet dieselbe Totzeit automatisch aus radius_test/current_test
    t_off = turnoff_exclusion_time(
        cfg.get("radius_test", cfg["radius"]),
        cfg.get("current_test", cfg["current"]),
    )
    print(f"Totzeit Ramp-off (R={cfg.get('radius_test', cfg['radius'])} m, "
          f"I={cfg.get('current_test', cfg['current'])} A): "
          f"t_excl={t_off['t_excl'] * 1e6:.1f} us "
          f"(t_clamp={t_off['t_clamp'] * 1e6:.1f} us, "
          f"tau={t_off['tau'] * 1e6:.2f} us, "
          f"L={t_off['L'] * 1e3:.3f} mH) -> "
          f"{int(np.sum(times < t_off['t_excl']))}/{len(times)} Kanaele "
          f"vor {t_off['t_excl'] * 1e6:.1f} us betroffen")

    # Synthetische "Felddaten": Proxy-Modell fuer steilstehendes Graphitfloez
    # (leitfaehige Zone in zufaelliger Tiefe, widerstaendiges Nebengestein)
    true_model = make_test_model(thicknesses, seed=42)
    field_data = run_1d_tdem_forward(
        thicknesses, true_model, times,
        src_location, rx_location,
        radius=cfg.get("radius_test", cfg["radius"]), current=cfg.get("current_test", cfg["current"]),
    )

    initial_model = np.array([1e-2] * n_layer)

    # radius/current muessen uebergeben werden, wenn das Surrogate
    # konditioniert trainiert wurde (Netz mit 15 Eingaben)
    inverter = SurrogateInversion(
        model, times, device=device, norm_stats=norm_stats,
        radius=cfg.get("radius_test", 20.0),
        current=cfg.get("current_test", 5.0),
    )
    result = inverter.invert(
        observed_data=field_data,
        initial_model=initial_model,
        n_iterations=cfg["n_iterations"],
        lr=cfg["inv_lr"],
        regularization_weight=cfg["reg_weight"],
    )

    print(f"\nPhase 3 abgeschlossen ({time.time() - t0:.1f}s)")

    # --- Ergebnis-Export und Visualisierung ---
    surrogate_pred = inverter.forward_surrogate(result)
    rel_error = np.abs(result - true_model) / true_model * 100

    results = {
        "timestamp": datetime.now().isoformat(),
        "mode": "DEV" if DEV_MODE else "FULL",
        "config": cfg,
        "surrogate_rmse_log10": rmse,
        "surrogate_rmse_by_class": rmse_by_class,
        "final_train_loss": losses[-1],
        "turnoff": {k: float(v) for k, v in t_off.items()},
        "true_model": true_model.tolist(),
        "layer_boundaries": np.cumsum(thicknesses).tolist(),
        "inverted_model": result.tolist(),
        "relative_error_percent": rel_error.tolist(),
        "total_runtime_s": round(time.time() - t0, 1),
    }
    save_results(results, os.path.join(
        run_dir, "tdem_dev_results.json" if DEV_MODE
        else "tdem_full_results.json"))
    plot_results(
        true_model=true_model,
        true_thicknesses=thicknesses,
        inverted_model=result,
        inverted_thicknesses=thicknesses,
        observed_data=field_data,
        surrogate_prediction=surrogate_pred,
        times=times,
        train_losses=losses,
        rmse=rmse,
        output_prefix=os.path.join(run_dir, "tdem_dev"),
    )

    # --- Vergleich ---
    print("\n" + "=" * 60)
    print("ERGEBNISSE")
    print("=" * 60)
    print(f"Wahres Modell:       {true_model}")
    print(f"Invertiertes Modell: {result}")
    print(f"Relativer Fehler:    "
          f"{np.abs(result - true_model) / true_model * 100}")
    print(f"\nGesamtzeit: {time.time() - t0:.1f}s")
    print("=" * 60)

    # --- Phase 3B (optional): Mehrfach-Test ---
    # 1. Mit FGM3D/75-Rauschen (Feldfall-Simulation)
    # 2. Ohne Rauschen (Bias-Diagnose: Haufen sich die invertierten
    #    Maxima auch ohne Rauschen bei ~Schicht 8, liegt es am
    #    Optimierer/Regularisierung; sonst am Rauschen)
    RUN_MULTI_TEST = True   # Bei Bedarf auf False setzen
    if RUN_MULTI_TEST:
        print("\n" + "=" * 60)
        print(f"MULTI-TEST (20 Modelle, MIT FGM3D/75-Rauschen, "
              f"{cfg.get('n_stacks', 32)} Stacks)")
        print("=" * 60)
        run_multi_test(
            model=model,
            norm_stats=norm_stats,
            thicknesses=thicknesses,
            times=times,
            src_location=src_location,
            rx_location=rx_location,
            radius=cfg.get("radius_test", cfg["radius"]),
            current=cfg.get("current_test", cfg["current"]),
            device=device,
            n_tests=20,
            n_iterations=500,
            lr=cfg["inv_lr"],
            regularization_weight=cfg["reg_weight"],
            noise=True,
            n_stacks=cfg.get("n_stacks", 32),
            n_restarts=cfg.get("n_restarts", 1),
            output_file=os.path.join(run_dir, "multi_test_results.json"),
            plot_file=os.path.join(run_dir, "multi_test_results.png"),
        )

        print("\n" + "=" * 60)
        print("MULTI-TEST (20 Modelle, OHNE Rauschen - Bias-Diagnose)")
        print("=" * 60)
        run_multi_test(
            model=model,
            norm_stats=norm_stats,
            thicknesses=thicknesses,
            times=times,
            src_location=src_location,
            rx_location=rx_location,
            radius=cfg.get("radius_test", cfg["radius"]),
            current=cfg.get("current_test", cfg["current"]),
            device=device,
            n_tests=20,
            n_iterations=500,
            lr=cfg["inv_lr"],
            regularization_weight=cfg["reg_weight"],
            noise=False,
            output_file=os.path.join(run_dir,
                                     "multi_test_results_no_noise.json"),
            plot_file=os.path.join(run_dir,
                                   "multi_test_results_no_noise.png"),
        )

        # --- Phase 3C: Profil-Test (einfallendes Floez, 10 Stationen) ---
        # Einfallswinkel: 90 = saiger (steilstehend), 80, 70, 60, 45, 20 = flach
        for dip in [90.0, 80.0, 70.0, 60.0, 45.0, 20.0]:
            print("\n" + "=" * 60)
            print(f"PROFIL-TEST: Floez {dip:.0f} Grad einfallend, "
                  f"10 Stationen, {cfg.get('n_stacks', 32)} Stacks")
            print("=" * 60)
            run_profile_test(
                model=model,
                norm_stats=norm_stats,
                thicknesses=thicknesses,
                times=times,
                src_location=src_location,
                rx_location=rx_location,
                radius=cfg.get("radius_test", cfg["radius"]),
                current=cfg.get("current_test", cfg["current"]),
                device=device,
                dip_deg=dip,
                n_stations=cfg.get("n_stations", 10),
                station_spacing=cfg.get("station_spacing", 5.0),
                n_iterations=cfg["n_iterations"],
                lr=cfg["inv_lr"],
                regularization_weight=cfg["reg_weight"],
                noise=True,
                n_stacks=cfg.get("n_stacks", 32),
            n_restarts=cfg.get("n_restarts", 1),
                output_prefix=os.path.join(run_dir, "profile"),
                klassisch=KLASSISCHE_INVERSION,  # SimPEG je Station (~30 s)
            )

    print("\n" + "=" * 60)
    print(f"GESAMTLAUFZEIT PROZESS: {time.time() - t0:.1f} s "
          f"({(time.time() - t0) / 60:.1f} min)")
    print("=" * 60)

    # --- Phase 3E (optional): Klassische SimPEG-Referenz-Inversion ---
    # Vergleicht Surrogate- gegen SimPEG-Inversion auf denselben Daten
    # (Trennung von Surrogate-Fehler und Inversionsverhalten).
    if KLASSISCHE_INVERSION:
        print("\n" + "=" * 60)
        print("KLASSISCHE REFERENZ-INVERSION (SimPEG vs. Surrogate)")
        print("=" * 60)
        t_ref = time.time()
        run_reference_inversion(
            model=model, norm_stats=norm_stats,
            thicknesses=thicknesses, times=times,
            radius=cfg.get("radius_test", 20.0),
            current=cfg.get("current_test", 5.0),
            device=device,
            output_file=os.path.join(run_dir, "reference_inversion.json"),
            max_iter_simpeg=cfg.get("max_iter_simpeg", 150),
            plot=True,  # Modell-/Datenvergleich + Floezverfolgung je Modell
        )
        print(f"\nGesamtlaufzeit Referenz-Inversion: "
              f"{time.time() - t_ref:.1f} s "
              f"({(time.time() - t_ref) / 60:.1f} min)")
    


if __name__ == "__main__":
    # Windows/multiprocessing: sicherstellen, dass der Spawn-Schutz
    # greift (Worker-Funktionen muessen aus einer gespeicherten Datei
    # importierbar sein; Skript NICHT aus Konsole/Temp-Datei starten).
    multiprocessing.freeze_support()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Verfuegbares Geraet: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram_gb:.1f} GB")

    # DEV_MODE entscheidet nur noch, welche CONFIG run_pipeline nutzt
    run_pipeline(device=device)