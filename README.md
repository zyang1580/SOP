# Scale over Preference: The Impact of AI-Generated Content on Online Content Ecology

This repository contains the official analytical codebase and visualization scripts for the paper **"Scale over Preference: The Impact of AI-Generated Content on Online Content Ecology"**. 

## 📖 Overview

The rapid proliferation of Artificial Intelligence-Generated Content (AIGC) is fundamentally restructuring online content ecologies. This study elucidates the distinct creation and consumption behaviors characterizing AIGC versus Human-Generated Content (HGC). 

We identified a prevalent **scale-over-preference dynamic**, wherein AIGC creators achieve aggregate engagement comparable to HGC creators through high-volume production, despite a marked consumer preference for HGC. Deeper time-series and matching analyses uncovered the ability of the algorithmic content distribution mechanism in moderating these competing interests regarding AIGC.

## 🔒 Data Availability Statement

This study leverages a comprehensive longitudinal dataset comprising tens of millions of users from a leading Chinese video-sharing platform. Due to strict corporate confidentiality agreements and user privacy protection policies, the raw interaction logs, creator metadata, and daily panel datasets cannot be open-sourced. 

However, to ensure methodological transparency and reproducibility of our analytical framework, this repository provides the complete, unredacted codebase used to process the data, execute the econometric models, and generate the figures presented in the manuscript.

## 📂 Repository Structure

The codebase is meticulously organized into three functional directories:

### 1. `core/` (Statistical & Econometric Engines)
Contains the foundational algorithms used across the study:
* `matching.py`: Implements hash-blocking and nearest-neighbor matching to mitigate selection bias across categorical and continuous covariates.
* `regression.py`: Houses the Dynamic OLS (DOLS) estimation engine, block bootstrapping for robust confidence intervals, and the Shin (1994) Test for residual stationarity.

### 2. `analysis/` (Empirical Operationalizations)
Contains sequentially executable Jupyter Notebooks mapping directly to the paper's methodology:
* `01_matching_creator.ipynb` to `04_matching_algorithm.ipynb`: Execution of matching pipelines for creators, consumers, video distribution, and algorithmic content distribution mechanisms.
* `05_granger_causality.ipynb`: Unit root testing and Granger causality analysis identifying directional precedence.
* `06_dols_operationalizations.ipynb`: Comprehensive DOLS estimations linking AIGC supply scale to initial visibility, relative exposure dynamics, and downstream ecological outcomes.

### 3. `visualizer/` (Figure Generation)
Contains the highly customized plotting scripts used to generate the multi-panel main figures:
* `01_visualizer_fig1.ipynb`: Reproduces **Figure 1**, illustrating the Scale-over-Preference dynamic in content creation and consumption.
* `02_visualizer_fig2.ipynb`: Reproduces **Figure 2**, illustrating how algorithmic content distribution mechanisms moderate the scale-over-preference dynamic.

## ⚙️ Usage Notes
While the scripts cannot be executed without the proprietary `.parquet` and `.csv` datasets, they are heavily annotated. Researchers and reviewers can trace the exact mathematical implementations—from exact hash blocking to recursive block bootstrapping—directly within the `core/` and `analysis/` modules.
