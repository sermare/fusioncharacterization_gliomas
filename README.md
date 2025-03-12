# Fusion Gene Analysis and Visualization Repository

This repository contains a comprehensive suite of tools for the analysis, visualization, and modeling of gene fusion data. The focus is on exploring fusion events in various tumor subtypes (e.g., Glioblastoma, Oligodendroglioma, Astrocytoma) using multiple fusion detection algorithms. The tools provided allow for data loading, preprocessing, statistical analysis, and the creation of informative visualizations, as well as the application of machine learning techniques to classify samples and uncover key drivers behind fusion events.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources and Inputs](#data-sources-and-inputs)
- [Repository Structure](#repository-structure)
- [Modules and Functions](#modules-and-functions)
  - [Data and Preprocessing Utilities](#data-and-preprocessing-utilities)
  - [Visualization Functions](#visualization-functions)
  - [Machine Learning Models](#machine-learning-models)
  - [Common Paths and Package Imports](#common-paths-and-package-imports)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project aims to integrate gene fusion data from various sources and fusion detection algorithms to provide a detailed view of fusion events across tumor subtypes. The analysis pipeline includes:

- **Data Loading and Preprocessing:** Reading raw fusion files, cleaning data, and merging datasets.
- **Statistical Analysis:** Aggregating fusion counts per patient and tumor subtype, performing hypothesis tests, and summarizing fusion event distributions.
- **Visualization:** Generating a wide array of plots, such as bar charts, heatmaps, clustermaps, density plots, and ROC curves, to effectively present the analysis results.
- **Machine Learning:** Training classification models to predict sample outcomes based on fusion features.
- **Bioinformatics Utilities:** Extracting sequence information, processing k-mer counts, calculating entropy, and mapping genomic coordinates.

---

## Data Sources and Inputs

The project uses multiple types of input files and external data sources, including:

- **Reference Genome and Annotations:**
  - Genome FASTA: `/costellolab/data3/sermare/arriba_hg38/hg38.fa`
  - Cytoband / Chromosome Sizes: `~/cytoBand.txt`
  
- **Fusion Data Files:**
  - SRA Fusion Files from multiple directories (e.g., `/costellolab/data3/sermare/datasets/SRA/` and `/costellolab/data3/sermare/datasetsSRA/`).
  
- **Purity and Metadata Files:**
  - Purity Scores: `~/jupyter-notebooks/14112024_purity_scores.csv`
  
- **Gene and Annotation References:**
  - Oncogene/TSG Data: `~/random_tsv_files/Census_allWed Mar 15 21_17_28 2023.tsv`
  - Transcription Factor List: `~/random_tsv_files/TF_names_v_1.01.txt`
  - Kinase Map: `~/random_tsv_files/Map/globalKinaseMap.csv`
  
- **GTEx Fusion Data:**
  - GTEx Fusion Files: `/c4/home/sermare/jupyter-notebooks/old_notebooks/GTEX_fusions.csv`
  - Fusion Inspector Data: `/c4/home/sermare/random_tsv_files/GTEX_TCGA_fusion_inspector.csv`

---

---

## Modules and Functions

### Data and Preprocessing Utilities

- **data_utils.py:** Contains functions for loading fusion data from directories.
- **preprocessing_utils.py:** Includes functions for cleaning data, parsing fusion coordinates, processing sequences (k-mer counts and Shannon entropy), and mapping genomic coordinates.

### Visualization Functions

- **plots.py:** Contains functions to create:
  - Bar charts
  - Heatmaps and clustermaps
  - Joint scatter plots with regression lines
  - ROC curves
  - Boxplots with significance annotations
  - Fusion gene and breakpoint density plots
  - Clustermap-based fusion maps

### Machine Learning Models

- **ml_models.py:** Provides functions for:
  - Splitting data
  - Training classifiers such as Random Forest and XGBoost
  - Evaluating models (accuracy, confusion matrix, ROC curves, feature importances)

### Common Paths and Package Imports

- **paths.py:** Centralizes important file paths and constants.
- **common_imports.py:** Imports all the frequently used packages so you can load them in one go.

---

## Usage

1. **Set Up Environment:**  
   Clone the repository and install dependencies listed in `requirements.txt`.

2. **Data Loading and Preprocessing:**  
   Use the functions in `data_utils.py` and `preprocessing_utils.py` to load and clean your fusion data.

3. **Visualization:**  
   Call plotting functions from `plots.py` to generate the visualizations. For example, to create a heatmap:
   ```python
   from plots import plot_heatmap
   plot_heatmap(your_dataframe, title="Fusion Heatmap", xlabel="Samples", ylabel="Fusion Genes")


Machine Learning:
Use functions in ml_models.py to train and evaluate models. For example:
from ml_models import train_random_forest
model, accuracy, report, cv_score = train_random_forest(X, y)
print(accuracy)
print(report)
Running the Pipeline:
The main.py script orchestrates the overall workflow. You can run it from the command line:
python main.py
Exploration:
Use the notebooks in the notebooks/ directory for interactive data exploration and report generation.
Contributing

Contributions are welcome! If you have suggestions or improvements, please create an issue or submit a pull request. When contributing, please ensure that your code adheres to the repository's style and that you update the documentation as needed.