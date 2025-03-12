# common_imports.py

# ============================
# Data Manipulation & Analysis
# ============================
import pandas as pd
import numpy as np
import glob
import re
import math
from collections import Counter
import itertools
import random

# ============================
# Plotting and Visualization
# ============================
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import scipy.cluster.hierarchy as sch

# ============================
# Statistics and Machine Learning
# ============================
from scipy.stats import gaussian_kde, pearsonr, roc_curve, auc, chi2_contingency, ttest_ind, mannwhitneyu
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# XGBoost classifier
from xgboost import XGBClassifier

# ============================
# Bioinformatics & Sequence Handling
# ============================
from Bio.Seq import Seq
import pyfaidx
from kipoiseq import Interval
import requests
from functools import lru_cache

# ============================
# Interactive Plotting (Optional)
# ============================
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================
# End of common_imports.py
# ============================
