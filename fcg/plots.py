# plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr
import scipy.cluster.hierarchy as sch
from matplotlib.colors import ListedColormap
from collections import Counter
import itertools
from sklearn.metrics import confusion_matrix

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import ListedColormap

import colorsys
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import ListedColormap
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import ListedColormap


def plot_fusion_heatmap(final_df):
    """
    Plots a heatmap for fusion gene analysis across samples.

    Parameters:
        final_df (pd.DataFrame): DataFrame containing fusion gene data with categorical and numerical features.
    """
    # Create a copy of the dataframe
    plot_df = final_df.copy()

    # Store original categorical values for legend
    histology_map = dict(enumerate(final_df["Histology"].astype("category").cat.categories))
    tumor_map = dict(enumerate(final_df["Tumor"].astype("category").cat.categories))

    # Convert categorical data to numerical codes
    plot_df["Histology"] = plot_df["Histology"].astype("category").cat.codes
    plot_df["Tumor"] = plot_df["Tumor"].astype("category").cat.codes
    plot_df["patient"] = plot_df["patient"].astype("category").cat.codes

    # Select features to visualize
    features = ["Histology", "Tumor", "RNAseq_number", "STAR_clonal_in_sample", "Arriba_clonal_in_sample"]

    # Prepare heatmap data
    heatmap_data = plot_df.set_index(["SF#", "patient"])[features].T

    # Define custom colormaps
    histology_colors = sns.color_palette("Set1", n_colors=len(set(plot_df["Histology"])))
    tumor_colors = sns.color_palette("Set2", n_colors=len(set(plot_df["Tumor"])))
    colormaps = [ListedColormap(histology_colors), ListedColormap(tumor_colors), "PuRd", "Blues", "Oranges"]
    feature_labels = ["Histology", "Tumor Type", "RNAseq \n Count", "STAR  \n Clonal Fusions", "Arriba \n Clonal Fusions"]

    # Create figure with additional space for legends and colorbars
    fig = plt.figure(figsize=(16, 9))  # Increased height to accommodate vertical colorbars
    gs = GridSpec(nrows=6, ncols=3, width_ratios=[20, 1, 5], height_ratios=[1, 1, 1, 1, 1, 1], 
                  wspace=0.1, hspace=0.0)

    # Plot each feature row
    ims = []  # Store image objects for later colorbar creation
    for i in range(len(features)):
        # Heatmap axis
        ax = plt.subplot(gs[i, 0])
        row_data = heatmap_data.iloc[i, :].values.reshape(1, -1)
        
        # Set vmin and vmax for each row
        if i == 0:  # Histology
            vmin, vmax = -0.5, len(set(plot_df["Histology"])) - 0.5
        elif i == 1:  # Tumor Type
            vmin, vmax = -0.5, len(set(plot_df["Tumor"])) - 0.5
        else:  # Numerical features
            vmin, vmax = heatmap_data.iloc[i].min(), heatmap_data.iloc[i].max()
            avg = (vmin + vmax) / 2

        # Create heatmap
        im = ax.imshow(row_data, aspect='auto', cmap=colormaps[i], vmin=vmin, vmax=vmax)
        if i >= 2:  # Store only numerical feature images for colorbars
            ims.append(im)
        
        # Customize axes
        ax.set_ylabel(feature_labels[i], rotation=0, labelpad=40, fontsize=12, ha='right')
        ax.set_yticks([])
        ax.set_xticks([])

        # Add legends
        if i == 0:  # Histology
            legend_ax = plt.subplot(gs[i, 1:3])
            legend_ax.axis('off')
            patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=histology_colors[j], 
                                 markersize=8, label=histology_map[j]) for j in histology_map]
            legend_ax.legend(handles=patches, loc='center left', fontsize=10, title='Histology', 
                             title_fontsize=9, frameon=False, handletextpad=0.5, labelspacing=0.3, 
                             borderaxespad=0, ncol=1)
        elif i == 1:  # Tumor Type
            legend_ax = plt.subplot(gs[i, 1:3])
            legend_ax.axis('off')
            patches = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=tumor_colors[j], 
                                 markersize=8, label=tumor_map[j]) for j in tumor_map]
            legend_ax.legend(handles=patches, loc='center left', fontsize=10, title='Tumor Type', 
                             title_fontsize=9, frameon=False, handletextpad=0.5, labelspacing=0.3, 
                             borderaxespad=0, ncol=1)

    # Add vertical colorbars below Tumor Type legend (using row 2 for colorbars)
    cbar_ax = plt.subplot(gs[2, 2])  # Place colorbars in column 2, row 2 (below Tumor Type)
    cbar_ax.axis('off')

    # Generate colorbars for numerical features
    vmin_vals = [heatmap_data.iloc[i].min() for i in range(2, len(features))]
    vmax_vals = [heatmap_data.iloc[i].max() for i in range(2, len(features))]
    avg_vals = [(vmin_vals[i] + vmax_vals[i]) / 2 for i in range(len(vmin_vals))]

    # Increase height for colorbars
    gs = GridSpec(nrows=6, ncols=3, width_ratios=[20, 1, 5], height_ratios=[1, 1, 1, 1, 0.1, 0.1], 
                  wspace=0.0, hspace=0.0)

    cbar_gs = GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[2, 2], width_ratios=[0.6, 2], hspace=4, wspace=5)

    for i, im in enumerate(ims):
        cbar_ax = plt.subplot(cbar_gs[i, 0])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical', shrink=1.0, aspect=20)  # Taller + thinner
        cbar.set_ticks([vmin_vals[i], avg_vals[i], vmax_vals[i]])
        cbar.set_ticklabels([f"Min: {int(vmin_vals[i])}", f"Avg: {int(avg_vals[i])}", f"Max: {int(vmax_vals[i])}"])
        cbar.ax.tick_params(labelsize=6)
        cbar_ax.set_title(f"{feature_labels[i+2]}", fontsize=8, pad=2, rotation=0, va='bottom')

    # Add title
    plt.suptitle("Fusion Gene Analysis Across Samples", fontsize=16, y=1.02)

    # Adjust layout
    plt.tight_layout(pad=0.5)
    plt.show()



###############################################################################
# Basic Plotting Functions
###############################################################################

def plot_bar_chart(data, x_col, y_col, title="", xlabel="", ylabel="", rotation=45, figsize=(8,6), palette="viridis"):
    """
    Create a simple bar chart.
    
    Parameters:
        data (DataFrame): Data to plot.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        title (str): Plot title.
        xlabel (str): x-axis label.
        ylabel (str): y-axis label.
        rotation (int): Rotation angle for x-tick labels.
        figsize (tuple): Figure size.
        palette: Color palette.
    """
    plt.figure(figsize=figsize)
    sns.barplot(data=data, x=x_col, y=y_col, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


def plot_heatmap(df, title="", xlabel="", ylabel="", cmap="viridis", annot=False, fmt=".2f", figsize=(10,8), **kwargs):
    """
    Plot a heatmap from a DataFrame.
    
    Parameters:
        df (DataFrame): The DataFrame to plot.
        title (str): Title of the plot.
        xlabel (str): x-axis label.
        ylabel (str): y-axis label.
        cmap (str): Colormap.
        annot (bool): If True, annotate cells.
        fmt (str): String formatting code to use when adding annotations.
        figsize (tuple): Figure size.
        **kwargs: Additional arguments to pass to sns.heatmap.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df, cmap=cmap, annot=annot, fmt=fmt, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_clustermap_heatmap(data, title="", row_cluster=True, col_cluster=True, figsize=(10,8), **kwargs):
    """
    Generate a clustermap heatmap using seaborn.
    
    Parameters:
        data (DataFrame): Data to cluster and plot.
        title (str): Plot title.
        row_cluster (bool): Whether to cluster rows.
        col_cluster (bool): Whether to cluster columns.
        figsize (tuple): Figure size.
        **kwargs: Additional keyword arguments for sns.clustermap.
    
    Returns:
        The clustermap object.
    """
    g = sns.clustermap(data, row_cluster=row_cluster, col_cluster=col_cluster, figsize=figsize, **kwargs)
    plt.title(title)
    plt.show()
    return g


def plot_joint_correlation(x, y, title="", x_label="", y_label="", polyfit=True, y_ticks=None):
    """
    Create a joint scatter plot with optional regression line and average lines.
    
    Parameters:
        x (array-like): Data for x-axis.
        y (array-like): Data for y-axis.
        title (str): Overall title for the plot.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
        polyfit (bool): Whether to fit and plot a regression line.
        y_ticks (list or None): Custom y-tick positions.
    
    Returns:
        The jointplot object.
    """
    g = sns.jointplot(x=x, y=y, kind="scatter", marginal_kws=dict(kde=True))
    g.plot_joint(sns.scatterplot, alpha=0.6)
    overall_y_avg = np.mean(y)
    g.ax_joint.axhline(overall_y_avg, color='red', linestyle='--', label=f'Overall Avg: {overall_y_avg:.2f}')
    if polyfit:
        m, b = np.polyfit(x, y, 1)
        g.ax_joint.plot(x, m*x+b, color="blue", linestyle="-", label=f"Line of Best Fit")
    g.set_axis_labels(x_label, y_label)
    g.fig.suptitle(title, y=1.02)
    if y_ticks is not None:
        plt.yticks(y_ticks)
    g.ax_joint.legend()
    plt.show()
    return g


def plot_boxplot_with_significance(data, x_col, y_col, title="", xlabel="", ylabel="", significance_annotation=None, figsize=(3,6)):
    """
    Create a boxplot with an optional significance annotation.
    
    Parameters:
        data (DataFrame): Data for plotting.
        x_col (str): Column name for categorical grouping.
        y_col (str): Column name for the numeric variable.
        title (str): Plot title.
        xlabel (str): x-axis label.
        ylabel (str): y-axis label.
        significance_annotation (tuple or None): Tuple (x1, x2, y, h, p_val) for annotation.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=data, x=x_col, y=y_col)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if significance_annotation:
        x1, x2, y, h, p_val = significance_annotation
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, color='black')
        ax.text((x1 + x2)/2, y+h, f"p = {p_val:.3e}", ha='center', va='bottom', color='black')
    plt.tight_layout()
    plt.show()


def plot_roc_curves(classifiers, X_test, y_test, figsize=(6,6)):
    """
    Plot ROC curves for a set of classifiers.
    
    Parameters:
        classifiers (dict): Dictionary of classifier name to trained classifier.
        X_test (array-like): Test features.
        y_test (array-like): True labels.
        figsize (tuple): Figure size for each subplot.
    """
    fig, axs = plt.subplots(1, len(classifiers), figsize=(6*len(classifiers), 6))
    if len(classifiers) == 1:
        axs = [axs]
    fig.suptitle('ROC Curves for Various Classifiers')
    for ax, (name, clf) in zip(axs, classifiers.items()):
        y_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title(name)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


###############################################################################
# Fusion and Clustermap Specific Plotting Functions
###############################################################################

def plot_breakpoint_density(breaks_df, cum_offset, ordered_chroms, chrom_sizes, title="Breakpoint Density"):
    """
    Plot a histogram of breakpoint densities across a linear genome coordinate.
    
    Parameters:
        breaks_df (DataFrame): Must contain a 'genome_coord' column and optionally 'weight'.
        cum_offset (dict): Mapping of chromosome to its cumulative offset.
        ordered_chroms (list): List of chromosomes in order.
        chrom_sizes (dict): Dictionary mapping chromosome to its size.
        title (str): Plot title.
    """
    plt.figure(figsize=(12,4))
    if 'weight' in breaks_df.columns:
        sns.histplot(data=breaks_df, x='genome_coord', weights='weight', bins=1000, edgecolor='none')
    else:
        sns.histplot(data=breaks_df, x='genome_coord', bins=1000, edgecolor='none')
    for chrom in ordered_chroms[1:]:
        plt.axvline(x=cum_offset[chrom], color='red', linewidth=0.8, alpha=0.6, linestyle='--')
    for i, chrom in enumerate(ordered_chroms):
        start = cum_offset[chrom]
        end = cum_offset[ordered_chroms[i+1]] if i < len(ordered_chroms)-1 else start + chrom_sizes.get(chrom, 0)
        midpoint = (start + end) / 2
        plt.text(midpoint, plt.ylim()[1]*0.9, chrom.replace("chr",""), ha='center', va='bottom', fontsize=8, rotation=90)
    plt.title(title)
    plt.xlabel('Linear Genome Coordinate')
    plt.ylabel('Weighted Count' if 'weight' in breaks_df.columns else 'Count')
    plt.tight_layout()
    plt.show()


def plot_gene_counts(gene_counts_df, histology, top_n=20):
    """
    Plot the top N gene counts for a given histology from a gene counts DataFrame.
    
    Parameters:
        gene_counts_df (DataFrame): DataFrame where index=gene and columns are histology types.
        histology (str): The column name to use for sorting and plotting.
        top_n (int): Number of top genes to display.
    """
    plot_df = gene_counts_df.sort_values(by=histology, ascending=False).head(top_n)
    plt.figure(figsize=(5,5))
    sns.barplot(x=plot_df.index, y=plot_df[histology], palette="viridis")
    plt.title(f"Gene Fusion Counts in {histology}")
    plt.xlabel("Gene")
    plt.ylabel("Count of Fusions")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_top_10_fusions(df, threshold):
    """
    Plot the top 10 fusion genes by unique patient count for fusions with total_algorithms greater than threshold.
    
    Parameters:
        df (DataFrame): Fusion gene DataFrame.
        threshold (int): Filter threshold for 'total_algorithms'.
    
    Returns:
        DataFrame: Top fusion genes with unique patient counts.
    """
    filtered_df = df[df['total_algorithms'] > threshold]
    top_10_fusions = filtered_df['fusion_gene'].value_counts().head(10).index
    patient_fusion = []
    for fusion in top_10_fusions:
        num_unique_patients = filtered_df[filtered_df['fusion_gene'] == fusion]['patient'].nunique()
        patient_fusion.append([fusion, num_unique_patients])
    fusion_df = pd.DataFrame(patient_fusion, columns=['Fusion Gene', 'Number of Patients'])
    
    plt.figure(figsize=(8,6))
    sns.barplot(data=fusion_df, x='Number of Patients', y='Fusion Gene', palette="viridis")
    plt.title(f"Top 10 Fusion Genes (total_algorithms > {threshold})")
    plt.xlabel("Number of Patients")
    plt.ylabel("Fusion Gene")
    plt.tight_layout()
    plt.show()
    
    return fusion_df


def process_and_plot_fusions_by_threshold(df, threshold):
    """
    For each patient in the DataFrame, count fusion events that occur in more than one sample,
    compute a precision metric, and plot the counts.
    
    Parameters:
        df (DataFrame): Fusion DataFrame.
        threshold (int): Threshold for filtering fusions based on total sample count.
    
    Returns:
        DataFrame: Aggregated patient fusion metrics.
    """
    results = []
    for patient in df['patient'].unique():
        patient_data = df[df['patient'] == patient].copy()
        if patient_data['file_path'].nunique() < 2:
            continue
        gene_counts = patient_data.groupby('fusion_gene')['file_path'].nunique()
        genes_multiple = gene_counts[gene_counts > 1].index.tolist()
        n_fusions = len(genes_multiple)
        patient_data['path_count'] = patient_data['fusion_gene'].apply(lambda x: gene_counts[x] if x in genes_multiple else 0)
        patient_data['Label'] = patient_data['path_count'].apply(lambda x: 'True Positive' if x > 1 else 'False Positive')
        label_counts = Counter(patient_data['Label'])
        total = sum(label_counts.values())
        tp = label_counts.get('True Positive', 0)
        precision = tp / total if total > 0 else 0
        results.append({
            'patient': patient,
            'fusions_in_multiple_samples': n_fusions,
            'total_fusions': total,
            'TP': tp,
            'Precision': precision
        })
    results_df = pd.DataFrame(results)
    
    plt.figure(figsize=(18,10))
    plt.bar(results_df['patient'], results_df['total_fusions'], label='Fusions in one sample')
    plt.bar(results_df['patient'], results_df['fusions_in_multiple_samples'], bottom=results_df['total_fusions'] - results_df['fusions_in_multiple_samples'], label='Fusions in >1 sample')
    for index, row in results_df.iterrows():
        plt.text(row['patient'], row['total_fusions'] / 2, f"{row['total_fusions'] - row['fusions_in_multiple_samples']}", ha='center')
        plt.text(row['patient'], row['total_fusions'] + 1, f"{row['fusions_in_multiple_samples']}", ha='center')
    plt.xlabel('Patient')
    plt.ylabel('Number of Gene Fusions')
    plt.title('Gene Fusions per Patient')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    return results_df


def plot_feature_importance(model, feature_names, top_n=10, figsize=(10,6)):
    """
    Plot the top N important features from a model.
    
    Parameters:
        model: Trained model with feature_importances_ attribute.
        feature_names (list): List of feature names.
        top_n (int): Number of top features to display.
        figsize (tuple): Figure size.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = [(feature_names[i], importances[i]) for i in indices[:top_n]]
    features, imp_values = zip(*top_features)
    plt.figure(figsize=figsize)
    plt.barh(range(top_n), imp_values, color='skyblue')
    plt.yticks(range(top_n), features)
    plt.gca().invert_yaxis()
    plt.xlabel("Feature Importance")
    plt.title("Top Important Features")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', figsize=(6,5)):
    """
    Plot a confusion matrix using a heatmap.
    
    Parameters:
        cm (array-like): Confusion matrix.
        class_names (list): List of class names.
        title (str): Plot title.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


###############################################################################
# Clustermap and Patient Fusion Map Functions
###############################################################################

def generate_clustermap(dfs, merged_df, patient_label, include_hal=False, figsize=(20,10)):
    """
    Generate a clustermap of fusion gene presence for a given patient.
    
    Parameters:
        dfs (dict): Dictionary mapping algorithm names to DataFrames.
        merged_df (DataFrame): Aggregated fusion DataFrame.
        patient_label (str): Patient identifier.
        include_hal (bool): If True, exclude hallucination fusions.
        figsize (tuple): Figure size for the clustermap.
    
    Returns:
        DataFrame: The binary matrix used for the clustermap.
    """
    # Gather unique sample identifiers for the patient from each algorithm
    patient_paths = {}
    for algo, df in dfs.items():
        patient_paths[algo] = []
        filtered_df = df[df['patient'].str.contains(patient_label)]
        for file_path in filtered_df['file_path'].unique():
            key = file_path.split('_R1')[0]
            patient_paths[algo].append(key)
    patient_paths = np.unique([elem for arr in patient_paths.values() for elem in arr])
    
    # Get fusion genes observed in the patient across the merged DataFrame.
    if include_hal:
        fusion_genes_patient = merged_df[(merged_df['patient'] == patient_label) &
                                         (merged_df['total_observed_multiple_files'] > 1) &
                                         (~merged_df['fusion_gene'].isin([]))].sort_values(by='total_observed_multiple_files', ascending=False)['fusion_gene'].unique()
    else:
        fusion_genes_patient = merged_df[(merged_df['patient'] == patient_label) &
                                         (merged_df['total_observed_multiple_files'] > 1)].sort_values(by='total_observed_multiple_files', ascending=False)['fusion_gene'].unique()
    
    # Build binary matrix (fusion gene presence per sample)
    fusion_matrix = pd.DataFrame(0, index=fusion_genes_patient, columns=patient_paths)
    for algo, df in dfs.items():
        filtered_df = df[df['patient'].str.contains(patient_label)]
        for file_path in filtered_df['file_path'].unique():
            key = file_path.split('_R1')[0]
            if key in fusion_matrix.columns:
                fusions = filtered_df[filtered_df['file_path'].str.contains(file_path)]['fusion_gene'].unique()
                for fusion in fusions:
                    if fusion in fusion_matrix.index:
                        fusion_matrix.at[fusion, key] = 1
    # Prepare matrix for clustermap: sort rows by total occurrences.
    fusion_matrix['total'] = fusion_matrix.sum(axis=1)
    fusion_matrix = fusion_matrix.sort_values(by='total', ascending=False).drop(columns='total')
    fusion_matrix = fusion_matrix.T
    fusion_matrix['total_fusions'] = fusion_matrix.sum(axis=1)
    fusion_matrix = fusion_matrix.sort_values(by='total_fusions', ascending=False)
    
    # Plot clustermap
    g = sns.clustermap(fusion_matrix.iloc[:,:-1], figsize=figsize, cmap=ListedColormap(['#440154FF', '#FDE725FF']),
                       method='average', metric='euclidean', col_cluster=False, row_cluster=True,
                       cbar_kws={'ticks': [0, 1]})
    g.ax_heatmap.figure.suptitle(f'Fusion Map for patient {patient_label}', x=0.5, y=1.05, fontsize=16)
    g.cax.set_position([0.95, 0.2, 0.04, 0.2])
    plt.show()
    return fusion_matrix

def create_patient_heatmap(merged_df):
    """
    Creates a heatmap visualization with categorical data and tumor purity bar plots.
    
    Parameters:
    merged_df (pandas.DataFrame): Input DataFrame with 'Patient', 'join_key', 'Histology', 
                                 'Tumor', 'STAR_clonal_in_sample', 'Arriba_clonal_in_sample',
                                 and 'plot_purity' columns
    """
    # Add name column
    merged_df['name'] = merged_df['Patient'] + merged_df['join_key']
    plot_df = merged_df.copy()

    # Get unique categories
    unique_histology = sorted(merged_df["Histology"].unique())
    unique_tumor = sorted(merged_df["Tumor"].unique())
    unique_patients = sorted(merged_df["Patient"].unique())

    # Create consistent mappings
    histology_map = {i: val for i, val in enumerate(unique_histology)}
    tumor_map = {i: val for i, val in enumerate(unique_tumor)}
    patient_map = {i: val for i, val in enumerate(unique_patients)}

    # Reverse mappings for encoding
    histology_code_map = {val: i for i, val in histology_map.items()}
    tumor_code_map = {val: i for i, val in tumor_map.items()}
    patient_code_map = {val: i for i, val in patient_map.items()}

    # Convert categorical data to numerical codes
    plot_df["Histology_code"] = plot_df["Histology"].map(histology_code_map)
    plot_df["Tumor_code"] = plot_df["Tumor"].map(tumor_code_map)
    plot_df["Patient_code"] = plot_df["Patient"].map(patient_code_map)

    # Select features to visualize
    features = ["Histology_code", "Tumor_code", "Patient_code", 
                "STAR_clonal_in_sample", "Arriba_clonal_in_sample"]

    # Prepare heatmap data
    heatmap_data = plot_df.set_index(["name"])[features].T

    # Adjust fusion values
    def adjust_fusion_value(x):
        return 20 if x >= 50 else min(x, 20)

    for fusion_feature in ["STAR_clonal_in_sample", "Arriba_clonal_in_sample"]:
        heatmap_data.loc[fusion_feature] = heatmap_data.loc[fusion_feature].apply(adjust_fusion_value)

    # Define custom colormaps
    histology_colors = sns.color_palette("Set1", n_colors=len(unique_histology))
    tumor_colors = sns.color_palette("Set2", n_colors=len(unique_tumor))

    # Generate patient colors
    random.seed(42)
    patient_colors = []
    patient_color_dict = {}
    for patient_code in range(len(unique_patients)):
        hue = random.random()
        saturation = random.uniform(0.3, 0.6)
        lightness = random.uniform(0.4, 0.6)
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        patient_color_dict[patient_code] = hex_color
        patient_colors.append(hex_color)

    # Colormaps for heatmap
    patient_colormap = ListedColormap([patient_color_dict[i] for i in range(len(unique_patients))])
    star_cmap = ListedColormap(plt.get_cmap("PuRd", 11)(range(11)) + ["black"])
    arriba_cmap = ListedColormap(plt.get_cmap("Blues", 11)(range(11)) + ["black"])

    colormaps = [ListedColormap(histology_colors), 
                 ListedColormap(tumor_colors), 
                 patient_colormap, 
                 star_cmap, 
                 arriba_cmap]

    feature_labels = ["Histology", "Tumor Type", "Patient", 
                     "STAR Clonal Fusions", "Arriba Clonal Fusions"]

    # Create figure
    fig = plt.figure(figsize=(24, 22))
    gs = GridSpec(nrows=11, ncols=5, width_ratios=[20, 1, 5, 5, 5], 
                 height_ratios=[0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 2, 0.5, 0.5, 0.5], 
                 wspace=0.3, hspace=0.1)

    # Plot heatmap rows
    ims = []
    for i in range(len(features)):
        ax = plt.subplot(gs[i, 0])
        row_data = heatmap_data.iloc[i, :].values.reshape(1, -1)
        
        vmin = -0.5 if i < 3 else 0
        vmax = [len(unique_histology), len(unique_tumor), len(unique_patients), 20, 20][i] - (0.5 if i < 3 else 0)

        im = ax.imshow(row_data, aspect='auto', cmap=colormaps[i], vmin=vmin, vmax=vmax, interpolation='none')
        if i >= 3:
            ims.append(im)
        
        num_columns = len(plot_df)
        for j in range(num_columns + 1):
            ax.vlines(j - 0.5, ymin=-0.5, ymax=0.5, color='black', linewidth=0.05)
            
        ax.set_ylabel(feature_labels[i], rotation=0, labelpad=50, fontsize=16, ha='right')
        ax.set_yticks([])
        ax.set_xticks([])

        # Legends for categorical variables
        if i == 0:  # Histology
            legend_ax = plt.subplot(gs[i, 2:4])
            legend_ax.axis('off')
            patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=histology_colors[j], 
                                markersize=12, label=histology_map[j]) for j in range(len(unique_histology))]
            legend_ax.legend(handles=patches, loc='center left', fontsize=14, title='Histology', 
                           title_fontsize=16, frameon=False, handletextpad=0.5, labelspacing=0.7, 
                           borderaxespad=0, ncol=1)
        elif i == 1:  # Tumor Type
            legend_ax = plt.subplot(gs[i, 2:4])
            legend_ax.axis('off')
            patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tumor_colors[j], 
                                markersize=12, label=tumor_map[j]) for j in range(len(unique_tumor))]
            legend_ax.legend(handles=patches, loc='center left', fontsize=14, title='Tumor Type', 
                           title_fontsize=16, frameon=False, handletextpad=0.5, labelspacing=0.7, 
                           borderaxespad=0, ncol=1)

    # Patient legends
    patient_info = {}
    for idx, row in plot_df.drop_duplicates('Patient').iterrows():
        patient_code = row['Patient_code']
        patient_name = row['Patient']
        histology_name = histology_map[row['Histology_code']]
        patient_info[patient_code] = {'name': patient_name, 'histology': histology_name, 
                                    'color': patient_color_dict[patient_code]}

    histology_groups = {}
    for patient_code, info in patient_info.items():
        histology_groups.setdefault(info['histology'], []).append(patient_code)

    histology_list = sorted(histology_groups.keys())
    legend_gs = GridSpecFromSubplotSpec(len(histology_list), 1, 
                                      subplot_spec=gs[2:7, 2:4], 
                                      height_ratios=[1] * len(histology_list),
                                      hspace=0.6)

    for i, histology_type in enumerate(histology_list):
        patients_in_group = sorted(histology_groups[histology_type])
        legend_ax = plt.subplot(legend_gs[i, 0])
        legend_ax.axis('off')
        patches = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=patient_info[p]['color'], markersize=10, 
                            label=patient_info[p]['name']) for p in patients_in_group]
        ncols = min(4, len(patches))
        legend_ax.legend(handles=patches, loc='center left', fontsize=11, 
                       title=f'Patients with {histology_type}', title_fontsize=14,
                       frameon=True, handletextpad=0.5, labelspacing=0.5,
                       borderaxespad=0, ncol=ncols)

    # Bar plot for purity (replacing line plot)
    purity_bar_ax = plt.subplot(gs[5, 0])
    purity_values = plot_df["plot_purity"].values
    sample_indices = range(len(purity_values))
    purity_bar_ax.bar(sample_indices, purity_values, color='darkred', alpha=0.7, width=1.0)
    purity_bar_ax.set_xlim(-0.5, len(purity_values)-0.5)
    purity_bar_ax.set_ylim(0, 1)
    purity_bar_ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    purity_bar_ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'])
    purity_bar_ax.tick_params(axis='y', labelsize=12)
    purity_bar_ax.set_xticks([])
    purity_bar_ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    purity_bar_ax.text(-0.1, 0.5, "Tumor Purity", transform=purity_bar_ax.transAxes,
                      fontsize=16, ha='right', va='center', rotation=0)

    # Colorbars for fusion features
    cbar_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[8:11, 3:5], height_ratios=[1, 1], hspace=1)
    for i, im in enumerate(ims):
        cbar_ax = plt.subplot(cbar_gs[i, 0])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink=0.5, aspect=60)
        cbar.set_ticks([0, 20])
        cbar.set_ticklabels(["Min: 0", "≥20"])
        cbar.ax.tick_params(labelsize=14)
        cbar_ax.set_title(f"{feature_labels[i+3]}", fontsize=16, pad=15, rotation=0, va='bottom')

    plt.tight_layout(pad=2.0)
    plt.show()

# Example usage:
# create_patient_heatmap(merged)

def plot_fusion_heatmap_with_patients(final_df):
    """
    Plots a heatmap for fusion gene analysis across samples, including histology, tumor type, 
    patient data, and clonal fusion counts.

    Parameters:
        final_df (pd.DataFrame): DataFrame containing fusion gene data with 'SF#', 'Histology', 'Tumor', 'Patient', and numeric fusion metrics.
    """
    # Create a copy of the dataframe
    plot_df = final_df.copy()

    # Store original categorical values for legends
    histology_map = dict(enumerate(final_df["Histology"].astype("category").cat.categories))
    tumor_map = dict(enumerate(final_df["Tumor"].astype("category").cat.categories))
    patient_map = dict(enumerate(final_df["Patient"].astype("category").cat.categories))

    # Convert categorical data to numerical codes
    plot_df["Histology"] = plot_df["Histology"].astype("category").cat.codes
    plot_df["Tumor"] = plot_df["Tumor"].astype("category").cat.codes
    plot_df["Patient"] = plot_df["Patient"].astype("category").cat.codes

    # Select features to visualize
    features = ["Histology", "Tumor", "Patient", "STAR_clonal_in_sample", "Arriba_clonal_in_sample"]

    # Prepare heatmap data
    heatmap_data = plot_df.set_index(["SF#"])[features].T

    # Define custom colormaps
    histology_colors = sns.color_palette("Set1", n_colors=len(set(plot_df["Histology"])))
    tumor_colors = sns.color_palette("Set2", n_colors=len(set(plot_df["Tumor"])))
    patient_colors = sns.color_palette("muted", n_colors=60)  # 60 unique colors for patients

    colormaps = [ListedColormap(histology_colors), ListedColormap(tumor_colors), 
                 ListedColormap(patient_colors), "PuRd", "Blues", "Oranges"]
    feature_labels = ["Histology", "Tumor Type", "Patient", 
                      "STAR Clonal Fusions", "Arriba Clonal Fusions"]

    # Create figure
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(nrows=8, ncols=4, width_ratios=[20, 5, 5, 5], height_ratios=[1, 1, 1, 1, 1, 1, 2, 2], 
                  wspace=0.2, hspace=0.3)

    ims = []
    for i in range(len(features)):
        ax = plt.subplot(gs[i, 0])
        row_data = heatmap_data.iloc[i, :].values.reshape(1, -1)
        
        if i == 0:  
            vmin, vmax = -0.5, len(set(plot_df["Histology"])) - 0.5
        elif i == 1:  
            vmin, vmax = -0.5, len(set(plot_df["Tumor"])) - 0.5
        elif i == 2:  
            vmin, vmax = -0.5, 59.5  # Updated to match 60 patients
        else:  
            vmin, vmax = heatmap_data.iloc[i].min(), heatmap_data.iloc[i].max()
            avg = (vmin + vmax) / 2

        im = ax.imshow(row_data, aspect='auto', cmap=colormaps[i], vmin=vmin,
                       vmax=vmax, interpolation='none')
        if i >= 3:
            ims.append(im)
        
        num_columns = len(plot_df)
        for j in range(num_columns + 1):
            ax.vlines(j - 0.5, ymin=-0.5, ymax=0.5, color='black', linewidth=0.05)
            
        ax.set_ylabel(feature_labels[i], rotation=0, labelpad=50, fontsize=16, ha='right')
        ax.set_yticks([])
        ax.set_xticks([])

        if i == 0:  # Histology
            legend_ax = plt.subplot(gs[i, 1:3])
            legend_ax.axis('off')
            patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=histology_colors[j], 
                                 markersize=12, label=histology_map[j]) for j in histology_map]
            legend_ax.legend(handles=patches, loc='center left', fontsize=14, title='Histology', 
                            title_fontsize=16, frameon=False, handletextpad=0.5, labelspacing=0.7, 
                            borderaxespad=0, ncol=1)
        elif i == 1:  # Tumor Type
            legend_ax = plt.subplot(gs[i, 1:3])
            legend_ax.axis('off')
            patches = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=tumor_colors[j], 
                                 markersize=12, label=tumor_map[j]) for j in tumor_map]
            legend_ax.legend(handles=patches, loc='center left', fontsize=14, title='Tumor Type', 
                            title_fontsize=16, frameon=False, handletextpad=0.5, labelspacing=0.7, 
                            borderaxespad=0, ncol=1)

    # Patient legend
    patient_legend_ax = plt.subplot(gs[6, 1:4])
    patient_legend_ax.axis('off')
    patches = [plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=patient_colors[j], 
                         markersize=12, label=patient_map[j]) for j in patient_map]
    patient_legend_ax.legend(handles=patches, loc='center left', fontsize=12, title='Patient', 
                            title_fontsize=14, frameon=False, handletextpad=0.5, labelspacing=0.5, 
                            borderaxespad=0, ncol=4)

    # Colorbars
    cbar_gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[7:9, 3], height_ratios=[0.5, 0.5, 0.5], hspace=2)

    vmin_vals = [heatmap_data.iloc[i].min() for i in range(3, len(features))]
    vmax_vals = [heatmap_data.iloc[i].max() for i in range(3, len(features))]
    avg_vals = [(vmin_vals[i] + vmax_vals[i]) / 2 for i in range(len(vmin_vals))]

    for i, im in enumerate(ims):
        cbar_ax = plt.subplot(cbar_gs[i, 0])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink=0.5, aspect=60)
        cbar.set_ticks([vmin_vals[i], avg_vals[i], vmax_vals[i]])
        cbar.set_ticklabels([f"Min: {int(vmin_vals[i])}", f"Avg: {int(avg_vals[i])}", f"Max: {int(vmax_vals[i])}"])
        cbar.ax.tick_params(labelsize=14)
        cbar_ax.set_title(f"{feature_labels[i+3]}", fontsize=16, pad=15, rotation=0, va='bottom')

    plt.suptitle("Fusion Gene Analysis Across Samples", fontsize=24, y=0.98)
    plt.tight_layout(pad=2.0)
    plt.show()


import colorsys
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import ListedColormap
import pandas as pd

def plot_fusion_scatter(merged):
    """
    Plots a scatter-based visualization (marplots) for fusion gene analysis across samples, 
    including histology, tumor type, patient data, and clonal fusion counts.

    Parameters:
        merged (pd.DataFrame): DataFrame containing fusion gene data with 'Patient', 'join_key', 
                               'Histology', 'Tumor', and numeric fusion metrics.
    """
    # Create a new unique name for indexing
    merged['name'] = merged['Patient'] + merged['join_key']

    # Create a copy of the dataframe
    plot_df = merged.copy()

    # Get unique categories
    unique_histology = sorted(merged["Histology"].unique())
    unique_tumor = sorted(merged["Tumor"].unique())
    unique_patients = sorted(merged["Patient"].unique())

    # Mapping categorical values
    histology_map = {i: val for i, val in enumerate(unique_histology)}
    tumor_map = {i: val for i, val in enumerate(unique_tumor)}
    patient_map = {i: val for i, val in enumerate(unique_patients)}

    # Reverse mappings for encoding
    histology_code_map = {val: i for i, val in histology_map.items()}
    tumor_code_map = {val: i for i, val in tumor_map.items()}
    patient_code_map = {val: i for i, val in patient_map.items()}

    # Convert categorical data to numerical codes consistently
    plot_df["Histology_code"] = plot_df["Histology"].map(histology_code_map)
    plot_df["Tumor_code"] = plot_df["Tumor"].map(tumor_code_map)
    plot_df["Patient_code"] = plot_df["Patient"].map(patient_code_map)

    # Select features to visualize
    features = ["Histology_code", "Tumor_code", "Patient_code", "STAR_clonal_in_sample", "Arriba_clonal_in_sample"]
    feature_labels = ["Histology", "Tumor Type", "Patient", "STAR Clonal Fusions", "Arriba Clonal Fusions"]

    # Define colors for categorical variables
    histology_colors = sns.color_palette("Set1", n_colors=len(unique_histology))
    tumor_colors = sns.color_palette("Set2", n_colors=len(unique_tumor))
    patient_colors = sns.color_palette("muted", n_colors=len(unique_patients))

    # Define custom colormaps
    colormaps = {
        "Histology_code": histology_colors,
        "Tumor_code": tumor_colors,
        "Patient_code": patient_colors
    }

    # Create figure layout
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(nrows=len(features), ncols=2, width_ratios=[20, 5], height_ratios=[1, 1, 1, 1, 1], wspace=0.3, hspace=0.5)

    # Iterate through features and plot scatter (marplot)
    for i, feature in enumerate(features):
        ax = plt.subplot(gs[i, 0])

        if feature in colormaps:  # Categorical features
            scatter_colors = [colormaps[feature][code] for code in plot_df[feature]]
            ax.scatter(range(len(plot_df)), plot_df[feature], c=scatter_colors, alpha=0.7, s=50)
            ax.set_yticks(range(len(colormaps[feature])))
            ax.set_yticklabels([histology_map.get(code, "") if feature == "Histology_code" else
                                tumor_map.get(code, "") if feature == "Tumor_code" else
                                patient_map.get(code, "")
                                for code in range(len(colormaps[feature]))])
        else:  # Numerical fusion features
            ax.scatter(range(len(plot_df)), plot_df[feature], c='darkblue', alpha=0.7, s=50)
            ax.set_yticks(np.linspace(plot_df[feature].min(), plot_df[feature].max(), num=5))

        ax.set_xticks([])
        ax.set_ylabel(feature_labels[i], fontsize=14, rotation=0, labelpad=50, ha='right')

    # Add legends
    legend_ax = plt.subplot(gs[:, 1])
    legend_ax.axis('off')

    histology_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=histology_colors[j], 
                                    markersize=10, label=histology_map[j]) for j in histology_map]
    tumor_patches = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=tumor_colors[j], 
                                markersize=10, label=tumor_map[j]) for j in tumor_map]

    legend_ax.legend(handles=histology_patches + tumor_patches, loc='center left', fontsize=12, title="Legend",
                     title_fontsize=14, frameon=False, handletextpad=0.5, labelspacing=0.7, borderaxespad=0, ncol=1)

    plt.suptitle("Fusion Gene Analysis - Scatter Plot Representation", fontsize=20)
    plt.show()
