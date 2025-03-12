# plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr, roc_curve, auc
import scipy.cluster.hierarchy as sch
from matplotlib.colors import ListedColormap
from collections import Counter
import itertools
from sklearn.metrics import confusion_matrix

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


###############################################################################
# End of Module
###############################################################################
