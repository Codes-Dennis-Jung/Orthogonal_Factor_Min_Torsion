import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import squareform
import seaborn as sns
import matplotlib.pyplot as plt

def factor_hcl(cor_mat, linkage_method="ward", k=None, direction_bars=True):
    dist_mat = squareform(1 - cor_mat)
    hcl = linkage(dist_mat, method=linkage_method)
    
    coph_corr = np.corrcoef(cophenet(hcl), dist_mat)[0, 1]
    print(f"Cophenetic Correlation between Dendrogram and Distance Matrix = {coph_corr:.2f}")

    hcl_labels = pd.DataFrame({'characteristic': cor_mat.index, 'hcl': pd.cut(np.arange(len(cor_mat)), bins=k, labels=np.arange(1, k+1))})
    
    hcl_col = np.tile(colours_theme[:8], int(np.ceil(k / 8)))[:k]
    
    plt.figure(figsize=(10, 8))
    dendrogram(hcl, labels=hcl_labels['hcl'].astype(str).values, color_threshold=0, above_threshold_color=hcl_col)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.show()

    return_list = {
        "cor": cor_mat,
        "labels": hcl_labels,
        "dend": hcl
    }
    
    if direction_bars:
        bar_colours = pd.merge(pd.DataFrame({"characteristic": cor_mat.columns}), char_info[['characteristic', 'direction']], on="characteristic", how="left")
        bar_colours['col_dir'] = np.where(bar_colours['direction'] == 1, "black", "white")
        colored_bars(colors=bar_colours['col_dir'], dend=hcl, rowLabels=["Long High"], y_shift=3, horiz=True)
        return_list['bar_colours'] = bar_colours
    
    return return_list

