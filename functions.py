import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.metrics import silhouette_samples, silhouette_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score
from scipy.spatial import distance_matrix 
import time
import sklearn.metrics as metrics
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering, OPTICS, MeanShift, estimate_bandwidth, DBSCAN, AffinityPropagation ,SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances_argmin
import scipy.cluster.hierarchy as hcluster
from  scipy.cluster import hierarchy
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')



def numerical_attributes_plot_distrbution(num_df):
    plt.figure(figsize=(20,20))
    for i in range(1, 10):
        plt.subplot(4, 3, i)
        sns.distplot(num_df[num_df.columns[i-1]],bins=14)

def process_outliers(df):
    for c in df:
        Q3, Q1 = np.percentile(df[c], [75,25])
    
    IQRange = Q3 - Q1
    minx = Q1 - (IQRange*1.5)
    maxx = Q3 + (IQRange*1.5)
    
    dt = df.drop(df.loc[df[c]< minx,att].index) 
    dt = df.drop(df.loc[df[c]> maxx,att].index) 
    
    return dt 


def min_max_scaled(cleaned_data):
    df_max_scaled = cleaned_data.copy()
    df_cols = df_max_scaled.columns
    min_max = MinMaxScaler()
    df_max_scaled = min_max.fit_transform(df_max_scaled)
    df_max_scaled = pd.DataFrame(df_max_scaled, columns=[df_cols])
    
    return df_max_scaled

from sklearn.decomposition import PCA

def plot_PCA_graph(X):
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title('Explained variance ratio', fontsize = 20)
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
        
def plot_corr_matrix(num_df):
    # correlation matrix
    num_corr = num_df.corr()
    plt.figure(figsize=(10,6))
    sns.heatmap(num_corr, annot=True, fmt=".3f",vmin=-1, vmax=1, linewidths=.5, cmap = sns.color_palette("RdBu", 100))
    plt.yticks(rotation=0)
    plt.show()


def silhouette_score(estimator, X,y):
    clusters = estimator.fit_predict(X)
    score = metrics.silhouette_score(X, clusters,metric='euclidean')
    return score

def silhouette_score2(estimator, X,y):
    clusters = estimator.fit_predict(X)
    n = len(np.unique(estimator.labels_))
    if n>1:
        score = metrics.silhouette_score(X, estimator.labels_)
        return score
    else:
        return 0

def silhouette_score3(estimator, X,y):
    clusters = estimator.fit_predict(X)
    #print("Number of labels: ",estimator,estimator.labels_)
    n = len(np.unique(clusters))
    if n>1:
        score = metrics.silhouette_score(X, clusters)
        #print("Number of labels: ",estimator,estimator.labels_)
        return score
    else:
        return 0

def estimator_evaluation(estimator, name, data , y):
    dt = {}
    estimator.fit(data)
    estimator_labels = estimator.fit_predict(X)
    labels = y.values.flatten()
    
    #intrinsic measures
    dt["silhouette_score"]=  metrics.silhouette_score(data, estimator.labels_,metric='euclidean')
    dt["davies_bouldin_score"] = metrics.davies_bouldin_score(data, estimator_labels)
    dt["calinski_harabasz_score"] = metrics.calinski_harabasz_score(data, estimator_labels)
   
    
    return dt
    
    
def estimator_evaluation2(estimator, name, data , y):
    dt = {}
    estimator.fit(data)
    estimator_labels = estimator.fit_predict(X)
    labels = y.values.flatten()
    
    #intrinsic measures
    dt["silhouette_score"]=  metrics.silhouette_score(data, estimator_labels,metric='euclidean')
    dt["davies_bouldin_score"] = metrics.davies_bouldin_score(data, estimator_labels)
    dt["calinski_harabasz_score"] = metrics.calinski_harabasz_score(data, estimator_labels)
  
    
    
    return dt







































