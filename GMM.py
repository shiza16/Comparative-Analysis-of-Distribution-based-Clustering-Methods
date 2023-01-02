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
import functions

def GMM_with_hyperparaeters(X):
    
    '''
    GMM with n_components = 2
    n_components refers to the number of mixture components.
    
    '''
    
    gmm1 = GaussianMixture(n_components = 2)
    gmm1.fit(X)
    gmm1_labels = gmm1.fit_predict(X)
    gmm1_dataset = pd.DataFrame(X.copy())
    gmm1_dataset.loc[:,'Cluster'] = gmm1_labels
    gmm1_dataset.Cluster.value_counts().to_frame()
    
    gmm_df1 = pd.DataFrame(estimator_evaluation2(gmm1, 'GaussianMixture ', X ), index=["GaussianMixture with n_components only"])

    '''
    GMM with n_components and covariance_type using Grid Search
    covariance_type parameter controls the degrees of freedom in the shape of cluster.
    '''
    
    gmm_grid = {
            "covariance_type": ['full', 'tied', 'diag', 'spherical'],
            "n_components": range(2, 10)
           }
    gmm_grid
    gmm_model2 = GaussianMixture()
    gmm_params2 = GridSearchCV(gmm_model2,gmm_grid,scoring=silhouette_score3,cv= 10).fit(X)
    print(gmm_params2.best_params_)
    gmm2 = GaussianMixture(covariance_type = gmm_params2.best_params_['covariance_type'],
                          n_components = gmm_params2.best_params_['n_components'])

    gmm2.fit(X)
    gmm2_labels = gmm2.fit_predict(X)
    gmm2_dataset = pd.DataFrame(X.copy())
    gmm2_dataset.loc[:,'Cluster'] = gmm2_labels
    gmm2_dataset.Cluster.value_counts().to_frame()
    
    gmm_df2 = pd.DataFrame(estimator_evaluation2(gmm2, 'GaussianMixture ', X), index=["GaussianMixture with covariance_type"])

    
    '''
    GMM with n_components and max_iter using Grid Search
    max_iter refers to the number of EM iterations to perform.
    '''
    
    gmm_grid = {
        "max_iter": [50, 100, 150, 200, 250, 300],
        "n_components": range(2, 10)
       }
    gmm_grid
    gmm_model3 = GaussianMixture()
    gmm_params3 = GridSearchCV(gmm_model3,gmm_grid,scoring=silhouette_score3,cv= 10).fit(X)
    print(gmm_params3.best_params_)
    gmm3 = GaussianMixture(max_iter = gmm_params3.best_params_['max_iter'],
                          n_components = gmm_params3.best_params_['n_components'])

    gmm3.fit(X)
    gmm3_labels = gmm3.fit_predict(X)
    gmm3_dataset = pd.DataFrame(X.copy())
    gmm3_dataset.loc[:,'Cluster'] = gmm3_labels
    gmm3_dataset.Cluster.value_counts().to_frame()


    gmm_df3 = pd.DataFrame(estimator_evaluation2(gmm3, 'GaussianMixture ', X), index=["GaussianMixture with max_iter"])
    
    '''
    GMM with n_components and init_params using Grid Search
    init_params refers to Initialization methods that helps to generate the initial centers for the model components
    '''
    
    gmm_grid = {
        "init_params": ['kmeans', 'k-means++', 'random', 'random_from_data'],
        "n_components": range(2, 10)
       }
    gmm_grid
    gmm_model4 = GaussianMixture()
    gmm_params4 = GridSearchCV(gmm_model4,gmm_grid,scoring=silhouette_score3,cv= 10).fit(X)
    print(gmm_params4.best_params_)
    gmm4 = GaussianMixture(init_params = gmm_params4.best_params_['init_params'],
                          n_components = gmm_params4.best_params_['n_components'])

    gmm4.fit(X)
    gmm4_labels = gmm4.fit_predict(X)
    gmm4_dataset = pd.DataFrame(X.copy())
    gmm4_dataset.loc[:,'Cluster'] = gmm4_labels
    gmm4_dataset.Cluster.value_counts().to_frame()

    gmm_df4 = pd.DataFrame(estimator_evaluation2(gmm4, 'GaussianMixture ', X ), index=["GaussianMixture with init_params"])

    '''
    GMM with n_components and n_init using Grid Search
    n_init refers to the number of initializations to be performed in GMM.
    '''
    
    gmm_grid = {
            "n_init": range(1, 6),
            "n_components": range(2, 10)
           }
    gmm_grid
    gmm_model5 = GaussianMixture()
    gmm_params5 = GridSearchCV(gmm_model5,gmm_grid,scoring=silhouette_score3,cv= 10).fit(X)
    print(gmm_params5.best_params_)
    gmm5 = GaussianMixture(n_init = gmm_params5.best_params_['n_init'],
                          n_components = gmm_params5.best_params_['n_components'])

    gmm5.fit(X)
    gmm5_labels = gmm5.fit_predict(X)
    gmm5_dataset = pd.DataFrame(X.copy())
    gmm5_dataset.loc[:,'Cluster'] = gmm5_labels
    gmm5_dataset.Cluster.value_counts().to_frame()

    gmm_df5 = pd.DataFrame(estimator_evaluation2(gmm5, 'GaussianMixture ', X), index=["GaussianMixture with n_init"])
    gmm_result = gmm_df1.append([gmm_df2, gmm_df3,gmm_df4,gmm_df5])


    return gmm_result



































