import krezi
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

def standardize_array(X):
    X_means = X.mean(axis = 0)
    X_stds = X.std(axis = 0)
    return (X - X_means) / X_stds 

K = 5
MAX_TRAINING_STEPS = 1000

if __name__ == '__main__':
    df = pd.read_csv("/Users/aadilzikre/Documents/Personal/notebooks/Rough/housing.csv")
    df.drop(['longitude', 'latitude', 'ocean_proximity'], axis=1, inplace=True)
    df = df.dropna()
    df.info()
    df.to_csv("housing_cpp.csv", index=False)

    X = df.values
    X = standardize_array(X)

    K_centroids = np.random.randn(K, X.shape[1])
    K_distances = np.zeros((X.shape[0], K))
    k_centroids_hist = np.zeros((MAX_TRAINING_STEPS, K, X.shape[1]))
    
    CONVERGED_AT = MAX_TRAINING_STEPS - 1 

    for i in range(MAX_TRAINING_STEPS):
        if i%10==0 : krezi.log_info(f"{i} steps done!!!")
        
        # calculate distances from the centroids
        for k in range(K):
            K_distances[:, k] = np.sum(np.power((X - K_centroids[k]), 2), axis = 1)
        
        # Assign labels as per the distance
        labels_ = K_distances.argmin(axis=1)
        
        # update centroids with new mean
        for k in range(K):
            K_centroids[k] = X[labels_==k,:].mean(axis=0)
        k_centroids_hist[i] = K_centroids
        
        # break if centroids stop changing
        if i > 0:
            if np.power((k_centroids_hist[i-1] - k_centroids_hist[i]), 2).sum() == 0:
                krezi.log_info(f"Training Converged at {i} steps!!!")
                CONVERGED_AT = i
                break

    if False: # Optional
        tsne = TSNE(n_components=2, random_state=1)
        projections = tsne.fit_transform(X)

        fig = px.scatter(
            projections, x=0, y=1,
            color=[str(i) for i in labels_]
        )
        fig.show()

        centroid_labels = list(range(K)) * CONVERGED_AT

        tsne = TSNE(n_components=2, random_state=1)
        projections = tsne.fit_transform(k_centroids_hist[:CONVERGED_AT, :, :].reshape(-1, X.shape[1]))

        fig = px.scatter(
            projections, x=0, y=1,
            color=[str(i) for i in centroid_labels]
        )
        fig.show()