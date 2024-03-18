# %%
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, silhouette_score, silhouette_samples
from scipy.spatial.distance import cosine
from scipy.cluster.vq import kmeans, vq
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import shutil
import tensorflow as tf
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from kerastuner.tuners import RandomSearch

# %%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# %%
# Create relative path method to access all the files in data/raw/train folder
def relative_path(path):
    return os.path.join(os.getcwd(), path)

# %%
def evaluate(final_df, stream,final_predictions):
    # Calculate F1 Score
    f1 = f1_score(final_df['Actual_Anomaly'], final_df['Predicted_Labels'])
    print(f"F1 Score: {f1}")
    # True Positives
    tp_filter = (final_df['Actual_Anomaly'] == 1) & (final_df['Predicted_Labels'] == 1)
    # True Negatives
    tn_filter = (final_df['Actual_Anomaly'] == 0) & (final_df['Predicted_Labels'] == 0)
    # False Positives
    fp_filter = (final_df['Actual_Anomaly'] == 0) & (final_df['Predicted_Labels'] == 1)
    # False Negatives
    fn_filter = (final_df['Actual_Anomaly'] == 1) & (final_df['Predicted_Labels'] == 0)

    # Precision
    precision = len(final_df[tp_filter]) / (len(final_df[tp_filter]) + len(final_df[fp_filter]))
    # Recall
    recall = len(final_df[tp_filter]) / (len(final_df[tp_filter]) + len(final_df[fn_filter]))

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f'False Negatives: {len(final_df[fn_filter])}')
    print(f'False Positives: {len(final_df[fp_filter])}')
    print(f'True Negatives: {len(final_df[tn_filter])}')
    print(f'True Positives: {len(final_df[tp_filter])}')
    
    # Add evaluation Metrics to .csv file for on sample where chan_id == stream
    final_predictions.loc[final_predictions['chan_id'] == stream, 'f1'] = f1
    final_predictions.loc[final_predictions['chan_id'] == stream, 'precision'] = precision
    final_predictions.loc[final_predictions['chan_id'] == stream, 'recall'] = recall
    final_predictions.loc[final_predictions['chan_id'] == stream, 'tp'] = len(final_df[tp_filter])
    final_predictions.loc[final_predictions['chan_id'] == stream, 'tn'] = len(final_df[tn_filter])
    final_predictions.loc[final_predictions['chan_id'] == stream, 'fp'] = len(final_df[fp_filter])
    final_predictions.loc[final_predictions['chan_id'] == stream, 'fn'] = len(final_df[fn_filter])
    
    # calculate range or sequence of anomalies predicted
    predicted_anomalies = []
    start = 0
    end = 0
    while end < len(final_df):
        slice = final_df.iloc[start:end+1]
        if not slice.empty:
            if slice['Predicted_Labels'].iloc[0] == 1:
                if start == 0:
                    start = end
            else:
                if start != 0:
                    end = end - 1
                    predicted_anomalies.append([start,end])
                    start = 0
        if end < len(final_df):
            end += 1
    final_predictions.loc[final_predictions['chan_id'] == stream, 'predicted_anomalies'] = str(predicted_anomalies)

    # Get the parent directory of the current working directory
    parent_dir = os.path.dirname(os.getcwd())

    # Save the CSV file to the reports directory in the current working directory
    final_predictions.to_csv(os.path.join(parent_dir, "reports", "final_predictions.csv"))
    return final_predictions

# %%
def find_optimal_k(X, k_range,stream):
    distortions = []  # Store distortion (inertia) for different K values

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=5, max_iter=300, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)  # Store the distortion (inertia) for the current K

    # Determine the optimal K using the elbow method
    optimal_k = determine_optimal_k(k_range, distortions,stream)

    print(f"Optimal K: {optimal_k}")

    # Fit KMeans again with the optimal k to get cluster labels and centers
    kmeans = KMeans(n_clusters=optimal_k, n_init=5, max_iter=300, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    cluster_centers = kmeans.cluster_centers_

    return optimal_k, cluster_labels, cluster_centers, kmeans

def determine_optimal_k(k_values, distortions,stream):
    """
    Determine the optimal K using a more robust method based on the distortion curve.

    Parameters:
    - k_values: List of K values.
    - distortions: List of distortions (inertia) for corresponding K values.

    Returns:
    - The optimal K value.
    """
    # Calculate the second derivative of distortions
    second_derivative = np.diff(np.diff(distortions))

    # Find the K with the maximum absolute second derivative
    optimal_k = k_values[np.argmax(np.abs(second_derivative)) + 1]

    # Plot the distortion curve and the selected K
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, distortions, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Distortion (Inertia)')
    plt.title('Optimal K Selection: Distortion vs. Number of Clusters (K)')

    # Highlight the selected K
    plt.axvline(x=optimal_k, color='r', linestyle='--', label='Optimal K')
    plt.legend()
    plt.show()

    # Get the parent directory of the current working directory
    parent_dir = os.path.dirname(os.getcwd())

    # Save the plot
    plt.savefig(os.path.join(parent_dir, 'reports', 'figures', f'{stream}_optimal_k.png'))
    
    return optimal_k

# %%
def append_ground_truth(test_results, labeled_anomalies, stream):
    anomaly_sequences = labeled_anomalies[labeled_anomalies['chan_id'] == stream]['anomaly_sequences'].tolist()
    actual_anomalies = [eval(seq) for seq in anomaly_sequences]
    
    # Initialize a binary array with zeros
    binary_labels = np.zeros(test_results.shape[0])

    # Check if actual_anomalies is a nested list
    if len(actual_anomalies) == 1 and isinstance(actual_anomalies[0], list) and len(actual_anomalies[0]) == 1:
        # Handle nested structure
        anomaly_sequence = actual_anomalies[0][0]
        start, end = anomaly_sequence[0], anomaly_sequence[1]
        binary_labels[start:end + 1] = 1
    else:
        # Handle non-nested structure (single-index anomalies)
        for sequence in actual_anomalies:
            if isinstance(sequence, int):
                binary_labels[sequence] = 1
    binary_frame = pd.DataFrame(binary_labels, columns=['Actual_Anomaly'])
    
    # Concatenate the DataFrames
    final_df = pd.concat([test_results, binary_frame], axis=1)
    return final_df


# %%
def create_sequences(data, timesteps):
    X, Y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:(i + timesteps)])
        Y.append(data[i + timesteps])
    return np.array(X), np.array(Y)

# %%
def LSTMAE(train_df,sequence_length):
    n_features = train_df.shape[2]
    latent_dim = 12           # Latent dimension set to 12 as requested

    # Define the model
    model = Sequential()
    
    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, n_features), return_sequences=True))
    model.add(LSTM(latent_dim, activation='relu', return_sequences=False))
    model.add(RepeatVector(sequence_length))  # Set the same number of timesteps for the decoder

    # Decoder
    model.add(LSTM(latent_dim, activation='relu', return_sequences=True))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(20, activation='relu')))
    model.add(TimeDistributed(Dense(n_features)))

    model.compile(optimizer='adam', loss='mse')
    return model

# %%
parent_dir = os.path.dirname(os.getcwd())
channels_folder = os.path.join(parent_dir, 'data','raw','train')
channels = os.listdir(channels_folder)

labeled_anomalies_file = os.path.join(parent_dir, 'data','processed','labeled_anomalies.csv')
labeled_anomalies = pd.read_csv(labeled_anomalies_file)

final_results= pd.DataFrame()

final_predictions = labeled_anomalies.copy()
# Add a columns for f1 score, precision, recall, tp, tn, fp, fn to final predictions dataframe
final_predictions['f1'] = 0
final_predictions['precision'] = 0
final_predictions['recall'] = 0
final_predictions['tp'] = 0
final_predictions['tn'] = 0
final_predictions['fp'] = 0
final_predictions['fn'] = 0
final_predictions['predicted_anomaly_sequences'] = 0
for channel in channels:
    stream = channel[:-4]
    print(f'Currently working on {stream} Channel')

    train_df = pd.DataFrame(np.load(os.path.join(parent_dir, 'data','raw','train',f'{channel}')))
    test_df = pd.DataFrame(np.load(os.path.join(parent_dir, 'data','raw','test',f'{channel}')))
    
    timesteps = 100  # sequence steps
    train_X, train_Y = create_sequences(train_df.values, timesteps)
    test_X, test_Y = create_sequences(test_df.values, timesteps)

    
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
    autoEncoder = LSTMAE(train_X, timesteps)
    autoEncoder.fit(train_X, train_Y, epochs=300, batch_size=32, validation_split=0.1, callbacks=[early_stop, reduce_lr])

    # Extract latent space
    latent_model = Model(inputs=autoEncoder.input, outputs=autoEncoder.layers[1].output)
    
    # Save Latent Model
    latent_model.save(os.path.join(parent_dir, 'models', 'latent_model', f'{stream}_latent_model.h5'))
    autoEncoder.save(os.path.join(parent_dir, 'models', 'autoencoder_model', f'{stream}_autoencoder_model.h5'))

    # Get reconstruction error
    test_reconstructions = autoEncoder.predict(test_X)
    test_mse = mean_squared_error(test_X, test_reconstructions)
    final_results['Reconstruction_Error'] = test_mse

    # Get latent space representation for Kmeans    
    train_latent = latent_model.predict(train_X)
    test_latent = latent_model.predict(test_X)

    # Aggregate the latent space (e.g., using max)
    train_latent_aggregated = np.max(train_latent, axis=1)
    test_latent_aggregated = np.max(test_latent, axis=1)
 
    train_optimal_k, train_cluster_labels, train_cluster_centers, kmeans = find_optimal_k(train_latent_aggregated, range(2, 10),stream)   
        
    # Compute the cosine distance between each row of test_latent_representation and the corresponding row of test_cluster_centers
    test_cluster_centers = kmeans.cluster_centers_  # Define test_cluster_centers
    test_cluster_labels = kmeans.predict(test_latent_aggregated)  # Define test_cluster_labels
    anomaly_scores = []
    
    for i in range(test_latent_aggregated.shape[0]):
        center = test_cluster_centers[test_cluster_labels[i],:]
        latent_vector = test_latent_aggregated[i,:]  
        score = euclidean(latent_vector, center)  # Use Euclidean distance here
        anomaly_scores.append(score)

    anomaly_scores = np.array(anomaly_scores)
    final_results['Anomaly_Score_Kmean'] = anomaly_scores

# %%



