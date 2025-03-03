from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gc
import pickle

def create_sequences_3(df, window_size=30, test_size=0.1, val_size=0.1):
    """
    Generates scaled sequences and splits into training, validation, and testing sets.
    
    Parameters:
    df (pd.DataFrame): The input dataframe with engine data.
    window_size (int): Number of time steps in each sequence.
    test_size (float): Fraction of data to reserve for testing.
    val_size (float): Fraction of data to reserve for validation.

    Returns:
    X_train, X_val, X_test, y_train, y_val, y_test, feature_scaler, target_scaler (np.ndarray): 
    Arrays of train/val/test sequences and targets, feature and target scalers.
    """
    features = [col for col in df.columns if col not in ['id', 'cycle', 'remaining_cycles']]
    target_column = 'remaining_cycles'
    
    # Initialize scalers
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Lists to collect sequences and targets
    X_sequences = []
    y_targets = []
    
    # Group by engine ID and create sliding windows before splitting
    for engine_id, engine_data in df.groupby('id'):
        engine_data = engine_data.sort_values(by='cycle')  # Sort by cycle to keep temporal order
        
        # Create sliding windows
        for i in range(len(engine_data) - window_size):
            X_sequence = engine_data[features].iloc[i:i + window_size].values
            y_target = engine_data[target_column].iloc[i + window_size - 1]  # Target is RUL of last cycle in window
            
            X_sequences.append(X_sequence)
            y_targets.append(y_target)
    
    # Convert to numpy arrays
    X_sequences = np.array(X_sequences)
    y_targets = np.array(y_targets)
    
    # Split data into training+validation and testing
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_sequences, y_targets, test_size=test_size, random_state=42)
    
    # Further split the training+validation data into training and validation
    val_ratio_adjusted = val_size / (1 - test_size)  # Adjust validation size proportionally
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=42)
    
    # Fit the scalers on the training data
    feature_scaler.fit(X_train.reshape(-1, X_train.shape[2]))  # Reshape for fitting scaler
    target_scaler.fit(y_train.reshape(-1, 1))  # Reshape target for fitting scaler
    
    # Apply the scalers to the data
    X_train = feature_scaler.transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)  # Reshape back after scaling
    X_val = feature_scaler.transform(X_val.reshape(-1, X_val.shape[2])).reshape(X_val.shape)
    X_test = feature_scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
    
    y_train = target_scaler.transform(y_train.reshape(-1, 1))
    y_val = target_scaler.transform(y_val.reshape(-1, 1))
    y_test = target_scaler.transform(y_test.reshape(-1, 1))
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_scaler, target_scaler


def dataReader():
    df = pd.read_excel("D:\\Projects\\BDA\\aircraft_reliability\\data\\PM_train.xlsx")
    df_truth = pd.read_excel("D:\\Projects\\BDA\\aircraft_reliability\\data\\PM_truth.xlsx")
    df_merged = pd.merge(df, df_truth, on='id')
    # Step 1: Get the maximum cycle for each engine
    max_cycle_per_engine = df.groupby('id')['cycle'].max().reset_index()
    max_cycle_per_engine.columns = ['id', 'max_cycle']

    # Step 2: Merge the maximum cycle with the df_truth to get the actual failure cycle
    df_merged = pd.merge(max_cycle_per_engine, df_truth, on='id')

    # Step 3: Calculate the actual failure cycle (when engine will fail)
    df_merged['failure_cycle'] = df_merged['max_cycle'] + df_merged['more']

    # Step 4: Merge this back with the main DataFrame to compute remaining cycles
    df = pd.merge(df, df_merged[['id', 'failure_cycle']], on='id')

    # Step 5: Calculate remaining cycles for each row by subtracting the current cycle from the failure cycle
    df['remaining_cycles'] = df['failure_cycle'] - df['cycle']
    df = df.drop('failure_cycle',axis=1)
    return df

def evaluate_model(X_train_1, X_train_2, y_train_1, y_train_2, model_1, model_2,target_scaler_1,target_scaler_2):
    """
    Evaluates two models on sequences with different lengths and computes average MAE and RMSE.
    
    Parameters:
    X_train_1, X_train_2 (np.ndarray): Sequences with different lengths.
    y_train_1, y_train_2 (np.ndarray): Respective targets for the sequences.
    model_1, model_2: The two trained models to evaluate.
    
    Returns:
    avg_rmse, avg_mae (float): Average RMSE and MAE for both models.
    """
    
    # Predictions from the models
    y_pred_1 = model_1.predict(X_train_1)
    y_pred_2 = model_2.predict(X_train_2)
    
    # Rescale predictions and targets back to original scale
    y_pred_1_rescaled = target_scaler_1.inverse_transform(y_pred_1.reshape(-1, 1))
    y_pred_2_rescaled = target_scaler_2.inverse_transform(y_pred_2.reshape(-1, 1))
    
    y_train_1_rescaled = target_scaler_1.inverse_transform(y_train_1.reshape(-1, 1))
    y_train_2_rescaled = target_scaler_2.inverse_transform(y_train_2.reshape(-1, 1))
    
    # Calculate MAE and RMSE for both models
    mae_1 = mean_absolute_error(y_train_1_rescaled, y_pred_1_rescaled)
    rmse_1 = np.sqrt(mean_squared_error(y_train_1_rescaled, y_pred_1_rescaled))
    
    mae_2 = mean_absolute_error(y_train_2_rescaled, y_pred_2_rescaled)
    rmse_2 = np.sqrt(mean_squared_error(y_train_2_rescaled, y_pred_2_rescaled))
    
    # Calculate average MAE and RMSE
    avg_mae = (mae_1 + mae_2) / 2
    avg_rmse = (rmse_1 + rmse_2) / 2
    
    return avg_rmse, avg_mae


def plot_rul(y_true, y_pred, sample_size=200, random_seed=42):
    """
    Plots predicted RUL vs true RUL for a subset of the data to keep the plot readable.

    Parameters:
    y_true (np.ndarray): True RUL values.
    y_pred (np.ndarray): Predicted RUL values.
    sample_size (int): Number of points to sample for plotting.
    random_seed (int): Random seed for consistent sampling.

    Returns:
    None
    """
    # Ensure both arrays are reshaped to 1D
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    # Ensure equal lengths
    assert len(y_true) == len(y_pred), "True and predicted RUL arrays must have the same length."

    # Randomly sample points to avoid overcrowding the plot
    np.random.seed(random_seed)
    indices = np.random.choice(len(y_true), size=sample_size, replace=False)
    
    # Extract samples
    y_true_sample = y_true[indices]
    y_pred_sample = y_pred[indices]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true_sample, y_pred_sample, alpha=0.7, label='Predicted vs True')
    plt.plot([y_true_sample.min(), y_true_sample.max()],
             [y_true_sample.min(), y_true_sample.max()],
             color='red', linestyle='--', label='Ideal Prediction')
    plt.title('Predicted RUL vs True RUL')
    plt.xlabel('True RUL')
    plt.ylabel('Predicted RUL')
    plt.legend()
    plt.grid()
    plt.show()

def plot_rul_line(y_true, y_pred, max_points=500):
    """
    Plots predicted RUL and true RUL as line plots against test indices.
    
    Parameters:
    y_true (np.ndarray): True RUL values.
    y_pred (np.ndarray): Predicted RUL values.
    max_points (int): Maximum number of points to plot to avoid overcrowding.

    Returns:
    None
    """
    # Ensure both arrays are reshaped to 1D
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    # Truncate data if it exceeds max_points
    if len(y_true) > max_points:
        step = len(y_true) // max_points
        indices = np.arange(0, len(y_true), step)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
        x_indices = indices
    else:
        x_indices = np.arange(len(y_true))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_indices, y_true, label='True RUL', color='blue', linewidth=2, alpha=0.8)
    plt.plot(x_indices, y_pred, label='Predicted RUL', color='orange', linestyle='--', linewidth=2, alpha=0.8)
    plt.title('True RUL vs Predicted RUL')
    plt.xlabel('Test Index')
    plt.ylabel('RUL')
    plt.legend()
    plt.grid(True)
    plt.show()