import numpy as np
import pandas as pd


def print_pis(df, label='y', protected_attributes='z'):
    # Print the probability of favorable outcome for each group
    y = df[label].to_numpy()
    z = df[protected_attributes].to_numpy()
    z_groups = np.unique(z)
    
    pis = [np.sum((y == 1) & (z == z_group)) / np.sum(z == z_group) for z_group in z_groups]
    
    return pis


def get_ni(df, protected_attributes='z'):
    z = df[protected_attributes].to_numpy()
    z_groups = np.unique(z)
    return [np.sum(z == z_group) for z_group in z_groups]


def get_ki(df, label='y', protected_attributes='z'):
    y = df[label].to_numpy()
    z = df[protected_attributes].to_numpy()
    z_groups = np.unique(z)
    return [np.sum((y == 1) & (z == z_group)) for z_group in z_groups]


def print_is(df):
    # Print the count of favorable outcome for each group
    y = df['y'].to_numpy()
    z = df['z'].to_numpy()
    z_groups = np.unique(z)
    
    pis = [np.sum((y == 1) & (z == z_group)) for z_group in z_groups]
    
    return pis


def print_pisy0(df):
    # Print the probability of favorable outcome for each group
    y = df['y'].to_numpy()
    z = df['z'].to_numpy()
    z_groups = np.unique(z)
    
    pis = [np.sum((y == 0) & (z == z_group)) / np.sum(z == z_group) for z_group in z_groups]
    
    return pis


def generate_dataframe(n, *p):
    # Generate z column with random values representing groups
    z = np.random.choice(np.arange(len(p)), size=n)
    
    # Generate y column based on the z value
    y = np.array([np.random.choice([1, 0], p=[p[z[i]], 1-p[z[i]]]) for i in range(n)])
    
    # Create the DataFrame
    df = pd.DataFrame({'y': y, 'z': z})
    return df