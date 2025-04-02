import random

import numpy as np
import pandas as pd
from preprocessing.fairfes.solver import m_favorable_downsampling, m_unfavorable_downsampling, m_favorable_upsampling, m_unfavorable_upsampling


def downsampling(df, tp=0.5, label='y', protected_attribute='z'):
    # NumPy Arrays
    """
    Perform downsampling to achieve fairness in the data.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to be downsampled.
    tp : float (default=0.5)
        The target probability of the favorable outcomes for each group.
    label : str, optional (default='y')
        The column name of the label (target) variable in the dataset.
    protected_attribute : str, optional (default='z')
        The column name of the protected attribute variable in the dataset.

    Returns
    -------
    balanced_df : pandas.DataFrame
        The downsampled dataset.
    """
    y = df[label].to_numpy()
    z = df[protected_attribute].to_numpy()
    z_groups, z_inverse = np.unique(z, return_inverse=True)
    ki_values = np.bincount(z_inverse, weights=y)
    ni_values = np.bincount(z_inverse)
    treatment_values = ki_values / ni_values
    remove_indices = []
    need_favorable_removal = np.where(treatment_values > tp)[0]
    need_unfavorable_removal = np.where(treatment_values < tp)[0]

    for i in need_favorable_removal:
        m_favorable = m_favorable_downsampling(ni_values[i], ki_values[i], tp)
        group_indices = np.flatnonzero((z_inverse == i) & (y == 1))
        remove_indices.extend(np.random.choice(group_indices, size=m_favorable, replace=False))

    for i in need_unfavorable_removal:
        m_unfavorable = m_unfavorable_downsampling(ni_values[i], ki_values[i], tp)
        group_indices = np.flatnonzero((z_inverse == i) & (y == 0))
        remove_indices.extend(np.random.choice(group_indices, size=m_unfavorable, replace=False))

    return df.reset_index(drop=True).drop(index=remove_indices).reset_index(drop=True)


def upsampling(df, tp=0.5, label='y', protected_attribute='z'):
    # NumPy Arrays
    """
    Upsamples the dataset to balance the favorable outcomes to the target probability tp.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to be upsampled.
    tp : float (default=0.5)
        The target probability of the favorable outcomes for each group.
    label : str, optional (default='y')
        The column name of the label (target) variable in the dataset.
    protected_attribute : str, optional (default='z')
        The column name of the protected attribute variable in the dataset.
    Returns
    -------
    balanced_df : pandas.DataFrame
        The upsampled dataset.
    """
    y = df[label].to_numpy()
    z = df[protected_attribute].to_numpy()
    z_groups, z_inverse = np.unique(z, return_inverse=True)
    ki_values = np.bincount(z_inverse, weights=y)
    ni_values = np.bincount(z_inverse)
    treatment_values = ki_values / ni_values

    add_indices = []
    need_unfavorable = np.where(treatment_values > tp)[0]
    need_favorable = np.where(treatment_values < tp)[0]

    for i in need_unfavorable:
        m_unfavorable = m_unfavorable_upsampling(ni_values[i], ki_values[i], tp)
        group_indices = np.flatnonzero((z_inverse == i) & (y == 0))
        add_indices.extend(np.random.choice(group_indices, size=m_unfavorable, replace=True))

    for i in need_favorable:
        m_favorable = m_favorable_upsampling(ni_values[i], ki_values[i], tp)
        group_indices = np.flatnonzero((z_inverse == i) & (y == 1))
        add_indices.extend(np.random.choice(group_indices, size=m_favorable, replace=True))

    upsampled_rows = df.iloc[add_indices]
    return pd.concat([df, upsampled_rows], ignore_index=True).reset_index(drop=True)
