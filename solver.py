import numpy as np

def m_favorable_downsampling(n, k, tp):
    # Calculate the mathematical optimal m of np.abs((k-m)/(n-m) - ptarget)
    # m represents the number of samples to remove
    m = (n * tp - k) / (tp - 1)
    
    # Round m to the nearest integer
    m = int(round(m))
    
    # Ensure m is within the valid range [0, k-1]
    if m < 1:
        m = 0
    elif m >= k:
        m = k - 1

    return m

def m_unfavorable_downsampling(n, k, tp):
    # Calculate the mathematical optimal m of np.abs((k-m)/(n-m) - ptarget)
    # m represents the number of samples to remove
    m = (n * tp - k) / (tp)
    
    # Round m to the nearest integer
    m = int(round(m))
    
    # Ensure m is within the valid range [0, n-k-1]
    if m < 1:
        m = 0
    elif m >= (n-k):
        m = n - k - 1

    return m


def m_favorable_upsampling(n, k, tp):
    # Calculate the mathematical optimal m of np.abs((k-m)/(n-m) - ptarget)
    # m represents the number of samples to remove
    m = (n * tp - k) / (1 - tp)
    
    # Round m to the nearest integer
    m = int(round(m))
    
    # Ensure m is within the valid range [0, k-1]
    if m < 1:
        m = 0
    return m


def m_unfavorable_upsampling(n, k, tp):
    # Calculate the mathematical optimal m of np.abs((k-m)/(n-m) - ptarget)
    # m represents the number of samples to remove
    m = -(n * tp - k) / (tp)
    
    # Round m to the nearest integer
    m = int(round(m))
    
    # Ensure m is within the valid range [0, k-1]
    if m < 1:
        m = 0

    return m