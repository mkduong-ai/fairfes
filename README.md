# FairFES - Fair Fast Exact Sampling Methods for Classification

Python implementation of "FairFES - Fair Fast Exact Sampling Methods for Classification". 
This repository provides the code for making decision-makers fair. 

## Example
```python
from helper import *
from samplers import upsamling, downsampling

# Create Sample DataFrame with acceptance rates of 20%, 40% and 70%
df = generate_dataframe(100000, 0.2, 0.4, 0.7) 
print_pisy0(df)  # [0.7956014362657091, 0.5997706422018348, 0.3013993541442411]

# Use upsampling with a desired target acceptance rate of 0.5 for each social group
df_up = upsampling(df, 0.5)  # Elapsed time in seconds: 0.0078
print_pisy0(df_up)  # [0.5, 0.5, 0.5]

# Use downsampling with a desired target acceptance rate of 0.5 for each social group
df_down = downsampling(df, 0.5)  # Elapsed time in seconds: 0.0085
print_pisy0(df_down)  # [0.5, 0.5, 0.5]
```

## File Overview
- **solver.py**: Contains all the functions to calculate the number of favorable and unfavorable observations to remove or add.
- **samplers.py**: Contains the upsampling and downsampling algorithms.
- **helper.py**: A set of utility functions

## Setup
This package relies on:
- `numpy`
- `pandas`
