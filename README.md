# DBSCAN Clustering for Public Charging Infrastructure Optimization

This repository contains the Python implementation (`dbscan.py`) of the DBSCAN clustering algorithm, which is used as part of the research on optimizing public charging infrastructure in urban neighborhoods, as detailed in the accompanying research paper "Public Charging Infrastructure Optimization in Urban Neighborhood.pdf". The paper is published on 4th International Conference on Electrical, Communication and Computer Engineering in December 2023 as primary author, co-sponsored by IEEE and indexed in SCOPUS.

## Overview

The `dbscan.py` file implements the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm to identify densely packed areas suitable for the installation of electric vehicle (EV) charging stations in urban neighborhoods. The algorithm helps in determining the optimal locations based on the density of EVs and proximity to the existing infrastructure.

## Research Paper

The research paper provides an in-depth analysis of urban charging infrastructure requirements, proposing a novel approach to optimize the placement and distribution of public charging stations. The paper outlines the methodology, data analysis, and the application of the DBSCAN algorithm to achieve the research objectives.

## Dependencies

To run `dbscan.py`, you will need the following Python libraries:
- NumPy
- SciPy
- scikit-learn

Please ensure these libraries are installed in your Python environment.

## Usage

To use the script, follow these steps:
1. Download the `dbscan.py` file from this repository.
2. Ensure you have the required data in the correct format as expected by the script.
3. Run the script in your Python environment to perform DBSCAN clustering on your data.

## Data

The script is designed to work with geospatial data representing the locations of EVs or potential sites for charging stations. The data should be in a format compatible with the scikit-learn DBSCAN implementation.

