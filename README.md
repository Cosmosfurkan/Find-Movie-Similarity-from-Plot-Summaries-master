# Movie Plot Clustering and Similarity Analysis

This repository contains code for analyzing movie plots through clustering and similarity analysis. The project aims to cluster movie plots based on their content and visualize their similarity using a dendrogram.

## Overview

The project utilizes natural language processing techniques to preprocess and analyze movie plots. It clusters movies based on the similarity of their plots using KMeans clustering and visualizes the clustering results using a dendrogram.

## Getting Started

To run the code, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have the necessary dependencies installed. You can install them using `pip install -r requirements.txt`.
3. Run the provided Python script.

## Features

- **Preprocessing**: Tokenization, stemming, and TF-IDF vectorization are performed on movie plots to prepare them for clustering.
- **Clustering**: KMeans clustering is applied to group similar movie plots together.
- **Visualization**: A dendrogram is generated to visualize the hierarchical clustering of movie plots.
- **Similarity Analysis**: The code identifies the movie most similar to a given movie (e.g., Braveheart).

## Usage

1. Prepare your movie plot dataset in CSV format, ensuring it contains a column named "plot" with the text of each movie plot.
2. Modify the script to load your dataset and adjust any parameters as needed.
3. Run the script to perform clustering and visualize the results.

## Example

In the provided code, movie plots are clustered and visualized using a dendrogram. Additionally, the movie most similar to Braveheart is identified as Gladiator.

## Contributing

Contributions are welcome! Feel free to submit bug fixes, feature enhancements, or suggestions via pull requests.

## Acknowledgments

Special thanks to the libraries used in this project, including matplotlib, numpy, pandas, nltk, and scikit-learn.
