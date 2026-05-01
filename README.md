# IDAI-710 Final Project

This repository contains the code for our final group project. We built artificial intelligence models to diagnose six different neurodegenerative diseases. The models use a combination of 3D MRI brain scans and standard clinical data to make predictions.

## Repository Structure

The codebase is split into three main folders so each group member can manage their own architecture and experiments.

* **Abraham** Contains the code for HyperFuseNet. This model uses an end to end approach where it processes the brain scans and builds a dynamic patient relationship graph at the exact same time.
* **Jacob** Contains the code for AWHGCN. This model uses a decoupled approach where it compresses the brain scans first before building the patient relationship graph.
* **Dan** Contains the code and experiments for Dan's specific modeling approach.

## Data and Setup

The dataset for this project relies on a restricted clinical cohort. To protect patient privacy and prevent massive file uploads the local data folder and all raw spreadsheet files are ignored by Git. 

To run the code you will need to download the dataset locally and place it in the correct directory as referenced in the individual notebooks.

## Requirements

The models are built using standard machine learning and data science libraries. There may be slight differences between projects, but each folder should have it's own requirements.txt file. Bare minimum, you will need the following installed in your environment to run the training scripts.

* PyTorch
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn