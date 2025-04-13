# Anomaly Detection in Manufacturing

This repository contains the codebase for the thesis Deep Learning for Industrial Anomaly Detection
It includes model implementations, a multithreaded real-time pipeline, and LLM-based anomaly explanation using a custom dual-view dataset.

##  Project Structure

- `models/` - Source code for model training and evaluation. All finished evaluations are available in <model>_eval.ipynb
- `data/` - Folder structure the paperclips dataset and inference videos
- `pipeline/` - Source code for real time detection pipeline
- `utils/` - Utility code for data preprocessing
- `README.md` - This file
- `pipeline.ipynb` - Real time anomaly detection pipeline runner notebook
- `envs/` - YAML Environment files for each model 


## Note on model code
All model files are sourced from their original GitHub repositories and remain unmodified unless explicitly stated otherwise

## Requirements

Each model has its own Conda environment to avoid dependency conflicts.  
Environment files are located in the `envs/` directory.

### Available Environments

| Model       | Environment File           |
|-------------|----------------------------|
| PatchCore   | `envs/patchcore_env.yml`   |
| DRAEM       | `envs/draem.yml`       |
| RD4AD       | `envs/rd4ad.yml`       |
| FastFlow    | `envs/fastflow_env.yml`    |

>  If you intend to run the **real-time multithreaded pipeline**, ensure the `patchcore_env` is installed and activated.

## Installation

Clone the repository and set up the desired environment:

```bash
git clone https://github.com/yourusername/your-thesis-repo.git
cd your-thesis-repo

# Create and activate the PatchCore environment for the pipeline
conda env create -f envs/patchcore_env.yml
conda activate patchcore_env
