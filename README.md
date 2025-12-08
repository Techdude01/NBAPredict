## NBA Playoff Prediction

This project predicts whether an NBA team will make the playoffs based on season-long stats. It explores three model families—**logistic regression**, **k-nearest neighbors (KNN)**, and **neural networks**—and compares their performance across several feature transformations.

### What this repo contains

- **Data prep** (`NBAPredict.ipynb`): cleans raw Kaggle game logs, aggregates per–team-season stats, and builds train/validation/test splits.
- **Exploratory analysis** (`exploratory_data_analysis.ipynb`): sanity checks, distributions, and basic relationships with the playoff target.
- **Model notebooks**
  - `LogPredict.ipynb`: custom logistic regression trained via gradient ascent with L2 regularization.
  - `KNNPredict.ipynb`: KNN models using scikit-learn with a grid over K, distance metric, and weighting.
  - `NeuralNetworkPredict.ipynb`: TensorFlow/Keras neural nets over different architectures and regularization settings.
- **Comparison** (`ModelComparison.ipynb`): loads the best run from each model family and summarizes accuracy, precision, recall, F1, and AUC.

### Quick start

1. Create and activate a Python environment (3.10+ recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on `.env.example` (paths and settings), or copy it directly:
   ```bash
   cp .env.example .env
   ```
4. Open the notebooks in Jupyter or VS Code / Cursor and run them in this order:
   1. `NBAPredict.ipynb` (data prep)
   2. `LogPredict.ipynb`, `KNNPredict.ipynb`, `NeuralNetworkPredict.ipynb` (training)
   3. `ModelComparison.ipynb` (evaluation + visuals)

### Notes on modeling

- Logistic regression is implemented from scratch to mirror a course homework, using gradient ascent with L2 regularization.
- KNN and neural networks use **scikit-learn** and **TensorFlow/Keras**, which are standard tools for these model types.
- The neural networks, in particular, lean on Keras to keep complex architectures and training loops manageable.

### Interpreting results

The comparison notebook reports each model’s test performance (accuracy, precision, recall, F1, AUC) and highlights the best configuration for each family. The high-level takeaway: neural networks slightly outperform the baselines on AUC, but all three approaches are competitive and sensitive to the feature engineering choices.
