# What Determines Competency in Company: Deciphering HRD Features using AI-Powered Model

This repository contains the code and datasets associated with the paper **"What Determines Competency in Company: Deciphering HRD Features using AI-Powered Model."** The purpose of this work is to explore and classify competency levels within companies by leveraging AI-driven models trained on HRD (Human Resource Development) features.

## Repository Structure

- `data/`: Contains the datasets used for training and testing the models.
- `models/`: Includes the trained models and the scripts for training.
- `notebooks/`: Jupyter notebooks for exploratory data analysis, feature extraction, and model evaluation.
- `src/`: Source code for data preprocessing, model training, and evaluation.
- `results/`: Output files, including model performance metrics and visualizations.

## Installation

To replicate the results of the paper, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/Competency-HRD-Features-AI.git
cd Competency-HRD-Features-AI
pip install -r requirements.txt
```

## To train the model using the provided dataset, run the following command:
python src/train_model.py --config configs/train_config.yaml

## Evaluating the Model
After training, you can evaluate the model on the test dataset:
python src/evaluate_model.py --model models/trained_model.pth --test_data data/test_dataset.csv

## Reproducing Results
To reproduce the results discussed in Section 4.2 of the paper, execute the corresponding Jupyter notebook:
jupyter notebook notebooks/evaluate_results.ipynb

## Results
The results of the model's performance, including accuracy, precision, recall, and F1-score, are available in the results/ directory. Comparative analysis and detailed discussions can be found in the paper.

## This paper is currently under review in 

## License
This project is licensed under the MIT License - see the LICENSE file for details.
