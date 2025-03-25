# BERT-phishing-detection

This project fine-tunes a BERT model (`google-bert/bert-base-uncased`) to classify phishing websites and profiles its energy consumption and carbon emissions using CodeCarbon. The goal is to demonstrate sustainable machine learning by optimizing efficiency (e.g., freezing base layers) while maintaining strong performance.

## Project Overview
- **Dataset**: `shawhin/phishing-site-classification` from Hugging Face.
- **Model**: BERT-base with frozen base layers, fine-tuning only the pooler and classifier.
- **Profiling**: Tracks energy usage and CO2 emissions with CodeCarbon.
- **Metrics**: Achieves ~89.3% accuracy and ~0.945 AUC on the validation set.
- **Emissions**: ~0.0025 kgCO2eq per training run (varies by hardware).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bert-phishing-classifier.git
   cd bert-phishing-classifier
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
  Note: A GPU is recommended for training (e.g., NVIDIA with CUDA support).      

## Usage
  Run the profiling script to train the madel and track the emissions.
  ```bash
    python profile.py
```
## Output
- Emissions data saved to emissions.csv
- Validation metrics printed to the console (e.g., {'Accuracy': 0.893, 'AUC': 0.945}).
- Model checkpoints saved in bert-phishing-classifier_teacher/.
Explore the development process, including detailed code and results, in phishing_classifier.ipynb.

## Files
- **profile.py:** Standalone script for training the BERT model and profiling emissions.
- **phishing_classifier.ipynb:** Jupyter notebook documenting the development process and results.
- **requirements.txt:** List of Python dependencies required to run the project.
- **README.md:** This file.

## Requirements
See requirements.txt for the full list of dependencies. Key libraries include:
- transformers==4.49.0
- codecarbon==2.8.3
- datasets
- numpy
- pandas

## Notes
- Emissions results may vary depending on your hardware (e.g., CPU vs. GPU) and runtime environment (e.g., Google Colab with a Tesla T4 GPU).
- This project aligns with green software principles by reducing computational overhead through techniques like freezing BERT’s base layers.

## Author
Yashasvi Acharya Bhatter (bhatteryash21@gmail.com)
