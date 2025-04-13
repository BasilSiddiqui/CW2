# Essay Score Prediction using Machine Learning

This project predicts essay scores (ranging from 1 to 6) based on linguistic features using supervised machine learning techniques. The models implemented include **Gaussian Naive Bayes** and **Random Forest Classifier**. The evaluation metric used is **Quadratic Weighted Kappa (QWK)** to measure the agreement between predicted and actual scores.

## 📁 Project Structure

```
├── data/                  # Raw and processed data files
├── notebooks/             # Jupyter notebooks for EDA and modeling
├── models/                # Trained model files
├── src/                   # Source code for preprocessing, training, and evaluation
├── README.md              # Project overview
└── requirements.txt       # Python dependencies
```

## 🔍 Features

- **Data Preprocessing:** Feature scaling and transformation for optimal model performance
- **Modeling:** 
  - Gaussian Naive Bayes
  - Random Forest Classifier
- **Evaluation:**
  - Accuracy, Confusion Matrix
  - Quadratic Weighted Kappa (QWK)

## 📊 Dataset

The dataset contains essays scored by humans, with accompanying linguistic features such as:
- Word count
- Average sentence length
- Use of complex words
- Spelling errors
- Readability scores (Flesch, Gunning Fog, etc.)

> Note: Dataset is either publicly available or provided for academic purposes. Data is preprocessed before training.

## 🛠️ How to Run

1. Clone this repository:

```bash
git clone https://github.com/yourusername/essay-score-prediction.git
cd essay-score-prediction
```

2. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the model training script:

```bash
python src/train_model.py
```

4. Evaluate results:

```bash
python src/evaluate_model.py
```

## ✅ Results

| Model                 | Accuracy | QWK Score |
|----------------------|----------|-----------|
| Gaussian Naive Bayes | 62%      | 0.49      |
| Random Forest        | 74%      | 0.67      |

> These results may vary based on random seed and feature selection.

## 📌 Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib / seaborn (for visualization)
- nltk / spacy (for text processing)

## 📚 Future Improvements

- Incorporate deep learning models (e.g., LSTM, BERT)
- Add feature selection and hyperparameter tuning
- Deploy using a web app or API (e.g., Flask or Streamlit)
