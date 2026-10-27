# Fake News Detection System

A machine-learning-based application that classifies news articles as **FAKE** or **REAL** using Natural Language Processing (NLP), TF-IDF feature extraction, and a trained Random Forest classifier. The project also provides optional evidence verification to help users investigate claims using relevant online sources.

## Project Overview

Fake news spreads rapidly through websites and social media, making it difficult to distinguish misleading information from reliable reporting. This project demonstrates how machine learning can analyze news text and identify patterns associated with previously labelled fake and real news articles.

The application provides a Streamlit web interface where users can enter news text, view a predicted classification, and optionally search for supporting evidence.

**Important:** The model's prediction is not proof that a claim is true or false. Users should verify important claims using reliable sources.

## Features

- **News Classification:** Predicts whether an article is likely FAKE or REAL.
- **Text Preprocessing:** Prepares news text for machine-learning analysis.
- **TF-IDF Vectorization:** Converts textual data into numerical features.
- **Machine Learning:** Uses a trained Random Forest classifier.
- **Interactive Interface:** Provides a user-friendly Streamlit application.
- **Evidence Verification:** Supports optional web searches and fact-check lookups.
- **Model Persistence:** Saves the trained pipeline for reuse.
- **Evaluation:** Supports model performance assessment using held-out data.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| Pandas | Dataset processing and manipulation |
| scikit-learn | Machine learning and model evaluation |
| TF-IDF | Text feature extraction |
| Random Forest | News classification |
| Streamlit | Web application interface |
| Joblib | Saving and loading the trained model |
| CSV | Dataset storage |
| DDGS | Optional web search |
| Google Fact Check Tools API | Optional fact-check lookup |

## Project Structure

```text
Fake-News-Detection/
├── app/
│   └── app.py
├── data/
│   ├── Fake.csv
│   ├── True.csv
│   └── news.csv
├── models/
│   └── fake_news_pipeline.joblib
├── src/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── preprocess.py
│   ├── predict.py
│   ├── train.py
│   └── verify_claims.py
├── .gitignore
├── requirements.txt
└── README.md
```

*Note: Generated datasets and trained model files may be excluded from GitHub. The exact files present depend on your local setup.*

## Dataset

The project uses two labelled CSV files:

- `Fake.csv` — articles labelled as fake news.
- `True.csv` — articles labelled as real news.

The preparation script combines the source data into a processed dataset for training.

The recorded training run processed **38,379 news articles**.

Dataset labels may contain errors or biases. The dataset may also contain repeated stories or patterns specific to particular publishers.

## How the System Works

1. **Data Collection:** Load the labelled fake and real news datasets.
2. **Data Preprocessing:** Combine the datasets, standardize the text and labels, and handle missing or duplicate records where necessary.
3. **Feature Extraction:** Convert the news text into numerical representations using TF-IDF.
4. **Model Training:** Train and evaluate candidate machine-learning classifiers.
5. **Model Saving:** Save the selected trained pipeline using Joblib.
6. **Prediction:** Accept article text through the Streamlit interface and predict its likely class.
7. **Evidence Verification:** Optionally search for relevant online sources or fact-check results for human review.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AyushThakur534/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Create a Virtual Environment

Using Python 3.11:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, consult the Python documentation or use Command Prompt with:

```cmd
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Ensure that `Fake.csv` and `True.csv` are available in the expected `data/` directory.

Run:

```bash
python -m src.prepare_data
```

### 5. Train the Model

```bash
python -m src.train
```

The training script should generate the trained pipeline at the path configured in the project, expected to be:

```text
models/fake_news_pipeline.joblib
```

### 6. Run the Application

```bash
streamlit run app/app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

*Follow the actual script configuration if your local version uses different paths or commands.*

## Model and Evaluation

The recorded training run selected a Random Forest classifier with TF-IDF features.

| Evaluation Item | Recorded Result |
|---|---|
| Articles processed | 38,379 |
| Selected classifier | Random Forest |
| Reported holdout accuracy | 99.63% |
| Holdout FAKE examples | 1,843 |
| Holdout REAL examples | 46 |

### Important Evaluation Note

The reported test set was highly imbalanced, containing far fewer REAL than FAKE examples. Therefore, the 99.63% accuracy should not be interpreted as proof of reliable real-world fake-news detection.

Further evaluation should include:

- Precision and recall for each class
- Macro F1-score
- Balanced accuracy
- Confusion matrix
- Testing on independent, recent news articles
- Checks for duplicate articles and publisher-related data leakage

## Evidence Verification

The optional evidence-verification component helps users locate related web sources and fact-check results.

Depending on configuration, it may use web search or the Google Fact Check Tools API.

For responsible use:

- Read the original source rather than relying only on a search snippet.
- Check the author's credibility and publication date.
- Confirm that the source addresses the exact claim.
- Compare multiple independent, reliable sources.
- Treat missing search results as inconclusive, not as proof that a claim is false.

## Challenges Faced

- Combining and standardizing labelled datasets.
- Handling missing values and duplicate records where present.
- Preparing text for TF-IDF feature extraction.
- Selecting a suitable machine-learning algorithm.
- Evaluating the model with an imbalanced test set.
- Integrating the trained pipeline with Streamlit.
- Finding relevant evidence while distinguishing search results from actual verification.

## Limitations

- Predictions depend on the quality and representativeness of the training data.
- The model may misclassify satire, unfamiliar topics, paraphrased claims, or breaking news.
- High accuracy on a particular dataset does not guarantee real-world performance.
- Evidence lookup depends on network availability and optional API configuration.
- Automated classification cannot replace independent fact-checking or expert judgment.

## Future Improvements

- Evaluate additional classifiers such as Logistic Regression and Linear SVM.
- Use balanced, source-diverse datasets.
- Improve duplicate detection and leakage prevention.
- Add clearer confidence and uncertainty indicators.
- Improve evidence ranking and claim matching.
- Expand testing to recent and multilingual news.
- Monitor model performance as news topics and writing styles change.

## Responsible Use

This project is intended for educational purposes and as a decision-support tool. It should not be used as the sole basis for labelling people, publishers, or news organizations as dishonest. Always examine the original evidence before making conclusions about a news claim.

## Author

**Ayush Thakur**  
UID: 24BCA10069  
Department: UIC-BCA  
Project: Data Mining — Fake News Detection System

## Contact Deatils

Linkedin:-ayushthakur.in

Email:-git@ayushthakur.in


## References

- [Python Documentation](https://docs.python.org/3/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Google Fact Check Tools API](https://developers.google.com/fact-check/tools/api)

---

*Developed as an academic Data Mining project to explore text classification, machine learning, and evidence-assisted news verification.*
