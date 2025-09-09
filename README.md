Question Pair Similarity Detection

This project aims to detect whether two given questions are semantically the same or not. It is based on Natural Language Processing (NLP) techniques and Machine Learning models, trained on the Quora Question Pairs dataset
.

🚀 Features

Preprocessing of text (cleaning, removing stopwords, lemmatization)

Feature extraction using TF-IDF / Word2Vec / Embeddings

Machine learning model to classify duplicate vs. non-duplicate questions

Evaluation with accuracy, precision, recall, and F1-score

Deployment ready with Streamlit

📂 Project Structure
question_pair/
│── data/                  # Dataset (CSV files)
│── notebooks/             # Jupyter Notebooks
│── models/                # Trained model files (.pkl)
│── app.py                 # Streamlit deployment file
│── requirements.txt       # Dependencies
│── README.md              # Project documentation

⚙️ Installation

Clone this repository and install dependencies:

git clone https://github.com/yourusername/question_pair.git
cd question_pair
pip install -r requirements.txt

📝 Usage

Run Jupyter Notebook (for training and experiments):

jupyter notebook question_pair.ipynb


Run Streamlit App (for deployment):

streamlit run app.py


Input two questions, and the model will predict whether they are duplicates or not.

📊 Dataset

Source: Quora Question Pairs Dataset (Kaggle)

Size: 400,000+ question pairs

Columns:

qid1, qid2: Question IDs

question1, question2: The text of the two questions

is_duplicate: Target label (1 = duplicate, 0 = not duplicate)

🔧 Methodology

Text Preprocessing:

Lowercasing, punctuation removal

Tokenization and stopword removal

Lemmatization using NLTK/Spacy

Feature Engineering:

TF-IDF vectors

Cosine similarity

Word embeddings (optional)

Model Training:

Logistic Regression / Random Forest / XGBoost

Hyperparameter tuning for best performance

Evaluation Metrics:

Accuracy

Precision, Recall, F1-score

ROC-AUC

📈 Results

Best model achieved: XX% Accuracy and YY% F1-Score (update with your results).

🌐 Deployment

The project can be deployed using Streamlit.

Hosted App Example: Streamlit Cloud

📌 Future Work

Integrate BERT / Transformer models for better accuracy

Enhance feature engineering with semantic similarity measures

Build a full-fledged API with FastAPI/Flask

🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

📜 License

This project is licensed under the MIT License.
