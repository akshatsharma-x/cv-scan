# CV-Scan: AI-Powered Resume Screener

CV-Scan is an advanced, automated resume screening dashboard built with Python and Streamlit. It leverages Natural Language Processing (NLP) techniques and Machine Learning algorithms to evaluate candidate resumes against a specific Job Description (JD), saving time for recruiters and ensuring an objective screening process.

## How It Works

The system operates by extracting text from candidate resumes and mapping them into mathematical representations (vectors) to measure how closely they match the target job description.

1. Document Parsing: The application accepts resumes in PDF, DOCX, or TXT formats. It extracts the raw text for processing.
2. Text Preprocessing: Using the NLTK library, the text is cleaned by converting it to lowercase, removing punctuation and stop-words, and applying lemmatization to reduce words to their base forms.
3. Feature Extraction and Scoring:
   - Basic Scan (Keyword Matching): Uses Term Frequency-Inverse Document Frequency (TF-IDF) to convert the text into numerical vectors. This measures the presence and frequency of exact keywords shared between the resume and the JD.
   - Advanced Scan (Semantic Matching): Uses a pre-trained BERT language model (specifically, all-MiniLM-L6-v2 via SentenceTransformers) to create deep contextual embeddings. This captures the underlying meaning of the resume, even if exact keywords aren't used. A Hybrid Score is then calculated by combining the TF-IDF score (weighted at 30%) and the BERT semantic score (weighted at 70%).

## Project Structure

- Resume_Screening_Project/: The core directory containing the Streamlit application.
  - src/app.py: The main Streamlit dashboard file.
  - src/train_models.py: Script to generate the pre-trained TF-IDF vectorizer, scaler, and baseline historical candidate datasets.
  - src/utils.py: Helper functions for NLP cleaning and document parsing.
  - models/: Directory containing serialized objects (e.g., .pkl files) required for scoring.
  - requirements.txt: Python dependency list.
- evaluate_classifiers.py: An experimental evaluation script that compares different machine learning classifiers (SVM, DBN, and Hybrid approaches) on the dataset, generating performance metrics (Accuracy, F1-Score, MCC, etc.).
- dataset/: Directory containing the training and historical resume data.
- NLP_Resume_Screening_Report: Available in Markdown, PDF, and DOCX formats, detailing the theoretical backing, algorithms, and experimental results of the project.

## Installation and Usage

1. Create and activate a virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate

2. Install the required dependencies:
   pip install -r Resume_Screening_Project/requirements.txt
   pip install torchvision

3. Generate the required machine learning artifacts (if not already present):
   cd Resume_Screening_Project
   python src/train_models.py

4. Launch the application:
   streamlit run src/app.py

5. Access the Dashboard: Open your web browser and navigate to http://localhost:8501.

## Dashboard Features

- Explainable AI: View the exact TF-IDF keywords driving a candidate's score.
- Semantic Space Visualization: Interactive 2D and 3D PCA scatter plots mapping the candidate's embedding against the job description and historical applicant pool.
- Competitive Ranking: A violin plot distribution showing where the current candidate ranks relative to other candidates in the dataset.
- Experimental Evaluation: A tab displaying confusion matrix metrics and bar charts comparing the performance of standard models against hybrid approaches.
