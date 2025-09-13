# 🎶 Audio Classification Application

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Toolkit-green)](#)

This repository contains my training exercise on building an **Audio
Classification Application** using **Librosa**, **TensorFlow**, and
**scikit-learn**.\
The project demonstrates how to extract audio features (MFCCs), train
machine learning / deep learning models, and evaluate classification
accuracy.

🔗 GitHub Repo:
[Audio-Classification-Application](https://github.com/angseesiang/Audio-Classification-Application)

------------------------------------------------------------------------

## 📖 Contents

-   `main.py` -- Core implementation:
    -   **Config** class: project configuration (sample rate, MFCC
        count, test split, etc.)
    -   **AudioProcessor**: extracts MFCC features from `.wav` audio
        files
    -   **ModelFactory**: creates models (`svm` or neural network)
    -   **AudioClassifier**: prepares data, trains the model, evaluates
        accuracy
-   `test_audio_classifier.py` -- Unit tests with **pytest** covering:
    -   Feature extraction
    -   Data preparation
    -   Model training and evaluation
-   `audio_files/` -- Example audio files (`audio_file_1.wav`,
    `audio_file_2.wav`, `example.wav`)
-   `requirements.txt` -- Required dependencies
-   `url.txt` -- Reference to the GitHub repository

------------------------------------------------------------------------

## 🚀 How to Use

### 1. Clone this repository

``` bash
git clone https://github.com/your-username/audio-classification-application.git
cd audio-classification-application
```

### 2. Create and activate a virtual environment (recommended)

It is best practice to isolate project dependencies in a virtual
environment.

``` bash
python -m venv venv
source venv/bin/activate   # On Linux / macOS
venv\Scripts\activate    # On Windows
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run the application

``` bash
python src/main.py
```

This will: - Load the audio files - Extract MFCC features - Train a
classifier (SVM or NN) - Print model accuracy

### 5. Run tests

``` bash
pytest tests/test_audio_classifier.py
```

------------------------------------------------------------------------

## 🛠️ Requirements

See [`requirements.txt`](requirements.txt): - `numpy` - `librosa` -
`tensorflow` - `scikit-learn` - `pytest`

------------------------------------------------------------------------

## 📌 Notes

-   This project was created during my **AI/ML training** to gain
    practical experience in **audio data processing** and
    **classification**.
-   It provides both **classical ML (SVM)** and **deep learning (NN)**
    approaches for comparison.

------------------------------------------------------------------------

## 📜 License

This repository is shared for **educational purposes**. Please credit if
you use it in your work.
