import json
from typing import Any, Dict, List, Tuple
import numpy as np
import librosa
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Data class to hold configuration parameters
class Config:
    def __init__(self, config: Dict[str, Any]):
        self.sample_rate: int = config.get('sample_rate', 22050)
        self.n_mfcc: int = config.get('n_mfcc', 13)
        self.test_size: float = config.get('test_size', 0.2)
        self.random_state: int = config.get('random_state', 42)
        self.num_classes: int = config.get('num_classes', 10)

# Audio Processor class for feature extraction
class AudioProcessor:
    def __init__(self, config: Config):
        self.config = config

    def extract_features(self, file_path: str) -> np.ndarray:
        y, sr = librosa.load(file_path, sr=self.config.sample_rate)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.config.n_mfcc)
        return np.mean(mfcc.T, axis=0)

# Model Factory class to create different models
class ModelFactory:
    @staticmethod
    def create_model(model_type: str, num_classes: int) -> Any:
        if model_type == 'svm':
            return SVC()
        elif model_type == 'nn':
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(13,)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model
        else:
            raise ValueError("Unsupported model type")

# Audio Classification class for training and evaluating the model
class AudioClassifier:
    def __init__(self, config: Config, model: Any):
        self.config = config
        self.model = model
        self.label_encoder = LabelEncoder()

    def prepare_data(self, features: List[np.ndarray], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array(features)
        y = self.label_encoder.fit_transform(labels)
        return train_test_split(X, y, test_size=self.config.test_size, random_state=self.config.random_state)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        if isinstance(self.model, SVC):
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        if isinstance(self.model, SVC):
            y_pred = self.model.predict(X_test)
            return accuracy_score(y_test, y_pred)
        else:
            _, accuracy = self.model.evaluate(X_test, y_test)
            return accuracy

# Example usage and test function
def main():
    # Load configuration from a JSON file
    config_json = """
    {
        "sample_rate": 22050,
        "n_mfcc": 13,
        "test_size": 0.2,
        "random_state": 42,
        "num_classes": 10
    }
    """
    config = Config(json.loads(config_json))

    # Initialize audio processor and model
    audio_processor = AudioProcessor(config)
    model = ModelFactory.create_model('nn', config.num_classes)

    # Example data
    files = ['audio_files/audio_file_1.wav', 'audio_files/audio_file_2.wav', 'audio_files/example.wav']
    labels = ['class1', 'class2', 'class3']

    features = [audio_processor.extract_features(file) for file in files]
    classifier = AudioClassifier(config, model)

    X_train, X_test, y_train, y_test = classifier.prepare_data(features, labels)
    classifier.train(X_train, y_train)
    accuracy = classifier.evaluate(X_test, y_test)
    
    print(f"Model accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
