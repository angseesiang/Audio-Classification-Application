import os
import sys
import numpy as np
import pytest

# Ensure src/ is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from main import Config, AudioProcessor, ModelFactory, AudioClassifier


@pytest.fixture
def config():
    return Config({
        "sample_rate": 22050,
        "n_mfcc": 13,
        "test_size": 0.25,
        "random_state": 42,
        "num_classes": 2,
    })


@pytest.fixture
def audio_processor(config):
    return AudioProcessor(config)


@pytest.fixture
def classifier(config):
    model = ModelFactory.create_model("nn", config.num_classes)
    return AudioClassifier(config, model)


def test_feature_extraction(audio_processor, tmp_path):
    # Create a dummy sine wave and save to a wav file
    sr = audio_processor.config.sample_rate
    t = np.linspace(0, 1, sr)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)
    import soundfile as sf
    wav_path = tmp_path / "dummy.wav"
    sf.write(wav_path, y, sr)

    features = audio_processor.extract_features(str(wav_path))
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == audio_processor.config.n_mfcc


def test_model_training(classifier):
    # Perfectly separable features
    features = [
        np.ones(13),       # class1
        np.ones(13) * 2,   # class1
        np.zeros(13),      # class2
        np.zeros(13) + 0.1 # class2
    ]
    labels = ["class1", "class1", "class2", "class2"]

    X_train, X_test, y_train, y_test = classifier.prepare_data(features, labels)
    classifier.train(X_train, y_train)
    accuracy = classifier.evaluate(X_test, y_test)

    # Should perform better than random guessing
    assert accuracy >= 0.5


def test_model_factory(config):
    nn_model = ModelFactory.create_model("nn", config.num_classes)
    assert nn_model is not None

    svm_model = ModelFactory.create_model("svm", config.num_classes)
    assert svm_model is not None

