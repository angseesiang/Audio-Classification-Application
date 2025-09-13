import pytest
import numpy as np
from main import Config, AudioProcessor, ModelFactory, AudioClassifier
import json

@pytest.fixture
def config():
    config_json = """
    {
        "sample_rate": 22050,
        "n_mfcc": 13,
        "test_size": 0.2,
        "random_state": 42,
        "num_classes": 2
    }
    """
    return Config(json.loads(config_json))

@pytest.fixture
def audio_processor(config):
    return AudioProcessor(config)

@pytest.fixture
def model(config):
    return ModelFactory.create_model('svm', config.num_classes)

@pytest.fixture
def classifier(config, model):
    return AudioClassifier(config, model)

def test_feature_extraction(audio_processor):
    features = audio_processor.extract_features('audio_files/example.wav')
    assert features.shape == (13,)

def test_data_preparation(classifier):
    features = [np.random.rand(13) for _ in range(4)]
    labels = ['class1', 'class2', 'class1', 'class2']
    X_train, X_test, y_train, y_test = classifier.prepare_data(features, labels)
    assert len(X_train) == 3
    assert len(X_test) == 1

def test_model_training(classifier):
    features = [np.random.rand(13) for _ in range(4)]
    labels = ['class1', 'class2', 'class1', 'class2']
    X_train, X_test, y_train, y_test = classifier.prepare_data(features, labels)
    classifier.train(X_train, y_train)
    accuracy = classifier.evaluate(X_test, y_test)
    assert accuracy > 0
