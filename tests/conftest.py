import pytest
import torch
from kvcache.utils import tiny_model

@pytest.fixture(scope="module")
def pair():
    torch.set_num_threads(2)
    return tiny_model()

@pytest.fixture
def model(pair):
    return pair[0]

@pytest.fixture
def hf(pair):
    return pair[1]

@pytest.fixture
def ids(model):
    torch.manual_seed(4)
    return torch.randint(0, model.spec.vocab_size, (1, 96))

@pytest.fixture
def tol():
    return 1e-4
