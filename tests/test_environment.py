
def test_python_environment():
    assert True


def test_numpy_import():
    import numpy as np

    values = np.array([1, 2, 3])
    assert values.sum() == 6
