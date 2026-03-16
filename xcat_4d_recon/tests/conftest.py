"""
pytest configuration: skip tests whose optional dependencies are not installed.
"""
import importlib
import pytest


def _has(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


# Markers used in tests
def pytest_configure(config):
    config.addinivalue_line("markers", "requires_torch: skip if PyTorch not installed")
    config.addinivalue_line("markers", "requires_nibabel: skip if nibabel not installed")


def pytest_collection_modifyitems(items):
    for item in items:
        if "test_pca_reduction" in item.nodeid or "test_pod_deeponet" in item.nodeid:
            if "test_pca_reduction" in item.nodeid and not _has("nibabel"):
                item.add_marker(pytest.mark.skip(reason="nibabel not installed"))
            if "test_pod_deeponet" in item.nodeid and not _has("torch"):
                item.add_marker(pytest.mark.skip(reason="torch not installed"))
