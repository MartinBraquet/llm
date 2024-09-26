import os

import pytest


@pytest.fixture(autouse=True)
def change_test_dir(request):
    # Get the directory of the current test file
    test_dir = os.path.dirname(request.fspath)
    # Save the current working directory
    original_dir = os.getcwd()
    # Change to the test file's directory
    os.chdir(test_dir)

    yield

    # After the test finishes, change back to the original directory
    os.chdir(original_dir)
