# Table of Contents

- [Unit Testing](#unit-testing)
  - [C++ Unit Testing](#c-unit-testing)
  - [Python Unit Testing](#python-unit-testing)
- [Documents](#documents)
  - [Building documentation](#building-documentation)

## Unit Testing
### C++ Unit Testing

**Build Unit Test**

Set Environment Variable `BUILD_TEST=ON` before compiling source code, the executable will stored in path `./build/gtest`.

```bash
export BUILD_TEST=ON
python setup.py develop

# or
BUILD_TEST=ON python setup.py develop
```

**Run Unit Test**

Tests using the GTest framework, you can run tests using following command after compiled.

```bash
cd build && ctest --output-on-failure --verbose
```

### Python Unit Testing

**Run Unit Test**

Execute following command to install requirements before running tests.

```bash
pip install -r test/requirements.txt
```

Based on [pytest.mark](https://docs.pytest.org/en/stable/how-to/mark.html) ability, you can choose running all test cases or ignore some of them.
Using following command to run python tests:

```bash
# Run all tests
pytest test/*

# Ignore npu relative tests
pytest -m 'not npu' test/*

# Only run npu tests
pytest -m 'npu' test/*
```

Labels of test case are defined in `pytest.ini` file in the root of project, you can add your own label to mark tests of your device.


## Documents
### Building documentation

Execute following command locally, if everything works fine, you will find the main page `index.html` in `build/html` folder.

```bash
cd docs

# Install dependencies
# brew install doxygen on mac
sudo apt-get install doxygen
pip install -r requirements.txt

# Build docs
TZ=UTC make clean -C cpp
TZ=UTC make html -C cpp
TZ=UTC make clean
TZ=UTC make html

# Since cpp docs and main docs build separately,
# we need combind them together
cp -R cpp/build/html build/html/cpp_html
```
