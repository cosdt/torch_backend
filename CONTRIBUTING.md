# Table of Contents

- [Unit Testing](#unit-testing)
  - [C++ Unit Testing](#c-unit-testing)
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
