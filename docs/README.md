# PyTorch Backend Documents

## Build From local

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
