python -m venv ./venv
./venv/scripts/activate.ps1
# for llama2
$env:FORCE_CMAKE=1
$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"
pip install llama-cpp-python --no-cache-dir --verbose
pip install llama-cpp-python 
python.exe -m pip install --upgrade pip
pip install openai==0.28
pip install setuptools
pip install llama-cpp-python
pip install llama-index
pip install --upgrade llama-index-core
pip install transformers
pip install langchain
pip install torch

# image description models
pip install transformers
pip install Pillow
pip install torch
pip install requests
pip install jupyterlab
pip install ipywidgets
pip install flask
pip install Flask-Cors

# for score evaluation
pip install rouge-score nltk sacrebleu
python -m nltk.downloader punkt
python -m nltk.downloader wordnet
pip install openpyxl
