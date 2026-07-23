#!/bin/bash

echo "Installing Python 3.10.8..."

pyenv install -s 3.10.8
pyenv global 3.10.8

python --version

python -m venv .venv

source .venv/bin/activate

pip install --upgrade pip
