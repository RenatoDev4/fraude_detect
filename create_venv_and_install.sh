#!/bin/bash

# Cria o ambiente virtual
python3 -m venv venv

# Ativa o ambiente virtual
source venv/bin/activate

# Instala as bibliotecas a partir do requirements.txt
pip install -r requirements.txt

# Faz upgrade do PIP
pip install --upgrade pip

# Desativa o ambiente virtual
deactivate