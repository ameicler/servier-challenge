FROM python:3.7

ADD src /app
WORKDIR /app

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda
RUN export PATH=/root/miniconda/bin:$PATH
RUN conda update -n base conda
RUN conda create -y --name servier python=3.6
RUN conda activate servier
RUN conda install -c conda-forge rdkit

ADD src /app
WORKDIR /app
RUN pip install numpy pandas tensorflow matplotlib flask keras requests
ENTRYPOINT ["python", "/app/flask_main.py"]
