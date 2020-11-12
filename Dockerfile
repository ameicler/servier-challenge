FROM python:3.7

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda
ENV PATH=/root/miniconda/bin:$PATH

RUN apt-get update \
  && apt-get install vim -y

RUN conda update -n base conda \
  && conda create -y --name servier python=3.6 \
  && activate servier \
  && conda install -c conda-forge rdkit

ADD app /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN python setup.py develop

EXPOSE 5000

ENTRYPOINT ["python", "/app/src/main.py"]
