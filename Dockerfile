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

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

COPY src /app/src
COPY setup.py /app/
WORKDIR /app

RUN python setup.py install

EXPOSE 5000

ENTRYPOINT ["python", "/app/src/main.py"]
