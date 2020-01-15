FROM ubuntu:18.04
MAINTAINER Natalie Prange <prangen@informatik.uni-freiburg.de>

RUN apt-get update && apt-get install -y make vim python3-pip

COPY bashrc bashrc
COPY Makefile /home/Makefile
COPY requirements.txt /home/requirements.txt
COPY *.py /home/

# Adjust paths for python scripts
COPY docker_paths.py /home/global_paths.py

# Set the python encoding
ENV PYTHONIOENCODING=ISO-8859-1

# Install python packages
RUN pip3 install -r /home/requirements.txt
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader wordnet

CMD ["/bin/bash", "--rcfile", "bashrc"]

# docker build -t qac .
# docker run -it -p 8181:80 -v /nfs/students/natalie-prange:/extern/data -v /nfs/students/natalie-prange/docker_output:/extern/output qac
