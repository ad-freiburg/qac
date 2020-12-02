FROM ubuntu:18.04
LABEL maintainer="prange@informatik.uni-freiburg.de"
ENV PYTHONIOENCODING=UTF-8

RUN apt-get update && apt-get install -y python3-pip
RUN python3 -m pip install --upgrade pip

# Install python packages
COPY requirements.txt /home/requirements.txt
RUN pip3 install -r /home/requirements.txt
RUN python3 -W ignore -m nltk.downloader stopwords
RUN python3 -W ignore -m nltk.downloader wordnet

COPY *.py /home/

# Adjust paths for python scripts
COPY docker_paths.py /home/global_paths.py

CMD ["python3", "/home/qac_api.py", "80"]

# docker build -t qac .
# docker run --restart unless-stopped -it --detach -p 8181:80 -v /nfs/students/natalie-prange:/data qac
