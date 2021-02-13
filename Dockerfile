FROM ubuntu

LABEL Author rajesh.r6r@gmail.com

USER root

WORKDIR /app

ADD . /app

RUN set -xe \
    && apt-get update -y --no-install-recommends --yes\
    && apt-get install python3-pip -y \
    && rm -rf /var/lib/apt/lists/* # remove the cached files

RUN pip3 install --upgrade pip

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt


EXPOSE 8000

CMD ["python3","predictionapp.py"]