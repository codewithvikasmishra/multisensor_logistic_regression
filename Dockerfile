FROM python:3.7

MAINTAINER Vikas Kumar Mishra
COPY requirements.txt ./requirements.txt

ENV FLASK_APP=api.py
ENV FLASK_RUN_HOST=0.0.0.0

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

COPY . /logistic_regression/app
WORKDIR /logistic_regression/app

CMD ["flask", "run"]