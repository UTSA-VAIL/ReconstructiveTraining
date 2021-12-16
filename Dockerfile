FROM tensorflow/tensorflow:latest-gpu

# PIP
# RUN python -m pip install --upgrade pip
COPY ./requirements.txt /tmp
RUN python -m pip install -r /tmp/requirements.txt

# APP
WORKDIR /app
COPY *.py /app/
COPY config /app/config/
COPY util /app/util/

# Suppress INFO
# ENV TF_CPP_MIN_LOG_LEVEL=1
# Suppress Warning
# ENV TF_CPP_MIN_LOG_LEVEL=2