FROM python:3.14-rc-slim

COPY docker_requirements.txt /tmp/
RUN pip install --no-cache-dir --default-timeout=400 -r /tmp/docker_requirements.txt && rm /tmp/docker_requirements.txt