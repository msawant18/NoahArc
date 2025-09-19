FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime
FROM python:3.10-slim

WORKDIR /app

# copy source and requirements
COPY pyproject.toml README.md requirements.txt requirements-dev.txt /app/
COPY src/ /app/src/
COPY src /app/src

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -r requirements-dev.txt || true
# install package
RUN pip install -e .

# copy rest of repo
COPY . /app

CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
