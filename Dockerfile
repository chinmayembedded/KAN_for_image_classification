FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install 
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
