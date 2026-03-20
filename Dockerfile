FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy BOTH files now
COPY train_model.py .
COPY predict.py .

# First train, then predict
CMD python train_model.py && python predict.py