FROM python:3.10-slim

WORKDIR /app

COPY flask_app/ /app/

COPY model/vectorizer.pkl /app/model/vectorizer.pkl

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5001

# Local
CMD ["python", "app.py"]   

# Production
# CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--timeout", "120", "app:app"]