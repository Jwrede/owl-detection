# Nutze ein stabiles Python Image (3.10 ist aktuell gut für Librosa)
FROM python:3.10-slim

# Arbeitsverzeichnis im Container
WORKDIR /app

# 1. System-Abhängigkeiten installieren
# librosa & soundfile brauchen 'libsndfile1' und 'ffmpeg'
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Requirements kopieren und installieren
# Wir machen das VOR dem Code-Copy, damit Docker das cachen kann
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Den Code kopieren (aber OHNE die Ordner aus .dockerignore)
COPY . .

# 5. Port freigeben (Standard Streamlit)
EXPOSE 8501

# 6. Healthcheck für Coolify (damit der Server weiß, ob die App läuft)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 7. Startbefehl: Streamlit App starten
ENTRYPOINT ["streamlit", "run", "visualize_results.py", "--server.port=8501", "--server.address=0.0.0.0"]
