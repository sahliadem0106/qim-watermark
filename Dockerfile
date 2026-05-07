# ─────────────────────────────────────────────────────
# Dockerfile — QIM Watermarking Web App
# ─────────────────────────────────────────────────────

# 1. Base image : official Python 3.11 (slim = lightweight Linux)
FROM python:3.11-slim

# 2. System dependencies needed by opencv-python-headless
#    libglib2.0-0 : required by OpenCV even in headless mode
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Working directory inside the container
WORKDIR /app

# 4. Copy requirements FIRST (Docker cache optimization)
#    → if only app.py changes, pip install is NOT re-run
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the project
COPY . .

# 7. Tell matplotlib to use the headless Agg backend
#    (no screen available in Docker / Azure)
ENV MPLBACKEND=Agg

# 8. Expose the port the app listens on
EXPOSE 5000

# 9. Start command — gunicorn = production-grade WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
