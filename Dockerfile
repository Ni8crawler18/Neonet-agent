# Neonet Vision Agent — OpenCV 5 agentic vision demo (CPU-only, AWS-deployable)
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY media/ ./media/

EXPOSE 8010
# On AWS: build this image, push to ECR, run on an EC2 t3.large (or Fargate);
# the operator console is served on :8010 behind an ALB.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8010"]
