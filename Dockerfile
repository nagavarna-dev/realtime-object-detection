# Headless container image for the detection pipeline.
# Run e.g.:  docker run --rm -v "$PWD/outputs:/app/outputs" \
#              detection python main.py --source clip.mp4 --no-display --max-frames 300
FROM python:3.11-slim

# OpenCV runtime needs these system libs even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Headless by default; override the command to point at a real source.
CMD ["python", "main.py", "--no-display"]
