FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed by geospatial and numeric libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gdal-bin \
       libgdal-dev \
       libgeos-dev \
       libproj-dev \
       pkg-config \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Ensure headers are discoverable for some wheels
ENV CPLUS_INCLUDE_PATH=/usr/include
ENV C_INCLUDE_PATH=/usr/include

WORKDIR /app

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy application code
COPY . /app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
