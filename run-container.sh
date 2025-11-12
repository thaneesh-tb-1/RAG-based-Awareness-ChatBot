#!/bin/bash
# Run DIA Chatbot Docker Container
# This script runs the container with proper volume mounts for gcloud credentials

IMAGE_NAME="thaneeshtb/dia-agent:latest"
GCLOUD_CONFIG_DIR="$HOME/.config/gcloud"

echo "Starting DIA Chatbot container..."
echo "Image: $IMAGE_NAME"
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found!"
    echo "The container may not work without environment variables."
    echo ""
fi

# Run the container with gcloud credentials mounted
docker run --rm \
  -p 8080:8080 \
  --env-file .env \
  -v "${GCLOUD_CONFIG_DIR}:/root/.config/gcloud:ro" \
  $IMAGE_NAME




