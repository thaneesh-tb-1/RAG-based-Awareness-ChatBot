# Run DIA Chatbot Docker Container
# This script runs the container with proper volume mounts for gcloud credentials

$IMAGE_NAME = "thaneeshtb/dia-agent:latest"
$GCLOUD_CREDENTIALS = "$env:APPDATA\gcloud\application_default_credentials.json"
$GCLOUD_CONFIG_DIR = "$env:APPDATA\gcloud"

Write-Host "Starting DIA Chatbot container..." -ForegroundColor Green
Write-Host "Image: $IMAGE_NAME" -ForegroundColor Yellow
Write-Host ""

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "Warning: .env file not found!" -ForegroundColor Yellow
    Write-Host "The container may not work without environment variables." -ForegroundColor Yellow
    Write-Host ""
}

# Run the container with gcloud credentials mounted
docker run --rm `
  -p 8080:8080 `
  --env-file .env `
  -v "${GCLOUD_CONFIG_DIR}:/root/.config/gcloud:ro" `
  $IMAGE_NAME




