# Docker Hub Deployment Script
# Replace YOUR_DOCKERHUB_USERNAME with your actual Docker Hub username

$DOCKERHUB_USERNAME = "thaneeshtb"
$IMAGE_NAME = "dia-agent"
$TAG = "latest"

Write-Host "Step 1: Building Docker image..." -ForegroundColor Green
docker build -t ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG} .

Write-Host "`nStep 2: Logging into Docker Hub..." -ForegroundColor Green
Write-Host "Please enter your Docker Hub credentials when prompted" -ForegroundColor Yellow
docker login

Write-Host "`nStep 3: Pushing image to Docker Hub..." -ForegroundColor Green
docker push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}

Write-Host "`n✅ Deployment complete!" -ForegroundColor Green
Write-Host "Your image is available at: docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}" -ForegroundColor Cyan

