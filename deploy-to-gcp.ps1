# GCP Cloud Run Deployment Script
# This script pushes your Docker image to GCP Artifact Registry and provides the image URL for Cloud Run

$PROJECT_ID = "tech-bharath"
$REGION = "asia-south1"
$REPOSITORY = "dia-chatbot"
$IMAGE_NAME = "dia-agent"
$TAG = "latest"

$ARTIFACT_REGISTRY_URL = "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}"

Write-Host "`n=== GCP Cloud Run Deployment ===" -ForegroundColor Cyan
Write-Host "Project: $PROJECT_ID" -ForegroundColor Yellow
Write-Host "Region: $REGION" -ForegroundColor Yellow
Write-Host "Repository: $REPOSITORY" -ForegroundColor Yellow
Write-Host ""

# Step 1: Authenticate with GCP
Write-Host "Step 1: Checking GCP authentication..." -ForegroundColor Green
$gcloudCheck = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>&1
if ($LASTEXITCODE -ne 0 -or !$gcloudCheck) {
    Write-Host "Please authenticate with GCP first:" -ForegroundColor Yellow
    Write-Host "  gcloud auth login" -ForegroundColor White
    Write-Host "  gcloud config set project $PROJECT_ID" -ForegroundColor White
    exit 1
}
Write-Host "✓ Authenticated" -ForegroundColor Green

# Step 2: Set project
Write-Host "`nStep 2: Setting GCP project..." -ForegroundColor Green
gcloud config set project $PROJECT_ID
Write-Host "✓ Project set to $PROJECT_ID" -ForegroundColor Green

# Step 3: Enable required APIs
Write-Host "`nStep 3: Enabling required GCP APIs..." -ForegroundColor Green
gcloud services enable artifactregistry.googleapis.com --quiet
gcloud services enable run.googleapis.com --quiet
gcloud services enable cloudbuild.googleapis.com --quiet
Write-Host "✓ APIs enabled" -ForegroundColor Green

# Step 4: Create Artifact Registry repository (if it doesn't exist)
Write-Host "`nStep 4: Creating Artifact Registry repository..." -ForegroundColor Green
$repoExists = gcloud artifacts repositories describe $REPOSITORY --location=$REGION --format="value(name)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating repository $REPOSITORY in $REGION..." -ForegroundColor Yellow
    gcloud artifacts repositories create $REPOSITORY --repository-format=docker --location=$REGION --description="DIA Chatbot Docker images"
    Write-Host "✓ Repository created" -ForegroundColor Green
} else {
    Write-Host "✓ Repository already exists" -ForegroundColor Green
}

# Step 5: Configure Docker authentication
Write-Host "`nStep 5: Configuring Docker authentication for Artifact Registry..." -ForegroundColor Green
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
Write-Host "✓ Docker authenticated" -ForegroundColor Green

# Step 6: Tag the image for Artifact Registry
Write-Host "`nStep 6: Tagging image for Artifact Registry..." -ForegroundColor Green
docker tag thaneeshtb/${IMAGE_NAME}:${TAG} $ARTIFACT_REGISTRY_URL
Write-Host "✓ Image tagged as $ARTIFACT_REGISTRY_URL" -ForegroundColor Green

# Step 7: Push to Artifact Registry
Write-Host "`nStep 7: Pushing image to Artifact Registry..." -ForegroundColor Green
Write-Host "This may take a few minutes..." -ForegroundColor Yellow
docker push $ARTIFACT_REGISTRY_URL

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ SUCCESS! Image pushed to Artifact Registry" -ForegroundColor Green
    Write-Host "`n=== Cloud Run Image URL ===" -ForegroundColor Cyan
    Write-Host $ARTIFACT_REGISTRY_URL -ForegroundColor White -BackgroundColor DarkGreen
    Write-Host ""
    
    Write-Host "=== Deploy to Cloud Run ===" -ForegroundColor Cyan
    Write-Host "You can now deploy to Cloud Run using:" -ForegroundColor Yellow
    Write-Host ""
    $deployCmd = "gcloud run deploy dia-chatbot --image $ARTIFACT_REGISTRY_URL --region $REGION --platform managed --allow-unauthenticated --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=$REGION,RAG_CORPUS=projects/$PROJECT_ID/locations/$REGION/ragCorpora/288230376151711744 --port 8080"
    Write-Host $deployCmd -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "`n❌ Error pushing image. Please check the error above." -ForegroundColor Red
    exit 1
}

