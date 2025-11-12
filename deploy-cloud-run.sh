#!/bin/bash
# Deploy DIA Chatbot to Cloud Run

gcloud run deploy dia-chatbot \
  --image asia-south1-docker.pkg.dev/tech-bharath/dia-chatbot/dia-agent:latest \
  --region asia-south1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=tech-bharath,GOOGLE_CLOUD_LOCATION=asia-south1,RAG_CORPUS=projects/tech-bharath/locations/asia-south1/ragCorpora/288230376151711744 \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10





