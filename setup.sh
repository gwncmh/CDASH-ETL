#!/bin/bash
# setup.sh — Chạy một lần để setup môi trường
# Usage: bash setup.sh

set -e

echo "========================================"
echo " Crime Analytics — Airflow Setup Script"
echo "========================================"

# ── 1. Kiểm tra prerequisites ─────────────────────────────────────────────────
echo ""
echo "[1/6] Checking prerequisites..."
command -v docker  >/dev/null || { echo "❌ Docker not found. Install: https://docs.docker.com/get-docker/"; exit 1; }
command -v gcloud  >/dev/null || { echo "❌ gcloud not found. Install: https://cloud.google.com/sdk/docs/install"; exit 1; }
echo "✓ docker, gcloud found"

# ── 2. Tạo thư mục cần thiết ─────────────────────────────────────────────────
echo ""
echo "[2/6] Creating directories..."
mkdir -p secrets logs dags scripts config
echo "✓ Directories created"

# ── 3. Tạo GCP Service Account ───────────────────────────────────────────────
echo ""
echo "[3/6] Setting up GCP Service Account..."
read -p "Enter your GCP Project ID: " PROJECT_ID

SA_NAME="airflow-crime-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Tạo service account
gcloud iam service-accounts create $SA_NAME \
    --display-name="Airflow Crime Analytics SA" \
    --project=$PROJECT_ID 2>/dev/null || echo "(SA already exists)"

# Cấp quyền cần thiết
echo "Granting IAM roles..."
for ROLE in \
    "roles/dataproc.editor" \
    "roles/bigquery.dataEditor" \
    "roles/bigquery.jobUser" \
    "roles/storage.objectAdmin"; do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="$ROLE" \
        --quiet
    echo "  ✓ $ROLE"
done

# Download key
KEY_FILE="secrets/gcp-sa-key.json"
gcloud iam service-accounts keys create $KEY_FILE \
    --iam-account=$SA_EMAIL \
    --project=$PROJECT_ID
echo "✓ Service account key saved to $KEY_FILE"

# ── 4. Upload PySpark scripts lên GCS ────────────────────────────────────────
echo ""
echo "[4/6] Uploading PySpark scripts to GCS..."
read -p "Enter your GCS bucket name (without gs://): " BUCKET

gsutil cp scripts/etl_clean_crimes.py  gs://${BUCKET}/scripts/
gsutil cp scripts/etl_gis_features.py  gs://${BUCKET}/scripts/
echo "✓ Scripts uploaded to gs://${BUCKET}/scripts/"

# ── 5. Update config.py ───────────────────────────────────────────────────────
echo ""
echo "[5/6] Updating config.py..."
read -p "Enter GCP region (e.g. us-central1): " REGION
read -p "Enter BigQuery dataset name: " BQ_DATASET

sed -i "s/YOUR_PROJECT_ID/$PROJECT_ID/g"    config/config.py
sed -i "s/YOUR_REGION/$REGION/g"            config/config.py
sed -i "s/YOUR_ZONE/${REGION}-a/g"          config/config.py
sed -i "s/YOUR_BUCKET_NAME/$BUCKET/g"       config/config.py
sed -i "s/YOUR_BQ_DATASET/$BQ_DATASET/g"    config/config.py
echo "✓ config/config.py updated"

# ── 6. Khởi động Airflow ──────────────────────────────────────────────────────
echo ""
echo "[6/6] Starting Airflow with Docker Compose..."
docker compose up airflow-init
docker compose up -d airflow-webserver airflow-scheduler

echo ""
echo "========================================"
echo " ✅ Setup complete!"
echo "========================================"
echo ""
echo " Airflow UI:  http://localhost:8080"
echo " Username:    admin"
echo " Password:    admin"
echo ""
echo " Bước tiếp theo:"
echo "  1. Mở Airflow UI → Admin → Connections"
echo "  2. Tạo connection: google_cloud_default"
echo "     - Conn Type: Google Cloud"
echo "     - Keyfile Path: /opt/airflow/secrets/gcp-sa-key.json"
echo "  3. Mở DAG 'crime_analytics_pipeline' và bật ON"
echo ""
echo " Để trigger thủ công:"
echo "  docker exec -it \$(docker ps -qf name=airflow-scheduler) \\"
echo "    airflow dags trigger crime_analytics_pipeline"
