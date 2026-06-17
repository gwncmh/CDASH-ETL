#!/bin/bash
# setup.sh — Chạy một lần để setup môi trường CDASH
# Usage: bash setup.sh
#
# Script này sẽ:
#   1. Kiểm tra prerequisites (docker, gcloud)
#   2. Tạo thư mục cần thiết
#   3. Tạo GCP Service Account và cấp quyền
#   4. Đóng gói PySpark scripts thành .zip rồi upload lên GCS
#   5. Khởi động Airflow (Docker Compose)
#
# Trước khi chạy, hãy sửa config/config.py với đúng giá trị:
#   GCP_PROJECT_ID, GCS_BUCKET, NOAA_API_TOKEN

set -e

echo "========================================"
echo " CDASH — Crime Analytics Setup Script"
echo "========================================"

# ── 0. Đọc PROJECT_ID và BUCKET từ config.py ──────────────────────────────
# Tránh hỏi lại nếu đã cấu hình trong file config
PROJECT_ID=$(python3 -c "
import sys; sys.path.insert(0,'config')
from config import GCP_PROJECT_ID; print(GCP_PROJECT_ID)
" 2>/dev/null || echo "")

BUCKET=$(python3 -c "
import sys; sys.path.insert(0,'config')
from config import GCS_BUCKET; print(GCS_BUCKET)
" 2>/dev/null || echo "")

if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "your-project-id" ]; then
    read -p "GCP Project ID chưa cấu hình trong config.py. Nhập vào đây: " PROJECT_ID
fi
if [ -z "$BUCKET" ] || [ "$BUCKET" = "your-bucket-name" ]; then
    read -p "GCS Bucket chưa cấu hình trong config.py. Nhập vào đây: " BUCKET
fi

echo ""
echo "  Project : $PROJECT_ID"
echo "  Bucket  : $BUCKET"
echo ""

# ── 1. Kiểm tra prerequisites ──────────────────────────────────────────────
echo "[1/5] Checking prerequisites..."
command -v docker  >/dev/null || { echo "❌ Docker not found. Install: https://docs.docker.com/get-docker/"; exit 1; }
command -v gcloud  >/dev/null || { echo "❌ gcloud not found. Install: https://cloud.google.com/sdk/docs/install"; exit 1; }
command -v gsutil  >/dev/null || { echo "❌ gsutil not found (thường đi kèm gcloud SDK)."; exit 1; }
command -v zip     >/dev/null || { echo "❌ zip not found. Cài bằng: sudo apt install zip"; exit 1; }
echo "✓ docker, gcloud, gsutil, zip found"

# ── 2. Tạo thư mục cần thiết ──────────────────────────────────────────────
echo ""
echo "[2/5] Creating directories..."
mkdir -p secrets logs
echo "✓ Directories created"

# ── 3. Tạo GCP Service Account ────────────────────────────────────────────
echo ""
echo "[3/5] Setting up GCP Service Account..."

SA_NAME="airflow-crime-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create "$SA_NAME" \
    --display-name="Airflow Crime Analytics SA" \
    --project="$PROJECT_ID" 2>/dev/null || echo "  (Service Account đã tồn tại, bỏ qua)"

echo "  Granting IAM roles..."
for ROLE in \
    "roles/dataproc.editor" \
    "roles/bigquery.dataEditor" \
    "roles/bigquery.jobUser" \
    "roles/storage.objectAdmin"; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="$ROLE" \
        --quiet
    echo "  ✓ $ROLE"
done

KEY_FILE="secrets/gcp-sa-key.json"
if [ ! -f "$KEY_FILE" ]; then
    gcloud iam service-accounts keys create "$KEY_FILE" \
        --iam-account="$SA_EMAIL" \
        --project="$PROJECT_ID"
    echo "✓ Service account key saved → $KEY_FILE"
else
    echo "  ($KEY_FILE đã tồn tại, bỏ qua tạo key mới)"
fi

# ── 4. Đóng gói và upload PySpark scripts lên GCS ─────────────────────────
echo ""
echo "[4/5] Packaging and uploading PySpark scripts to GCS..."

GCS_SCRIPTS="gs://${BUCKET}/scripts"

# 4a. Đóng gói từng package thành .zip (Dataproc cần import theo cách này)
echo "  Packaging stage3_silver..."
cd scripts/etl
zip -r ../../stage3_silver.zip stage3_silver/src/ -x "**/__pycache__/*" "**/*.pyc"

echo "  Packaging stage4_gold..."
zip -r ../../stage4_gold.zip stage4_gold/src/ -x "**/__pycache__/*" "**/*.pyc"

echo "  Packaging utils..."
zip -r ../../utils.zip utils/ -x "**/__pycache__/*" "**/*.pyc"
cd ../..

# 4b. Upload các main script (entry-point của từng PySpark job)
gsutil cp scripts/etl/stage2_bronze/data_ingestion.py \
           "${GCS_SCRIPTS}/stage2_bronze/data_ingestion.py"
echo "  ✓ stage2_bronze/data_ingestion.py"

gsutil cp scripts/etl/stage3_silver/main.py \
           "${GCS_SCRIPTS}/stage3_silver_main.py"
echo "  ✓ stage3_silver_main.py"

gsutil cp scripts/etl/stage4_gold/main.py \
           "${GCS_SCRIPTS}/stage4_gold_main.py"
echo "  ✓ stage4_gold_main.py"

# 4c. Upload các .zip package
gsutil cp stage3_silver.zip stage4_gold.zip utils.zip "${GCS_SCRIPTS}/"
echo "  ✓ stage3_silver.zip, stage4_gold.zip, utils.zip"

# 4d. Upload init script cho Dataproc (cài pip packages trên worker nodes)
if [ -f "scripts/infra/init_pip.sh" ]; then
    gsutil cp scripts/infra/init_pip.sh "${GCS_SCRIPTS}/init_pip.sh"
    echo "  ✓ init_pip.sh"
else
    echo "  ⚠ scripts/infra/init_pip.sh không tìm thấy."
    echo "    Tạo file này để Dataproc cluster cài đúng dependencies (h3, lightgbm...)."
    echo "    Xem mẫu bên dưới:"
    echo ""
    echo "    ---- scripts/infra/init_pip.sh ----"
    echo "    #!/bin/bash"
    echo "    pip install h3==4.1.0 lightgbm==4.6.0 scikit-learn==1.4.0"
    echo "    ------------------------------------"
fi

# 4e. Dọn .zip tạm trong thư mục gốc
rm -f stage3_silver.zip stage4_gold.zip utils.zip
echo "  ✓ Cleaned up local .zip files"

echo ""
echo "✓ Tất cả scripts đã upload lên ${GCS_SCRIPTS}/"

# ── 5. Khởi động Airflow ───────────────────────────────────────────────────
echo ""
echo "[5/5] Starting Airflow with Docker Compose..."
docker compose up airflow-init
docker compose up -d

echo ""
echo "========================================"
echo " ✅ Setup hoàn tất!"
echo "========================================"
echo ""
echo " Airflow UI : http://localhost:8080"
echo " Username   : admin"
echo " Password   : admin"
echo " Dashboard  : http://localhost:3000"
echo ""
echo " Bước tiếp theo:"
echo "  1. Mở Airflow UI → Admin → Connections"
echo "  2. Tạo connection: google_cloud_default"
echo "     - Conn Type   : Google Cloud"
echo "     - Keyfile Path: /opt/airflow/secrets/gcp-sa-key.json"
echo ""
echo "  3. Trigger Full Load lần đầu:"
echo "     - Bật DAG 'crime_analytics_pipeline'"
echo "     - Nhấn Trigger DAG → truyền: {\"full_load\": true}"
echo "     - Quá trình mất khoảng 8–9 giờ"
echo ""
echo "  4. Sau khi pipeline xong, export GeoJSON cho Kepler.gl:"
echo "     curl -X POST http://localhost:5000/export/all"
echo ""
echo " Để trigger thủ công qua CLI:"
echo "  docker exec -it \$(docker ps -qf name=airflow-scheduler) \\"
echo "    airflow dags trigger crime_analytics_pipeline"