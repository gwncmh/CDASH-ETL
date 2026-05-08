#!/bin/bash
# cloud_scheduler_setup.sh
#
# Dùng Cloud Scheduler để trigger Airflow DAG hàng tuần
# qua REST API — thay thế cho Cloud Composer
#
# Yêu cầu: Airflow chạy trên VM có public IP hoặc dùng ngrok

set -e

PROJECT_ID=sage-mind-489618-n5
AIRFLOW_URL=$2  # vd: http://YOUR_VM_IP:8080

if [ -z "$PROJECT_ID" ] || [ -z "$AIRFLOW_URL" ]; then
    echo "Usage: bash cloud_scheduler_setup.sh PROJECT_ID AIRFLOW_URL"
    echo "Example: bash cloud_scheduler_setup.sh my-project http://34.x.x.x:8080"
    exit 1
fi

DAG_ID="crime_analytics_pipeline"
JOB_NAME="trigger-crime-pipeline-weekly"

echo "[Scheduler] Creating Cloud Scheduler job: $JOB_NAME"

# Xóa job cũ nếu có
gcloud scheduler jobs delete $JOB_NAME \
    --project=$PROJECT_ID \
    --location=us-central1 \
    --quiet 2>/dev/null || true

# Tạo job mới — chạy Chủ Nhật 2:00 SA (GMT+7)
# Cron: 0 2 * * 0  = 2:00 SA mỗi CN (UTC+0)
# Nếu bạn ở VN (UTC+7), dùng: 0 19 * * 6 (19:00 Thứ 7 UTC = 2:00 SA CN UTC+7)
gcloud scheduler jobs create http $JOB_NAME \
    --project=$PROJECT_ID \
    --location=us-central1 \
    --schedule="0 19 * * 6" \
    --time-zone="UTC" \
    --uri="${AIRFLOW_URL}/api/v1/dags/${DAG_ID}/dagRuns" \
    --message-body='{"conf": {"triggered_by": "cloud_scheduler"}}' \
    --headers="Content-Type=application/json,Authorization=Basic YWRtaW46YWRtaW4=" \
    --http-method=POST \
    --attempt-deadline=5m

# Note: YWRtaW46YWRtaW4= là base64 của "admin:admin"
# Đổi nếu bạn dùng password khác:
# echo -n "username:password" | base64

echo "✓ Cloud Scheduler job created"
echo ""
echo "Job sẽ trigger DAG: $DAG_ID"
echo "Lịch: Mỗi Chủ Nhật 2:00 SA (giờ Việt Nam)"
echo ""
echo "Test ngay:"
echo "  gcloud scheduler jobs run $JOB_NAME --location=us-central1 --project=$PROJECT_ID"
