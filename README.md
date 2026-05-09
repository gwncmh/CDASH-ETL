# 🔍 CDASH – Crime Data Analytics & Spatial Hotspot Detection System

**Trường Đại học Công nghệ – Đại học Quốc gia Hà Nội**  
Giảng viên hướng dẫn: PGS.TS. Nguyễn Ngọc Hóa

| Thành viên | MSSV |
|---|---|
| Trần Bình Dương | 23021514 |
| Nguyễn Công Mạnh Hùng | 23021567 |
| Nguyễn Mai Thanh Thư | 23021731 |

---

## 📋 Mục lục

1. [Tổng quan](#-tổng-quan)
2. [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
3. [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
4. [Cấu trúc dữ liệu](#-cấu-trúc-dữ-liệu)
5. [Cài đặt](#-cài-đặt)
6. [Vận hành](#-vận-hành)
7. [Xử lý lỗi](#-xử-lý-lỗi)

---

## 🌆 1. Tổng quan

CDASH là một hệ thống **Data Lakehouse** triển khai trên đám mây, giải quyết bài toán phân tích dữ liệu tội phạm đô thị theo hướng chủ động thay vì phản ứng thụ động. Hệ thống xử lý dữ liệu tội phạm thành phố Chicago từ năm 2001 đến hiện tại (hàng chục triệu bản ghi), kết hợp với dữ liệu thời tiết và kinh tế - xã hội để dự báo điểm nóng tội phạm theo thời gian thực.

### Bài toán giải quyết

- **Data Pipeline tự động** – Xử lý hàng chục triệu bản ghi không gian-thời gian, đảm bảo idempotent và tự phục hồi khi lỗi.
- **Tích hợp đa nguồn** – Kết hợp dữ liệu tội phạm với thời tiết (NOAA), điểm tụ tập (POI) và kinh tế - xã hội (US Census Bureau).
- **Dự báo học máy** – Mô hình XGBoost dự đoán nguy cơ tội phạm theo ô lưới H3 cho ngày tiếp theo.
- **Trực quan hóa** – Dashboard KPI 2D (Looker Studio) và bản đồ heatmap 3D tương tác (Kepler.gl).

### Phạm vi

| | |
|---|---|
| **Địa lý** | Thành phố Chicago, bang Illinois, Hoa Kỳ |
| **Dữ liệu** | Chicago Crime Dataset (2001 – hiện tại) |
| **Môi trường** | Google Cloud Platform |

---

## 🏗️ 2. Kiến trúc hệ thống

Hệ thống theo kiến trúc **Medallion Architecture** với 3 tầng triển khai và 6 stage xử lý dữ liệu.

### Ba tầng triển khai

```
┌─────────────────────────────────────────────────────────────┐
│  Tier 1 – Orchestration (Local)                             │
│  Docker + Apache Airflow  →  Lập lịch & điều phối pipeline  │
└────────────────────────┬────────────────────────────────────┘
                         │ kích hoạt
┌────────────────────────▼────────────────────────────────────┐
│  Tier 2 – Data & Processing (Google Cloud Platform)         │
│  GCS (Data Lake)  →  Dataproc/PySpark  →  BigQuery (DWH)    │
└────────────────────────┬────────────────────────────────────┘
                         │ phục vụ
┌────────────────────────▼────────────────────────────────────┐
│  Tier 3 – Presentation (Web Browser)                        │
│  Looker Studio (Dashboard 2D)  +  Kepler.gl (Bản đồ 3D)    │
└─────────────────────────────────────────────────────────────┘
```

### Sáu stage xử lý dữ liệu

| Stage | Tên | Mô tả |
|---|---|---|
| 1 | **Ingestion** | Thu thập dữ liệu từ BigQuery Public Data, NOAA, US Census → GCS |
| 2 | **Bronze Layer** | Validate schema, partition dữ liệu thô theo ngày |
| 3 | **Silver Layer** | Làm sạch, chuẩn hóa, H3 encoding, deduplication (PySpark) |
| 4 | **Gold Layer** | JOIN với thời tiết & kinh tế - xã hội, feature engineering |
| 5 | **Analytics & ML** | Huấn luyện XGBoost (hàng tuần), dự báo điểm nóng (hàng ngày) |
| 6 | **Serving** | Looker Studio + Kepler.gl render dashboard và bản đồ |

---

## 🛠️ 3. Công nghệ sử dụng

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| Orchestration | Apache Airflow (Docker) | Lập lịch, retry, giám sát pipeline |
| Data Lake | Google Cloud Storage | Lưu trữ dữ liệu thô theo phân vùng ngày |
| ETL Engine | PySpark trên Dataproc | Xử lý phân tán, làm sạch & biến đổi |
| Data Warehouse | Google BigQuery | Lưu trữ, truy vấn BI và ML |
| ML | XGBoost + PySpark | Huấn luyện & dự báo điểm nóng |
| Dashboard 2D | Looker Studio | KPI, biểu đồ xu hướng, bộ lọc tương tác |
| Bản đồ 3D | Kepler.gl | Heatmap H3, time-playback |
| Ngôn ngữ | Python 3.10+ | Pipeline, ETL, ML scripts |

### Yêu cầu phiên bản

```
Python     >= 3.10
PySpark    3.3
XGBoost    1.7
h3         4.1
pandas     2.0
Kepler.gl  >= 2.5
```

---

## 🗄️ 4. Cấu trúc dữ liệu

Dữ liệu trong BigQuery theo mô hình **Star Schema** với 5 bảng chính:

```
fact_crimes              ← Bảng trung tâm (phân vùng theo incident_date)
    ├── dim_weather      ← JOIN theo incident_date
    └── dim_socioeconomic ← JOIN theo community_area

enriched_crime_data      ← Bảng tổng hợp (phục vụ ML & Dashboard)
prediction_results       ← Kết quả dự báo XGBoost
```

### Bảng chính: `fact_crimes`

| Cột | Kiểu | Mô tả |
|---|---|---|
| `case_number` | STRING | Mã vụ việc (PK) |
| `incident_date` | DATE | Ngày xảy ra (Partition key) |
| `h3_index` | STRING | Mã ô lưới H3 độ phân giải 8 (Clustering key) |
| `primary_type` | STRING | Loại tội phạm (đã chuẩn hóa) |
| `latitude` / `longitude` | FLOAT64 | Tọa độ địa điểm |
| `community_area` | INTEGER | Mã khu vực cộng đồng (1–77) |
| `arrest` | BOOLEAN | Có bắt giữ hay không |

### Bảng dự báo: `prediction_results`

| Cột | Kiểu | Mô tả |
|---|---|---|
| `prediction_date` | DATE | Ngày được dự báo |
| `h3_index` | STRING | Ô lưới H3 |
| `crime_probability` | FLOAT64 | Xác suất tội phạm (0.0 – 1.0) |
| `risk_level` | STRING | LOW / MEDIUM / HIGH / CRITICAL |
| `model_version` | STRING | Format: `YYYY-MM-DD.v{n}` |

### Chiến lược tối ưu BigQuery

- **Date Partitioning** trên `incident_date` → giảm scan khi truy vấn theo khoảng thời gian
- **Clustering** theo `h3_index` → tăng tốc truy vấn theo vùng địa lý
- **BI Engine cache** (1GB RAM) → thời gian phản hồi Looker Studio < 5 giây

---

## ⚙️ 5. Cài đặt

### Yêu cầu tiên quyết

- [ ] Tài khoản Google Cloud Platform với quyền tạo Project
- [ ] Google Cloud SDK đã cài đặt và xác thực
- [ ] Docker & Docker Compose
- [ ] Python 3.10+
- [ ] Git

### Bước 1 – Tạo GCP Project và kích hoạt API

```bash
# Kích hoạt các API cần thiết
gcloud services enable storage.googleapis.com \
                       dataproc.googleapis.com \
                       bigquery.googleapis.com

# Tạo Service Account với các quyền: Storage Admin, BigQuery Admin, Dataproc Admin
# Tải xuống key JSON và đặt vào thư mục config/
```

### Bước 2 – Tạo hạ tầng GCS và BigQuery

```bash
# Tạo GCS bucket
gsutil mb -l us-central1 gs://cdash-data-lake

# Tạo BigQuery dataset
bq mk --location=US cdash_warehouse

# Tạo schema các bảng
python config/setup_bq_schema.py
```

### Bước 3 – Tạo Dataproc Cluster

```bash
# Clone repository
git clone https://github.com/gwncmh/cdash.git
cd cdash

# Tạo cluster
gcloud dataproc clusters create cdash-cluster \
    --region=us-central1 \
    --num-workers=1 \
    --worker-machine-type=e2-standard-4
```

### Bước 4 – Khởi động Apache Airflow

```bash
# Khởi động Airflow bằng Docker Compose
docker-compose up -d

# Truy cập Airflow UI
open http://localhost:8080
```

Cấu hình các Airflow Variables sau trong UI:

| Variable | Mô tả |
|---|---|
| `PROJECT_ID` | Google Cloud Project ID |
| `BUCKET_NAME` | Tên GCS bucket |
| `BQ_DATASET` | Tên BigQuery dataset |
| `CLUSTER_NAME` | Tên Dataproc cluster |

### Cấu trúc thư mục GCS

```
gs://cdash-data-lake/
├── raw/
│   ├── chicago_crime/{year}/{month}/{day}/
│   ├── noaa_weather/{year}/{month}/{day}/
│   └── census_socioeconomic/{year}/
└── models/
    └── YYYY-MM-DD.v{n}/
```

---

## 🚀 6. Vận hành

### Kích hoạt pipeline lần đầu

1. Truy cập Airflow UI tại `http://localhost:8080`
2. Tìm DAG `cdash_daily_pipeline` → bật trạng thái (Unpause)
3. Nhấn **Trigger DAG** để chạy thủ công lần đầu
4. Quá trình nạp dữ liệu từ 2001 đến hiện tại mất khoảng **2–4 giờ**
5. Theo dõi qua **Graph View** trong Airflow UI

> Sau lần đầu, pipeline tự động chạy hàng ngày theo lịch đã cấu hình.

### Giám sát thường nhật

```bash
# Kiểm tra số bản ghi fact_crimes hôm nay
bq query --use_legacy_sql=false \
  "SELECT COUNT(*) FROM cdash_warehouse.fact_crimes
   WHERE incident_date = CURRENT_DATE()"

# Kiểm tra kết quả dự báo hôm nay
bq query --use_legacy_sql=false \
  "SELECT COUNT(*), model_version FROM cdash_warehouse.prediction_results
   WHERE prediction_date = CURRENT_DATE()
   GROUP BY model_version"
```

### Chỉ số chất lượng mô hình

| Chỉ số | Ngưỡng tối thiểu |
|---|---|
| AUC-ROC | ≥ 0.75 |
| F1-Score | ≥ 0.60 |

Mô hình được retrain **hàng tuần** với dữ liệu 30 ngày gần nhất. Nếu chỉ số thấp hơn ngưỡng, Airflow gửi email cảnh báo.

---

## 🔧 7. Xử lý lỗi

| Lỗi | Nguyên nhân | Cách khắc phục |
|---|---|---|
| Task Ingestion thất bại | API nguồn gián đoạn | Kiểm tra log, chờ nguồn phục hồi hoặc cập nhật endpoint |
| Sensor chờ quá lâu | API nguồn chậm | Tăng `timeout` cho Sensor trong DAG config |
| Spark job OOM Error | Dữ liệu tăng đột biến | Tăng executor memory hoặc thêm worker node |
| BigQuery quota exceeded | Vượt 10TB free tier/ngày | Tối ưu truy vấn, thêm điều kiện filter phân vùng |
| ML AUC-ROC thấp | Dữ liệu huấn luyện ít | Kiểm tra bản ghi 30 ngày gần nhất, thêm feature mới |
| GeoJSON không cập nhật | Auto-Exporter thất bại | Kiểm tra log, chạy lại thủ công |
| Dashboard Looker trống | Múi giờ bộ lọc sai | Đảm bảo filter đúng timezone, data đã load đến hôm nay |

> Airflow tự động retry **3 lần** trước khi gửi email cảnh báo cho kỹ sư dữ liệu.

---

## 📊 Hiệu năng mục tiêu

| Tiêu chí | Chỉ số |
|---|---|
| Thời gian phản hồi truy vấn | < 5 giây với 10 triệu dòng |
| Thông lượng ETL | 1 triệu bản ghi trong < 30 phút |
| Độ tin cậy pipeline | Tự phục hồi, không mất dữ liệu khi lỗi |

---

## 📄 Giấy phép

Dự án phục vụ mục đích học thuật tại Trường Đại học Công nghệ – ĐHQGHN.  
Toàn bộ dữ liệu đầu vào là dữ liệu công khai, đã được ẩn danh hóa. Hệ thống không lưu trữ, xử lý hay hiển thị thông tin cá nhân định danh.

---

*Hà Nội, 2026*
