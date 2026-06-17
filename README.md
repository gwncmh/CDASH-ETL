# CDASH – Crime Data Analytics & Spatial Hotspot Detection System

**Trường Đại học Công nghệ – Đại học Quốc gia Hà Nội**  
Giảng viên hướng dẫn: PGS.TS. Nguyễn Ngọc Hóa

| Thành viên | MSSV |
|---|---|
| Trần Bình Dương | 23021514 |
| Nguyễn Công Mạnh Hùng | 23021567 |
| Nguyễn Mai Thanh Thư | 23021731 |

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Công nghệ sử dụng](#3-công-nghệ-sử-dụng)
4. [Cấu trúc dữ liệu](#4-cấu-trúc-dữ-liệu)
5. [Cài đặt](#5-cài-đặt)
6. [Vận hành](#6-vận-hành)
7. [Mô hình ML](#7-mô-hình-ml)
8. [Xử lý lỗi](#8-xử-lý-lỗi)

---

## 1. Tổng quan

CDASH là một hệ thống **Data Lakehouse** triển khai trên Google Cloud Platform, giải quyết bài toán phân tích và dự báo tội phạm đô thị theo hướng chủ động. Hệ thống xử lý dữ liệu tội phạm thành phố Chicago từ năm 2001 đến hiện tại, kết hợp với dữ liệu thời tiết, điểm tụ tập (POI) và đặc điểm kinh tế – xã hội để dự báo điểm nóng tội phạm theo ngày.

**Phạm vi địa lý:** Thành phố Chicago, bang Illinois, Hoa Kỳ  
**Môi trường triển khai:** Google Cloud Platform (Google Cloud Storage + Dataproc + BigQuery)

### Bài toán giải quyết

- **Data Pipeline tự động** – Xử lý hàng chục triệu bản ghi không gian-thời gian với đảm bảo idempotent và tự phục hồi khi lỗi (Airflow retry 2 lần).
- **Tích hợp đa nguồn** – Dữ liệu từ BigQuery Public, thời tiết từ NOAA, điểm tụ tập từ OpenStreetMap, chỉ số kinh tế – xã hội từ ACS Census API.
- **Dự báo học máy** – Mô hình LightGBM GroupBlend (temporal / spatial / context) dự đoán nguy cơ tội phạm theo ô lưới H3 cho ngày tiếp theo.
- **Trực quan hóa** – Dashboard KPI 2D (Looker Studio) và bản đồ heatmap 3D tương tác (Kepler.gl).

---

## 2. Kiến trúc hệ thống

Hệ thống theo kiến trúc **Medallion** với 3 tầng triển khai và 6 stage xử lý dữ liệu.

### Ba tầng triển khai

```
┌──────────────────────────────────────────────────────────────┐
│  Tier 1 – Orchestration (Local)                              │
│  Docker + Apache Airflow  →  Lập lịch & điều phối pipeline  │
└───────────────────────────┬──────────────────────────────────┘
                            │ kích hoạt
┌───────────────────────────▼──────────────────────────────────┐
│  Tier 2 – Data & Processing (Google Cloud Platform)          │
│  GCS (Data Lake)  →  Dataproc / PySpark  →  BigQuery (DWH)  │
└───────────────────────────┬──────────────────────────────────┘
                            │ phục vụ
┌───────────────────────────▼──────────────────────────────────┐
│  Tier 3 – Presentation (Web Browser)                         │
│  Looker Studio (Dashboard 2D)  +  Kepler.gl (Bản đồ 3D)     │
└──────────────────────────────────────────────────────────────┘
```

### Sáu stage xử lý dữ liệu

| Stage | Tên | Mô tả |
|---|---|---|
| 1 | **Ingestion** | Thu thập Chicago Crime từ BigQuery Public Data, thời tiết từ NOAA CDO API, điểm tụ tập từ OSMnx, chỉ số kinh tế – xã hội từ ACS Census API |
| 2 | **Bronze Layer** | Validate schema (`MANDATORY_COLUMNS`), parse timestamp (ISO-8601 & Chicago Portal format), partition theo ngày lên GCS |
| 3 | **Silver Layer** | Chuẩn hóa schema, lọc không gian (bounding box Chicago), lọc thời gian, H3 encoding (resolution 8) bằng Pandas UDF, deduplication bằng Left Anti-Join |
| 4 | **Gold Layer** | JOIN dữ liệu thời tiết (nearest-station Haversine), điểm tụ tập (H3 hash join), kinh tế – xã hội; tính rolling features; ghi vào BigQuery `enriched_crime_data` |
| 5 | **Analytics & ML** | Huấn luyện LightGBM GroupBlend (hàng tuần), dự báo điểm nóng (hàng ngày), ghi vào `prediction_results` |
| 6 | **Serving** | Flask exporter xuất GeoJSON; Nginx serve dashboard HTML + GeoJSON; Looker Studio + Kepler.gl render |

### DAG Airflow

```
start
  ├── ingest_chicago_crime
  ├── ingest_weather
  ├── ingest_socioeconomic
  ├── ingest_poi
  └── create_dataproc_cluster
        ↓
      etl_bronze  ←  [create_cluster + ingest_crime]
        ↓
      etl_silver
        ↓
      etl_gold    ←  [etl_silver + ingest_weather + ingest_socioeconomic + ingest_poi]
        ↓
      delete_dataproc_cluster
        ↓
      ml_train_and_predict
        ↓
      export_crimes_geojson + export_forecast_geojson
        ↓
      end
```

## 3. Công nghệ sử dụng

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| Orchestration | Apache Airflow 2.8.1 (Docker) | Lập lịch, retry, giám sát pipeline |
| Data Lake | Google Cloud Storage | Lưu trữ raw/bronze/silver theo phân vùng ngày |
| ETL Engine | PySpark 3.3 trên Dataproc | Xử lý phân tán, làm sạch & biến đổi |
| Data Warehouse | Google BigQuery | Lưu trữ, truy vấn BI và ML |
| ML | LightGBM + GroupBlend | Huấn luyện & dự báo điểm nóng theo H3 |
| GIS | H3 4.1 (Pandas UDF) | Mã hóa ô lưới không gian |
| Dashboard 2D | Looker Studio | KPI, biểu đồ xu hướng, bộ lọc tương tác |
| Bản đồ 3D | Kepler.gl | Heatmap H3, time-playback |
| Serving | Flask + Nginx (Docker) | Export GeoJSON, serve dashboard |
| Ngôn ngữ | Python 3.11 | Pipeline, ETL, ML, Flask API |

### Yêu cầu phiên bản

```
Python     >= 3.10
PySpark    3.3
LightGBM   4.6
h3         4.1
pandas     2.0
Kepler.gl  >= 2.5
```

---

## 4. Cấu trúc dữ liệu

Dữ liệu trong BigQuery theo mô hình **Star Schema**:

```
enriched_crime_data      ← Bảng tổng hợp (phục vụ ML & Dashboard)
    ├── dim_weather      ← JOIN theo incident_date + station_id
    └── dim_socioeconomic ← JOIN theo community_area

prediction_results       ← Kết quả dự báo LightGBM GroupBlend
```

### Bảng chính: `enriched_crime_data`

| Cột | Kiểu | Mô tả |
|---|---|---|
| `case_number` | STRING | Mã vụ việc |
| `date` | DATE | Ngày xảy ra (Partition key theo tháng) |
| `h3_index` | STRING | Mã ô lưới H3 resolution 8 (Clustering key) |
| `primary_type` | STRING | Loại tội phạm (đã chuẩn hóa UPPER TRIM) |
| `latitude` / `longitude` | FLOAT64 | Tọa độ địa điểm |
| `community_area` | INTEGER | Mã khu vực (1–77) |
| `hour_of_day`, `day_of_week`, `month` | INTEGER | Đặc trưng thời gian |
| `temp_max`, `precipitation`, `wind_speed` | FLOAT64 | Thời tiết từ NOAA |
| `dist_nearest_*` | FLOAT64 | Khoảng cách tới school/station/park/nightlife |
| `crime_density_7d`, `crime_density_30d` | FLOAT64 | Rolling crime density |
| `unemployment_rate`, `hardship_index` | FLOAT64 / INT | Kinh tế – xã hội |
| `population_density`, `per_capita_income` | FLOAT64 | Dân số & thu nhập |

### Bảng dự báo: `prediction_results`

| Cột | Kiểu | Mô tả |
|---|---|---|
| `prediction_date` | DATE | Ngày được dự báo |
| `h3_index` | STRING | Ô lưới H3 |
| `hotspot_score` | FLOAT64 | Điểm rủi ro tổng hợp (GroupBlend) |
| `crime_probability` | FLOAT64 | Alias của `hotspot_score` (backward-compatible) |
| `risk_level` | STRING | LOW / MEDIUM / HIGH / CRITICAL (top-k ranking) |
| `confidence_level` | STRING | LOW / MEDIUM / HIGH |
| `dominant_type` | STRING | VIOLENT / PROPERTY / MIXED / UNKNOWN |
| `temporal_score`, `spatial_score`, `context_score` | FLOAT64 | Điểm từng nhóm model |
| `model_version` | STRING | Format: `YYYY-MM-DD.hotspot_v24_groupblend_confidence` |
| `community_area_name` | STRING | Tên khu vực cộng đồng |

### Chiến lược tối ưu BigQuery

- **Partition theo tháng** trên `date` → giảm scan khi truy vấn theo khoảng thời gian.
- **Clustering** theo `h3_index` → tăng tốc truy vấn địa lý.
- **BI Engine cache** (1 GB RAM) → phản hồi Looker Studio < 5 giây.

### Cấu trúc GCS

```
gs://cdash-data-lake/
├── raw/
│   ├── chicago_crime/{year}/{month}/{day}/data.csv
│   ├── chicago_crime/{year}/{month}/{day}/_SUCCESS
│   ├── weather/weather.csv
│   ├── weather/chicago_stations.csv
│   ├── socioeconomic/socioeconomic.csv
│   ├── socioeconomic/raw_population.csv
│   ├── poi/chicago_pois.csv
│   └── crosswalk/tract_to_ca.csv
├── bronze/chicago_crime/part_year=*/part_month=*/part_day=*/
├── silver/chicago_crime/
└── scripts/
    ├── stage2_bronze/data_ingestion.py
    ├── stage3_silver_main.py + stage3_silver.zip
    ├── stage4_gold_main.py  + stage4_gold.zip
    ├── utils.zip
    └── init_pip.sh
```

---

## 5. Cài đặt

### Yêu cầu tiên quyết

- Tài khoản Google Cloud Platform với quyền tạo Project
- Google Cloud SDK đã cài đặt và xác thực (`gcloud auth login`)
- Docker & Docker Compose
- Python 3.10+
- Git

### Bước 1 – Clone và chạy setup script

```bash
git clone https://github.com/gwncmh/cdash.git
cd cdash
bash scripts/infra/setup.sh
```

Script tự động: kiểm tra prerequisites, tạo thư mục, tạo GCP Service Account với các quyền `Storage Admin / BigQuery Admin / Dataproc Editor`, upload PySpark scripts lên GCS, và khởi động Airflow.

### Bước 2 – Kích hoạt GCP APIs

```bash
gcloud services enable storage.googleapis.com \
                       dataproc.googleapis.com \
                       bigquery.googleapis.com
```

### Bước 3 – Tạo hạ tầng GCS và BigQuery

```bash
gsutil mb -l us-central1 gs://cdash-data-lake
bq mk --location=US cdash_warehouse
python config/setup_bq_schema.py
```

### Bước 4 – Cấu hình `config/config.py`

```python
GCP_PROJECT_ID = "your-project-id"
GCS_BUCKET     = "your-bucket-name"
BQ_DATASET     = "cdash_warehouse"
NOAA_API_TOKEN = "your-noaa-token"   # lấy tại https://www.ncdc.noaa.gov/cdo-web/token
```

### Bước 5 – Khởi động Airflow

```bash
docker compose up airflow-init
docker compose up -d
```

Truy cập Airflow UI tại `http://localhost:8080` (username: `admin` / password: `admin`).

Tạo Airflow Connection `google_cloud_default`:
- Conn Type: `Google Cloud`
- Keyfile Path: `/opt/airflow/secrets/gcp-sa-key.json`

---

## 6. Vận hành

### Kích hoạt pipeline lần đầu (Full Load)

1. Vào Airflow UI → DAG `crime_analytics_pipeline` → bật **Unpause**.
2. Nhấn **Trigger DAG** → truyền config `{"full_load": true}` để nạp dữ liệu từ 2001.
3. Quá trình full load mất khoảng **8-9 giờ**.
4. Theo dõi qua **Graph View** trong Airflow UI.

Sau lần đầu, mỗi lần khởi động Docker là hệ thống sẽ chạy theo incremental load.

### Incremental load hàng tuần

Mỗi lần chạy, pipeline xử lý khoảng **17 ngày dữ liệu** kết thúc ở `execution_date - 5 ngày` (buffer để đảm bảo dữ liệu nguồn ổn định). Deduplication bằng Left Anti-Join trên `unique_key` đảm bảo không ghi trùng.

### Giám sát thường nhật

```bash
# Kiểm tra số bản ghi enriched hôm nay
bq query --use_legacy_sql=false \
  "SELECT COUNT(*) FROM cdash_warehouse.enriched_crime_data
   WHERE date = CURRENT_DATE()"

# Kiểm tra kết quả dự báo mới nhất
bq query --use_legacy_sql=false \
  "SELECT risk_level, COUNT(*) as n, model_version
   FROM cdash_warehouse.prediction_results
   WHERE prediction_date = (SELECT MAX(prediction_date) FROM cdash_warehouse.prediction_results)
   GROUP BY 1, 3
   ORDER BY 1"
```

### Export GeoJSON cho Kepler.gl

```bash
# Trigger export thủ công
curl -X POST http://localhost:5000/export/all

# Kiểm tra trạng thái file
curl http://localhost:5000/status
```

File được ghi vào `/exports/` và được Nginx serve tại `http://localhost:3000/exports/`.

---

## 7. Mô hình ML

### Kiến trúc GroupBlend (v24)

Pipeline ML huấn luyện **3 mô hình LightGBM độc lập**, mỗi mô hình sử dụng một nhóm đặc trưng riêng, sau đó blend kết quả:

| Nhóm | Trọng số | Loại đặc trưng |
|---|---|---|
| `temporal` | 40% | Rolling mean/sum/std (3–90 ngày), lag, EWM, burst ratio, momentum |
| `spatial` | 40% | H3 spatial prior, neighbor spillover (ring-1/ring-2), community context |
| `context` | 20% | Socioeconomic, POI, crime-type composition, arrest deterrence |

**Target:** `next_hotspot = 1` nếu số vụ án ngày hôm sau ≥ 2 tại ô H3 đó.

### Phân loại rủi ro (top-k ranking)

| Mức | Tiêu chí |
|---|---|
| CRITICAL | Top 5% ô H3 theo `hotspot_score` |
| HIGH | Top 6–15% |
| MEDIUM | Top 16–30% |
| LOW | 70% còn lại |

### Ngưỡng chất lượng mô hình

| Chỉ số | Ngưỡng tối thiểu |
|---|---|
| AUC-ROC | ≥ 0.75 |
| F1-Score | ≥ 0.50 |

Nếu không đạt ngưỡng, Airflow ghi cảnh báo vào log. Mô hình được retrain **mỗi lần chạy** với dữ liệu 320 ngày gần nhất.

### Chống data leakage

- Rolling features tính trên `shift(-1)` (không đếm vụ án hiện tại).
- Spatial prior (`h3_train_*`, `comm_train_*`) chỉ tính trên tập train, không leak sang val/test.
- Assertion kiểm tra overlap giữa các split trước khi huấn luyện.

---

## 8. Xử lý lỗi

| Lỗi | Nguyên nhân | Cách khắc phục |
|---|---|---|
| Task Ingestion thất bại | API nguồn gián đoạn (NOAA / BigQuery Public) | Kiểm tra log, chờ nguồn phục hồi; Airflow tự retry 3 lần |
| `_SUCCESS` marker không tồn tại | Stage 1 chưa hoàn thành | Chạy lại `ingest_chicago_crime` thủ công với `--overwrite` |
| Stage 3 không có partition mới | Dữ liệu Bronze chưa có trong khoảng ngày | Kiểm tra Bronze GCS path, chạy lại Stage 2 |
| Spark job OOM Error | Dữ liệu tăng đột biến | Tăng `spark.executor.memory` trong `CLUSTER_CONFIG` hoặc thêm worker |
| BigQuery quota exceeded | Vượt 10 TB free tier / ngày | Tối ưu truy vấn, thêm điều kiện partition filter |
| Weather JOIN null > 50% | Station CSV sai hoặc thiếu | Kiểm tra `chicago_stations.csv` trên GCS, đảm bảo cột `ID/LATITUDE/LONGITUDE` đúng |
| H3 null ratio > 5% (DQC fail) | Tọa độ ngoài bounding box Chicago | Kiểm tra bộ lọc `CHICAGO_BOUNDS` trong `data_cleaner.py` |
| ML AUC-ROC thấp | Dữ liệu huấn luyện ít hoặc drift | Kiểm tra 320 ngày gần nhất, xem log ablation study |
| GeoJSON không cập nhật | Flask exporter thất bại | Kiểm tra log container `exporter`, gọi lại `POST /export/all` |
| Dashboard Looker trống | Múi giờ bộ lọc sai hoặc data chưa load | Đảm bảo filter đúng timezone UTC, kiểm tra `enriched_crime_data` |

---

## Hiệu năng mục tiêu

| Tiêu chí | Chỉ số |
|---|---|
| Thời gian phản hồi truy vấn BI | < 5 giây |
| Thông lượng ETL Silver | ~1 triệu bản ghi < 30 phút |
| Độ tin cậy pipeline | Idempotent, tự phục hồi, không mất dữ liệu khi lỗi |
| Tần suất cập nhật dự báo | Mỗi lần chạy |

---

## Giấy phép

Dự án phục vụ mục đích học thuật tại Trường Đại học Công nghệ – ĐHQGHN.  
Toàn bộ dữ liệu đầu vào là dữ liệu công khai, đã được ẩn danh hóa. Hệ thống không lưu trữ, xử lý hay hiển thị thông tin cá nhân định danh.

---

*Hà Nội, 2026*
