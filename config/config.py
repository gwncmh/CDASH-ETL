# ============================================================
#  config.py — Thay tất cả giá trị YOUR_* trước khi chạy
# ============================================================

GCP_PROJECT_ID      = "sage-mind-489618-n5"          # Project gốc của nhóm
GCP_REGION          = "us-central1"              
GCP_ZONE            = "us-central1-a"                

GCS_BUCKET          = "chicago-crime-raw-group15"    # KHÔNG CÓ chữ gs:// ở trước nhé
GCS_SCRIPTS_PATH    = "scripts"                   
GCS_DATA_PATH       = "data/raw"                     # Lưu ý: Thư hãy gom các file csv vào thư mục này trên Bucket

BQ_DATASET          = "cdash_warehouse"           
BQ_LOCATION         = "US"                           

# Tên các bảng BigQuery (Được thiết kế theo mô hình Star Schema cực kỳ chuyên nghiệp)
BQ_TABLE_CRIMES         = "fact_crimes"           # Bảng Fact: Chứa dữ liệu vụ án cốt lõi
BQ_TABLE_WEATHER        = "dim_weather"           # Bảng Dim: Chứa dữ liệu thời tiết
BQ_TABLE_SOCIOECONOMIC  = "dim_socioeconomic"     # Bảng Dim: Chứa dữ liệu kinh tế - xã hội
BQ_TABLE_ENRICHED       = "enriched_crime_data"   # Bảng tổng hợp (Join 3 bảng trên) để đưa vào AI
BQ_TABLE_PREDICTIONS    = "prediction_results"    # Bảng chứa kết quả dự đoán của mô hình XGBoost

# Dataproc cluster config (ĐÃ ĐƯỢC CHỈNH SỬA ĐỂ TIẾT KIỆM TIỀN CHO SINH VIÊN)
DATAPROC_CLUSTER_NAME   = "crime-etl-cluster"
DATAPROC_NUM_WORKERS    = 2                      # Quan trọng: Đổi về 0 để chạy Single Node (Tiết kiệm 66% tiền)
DATAPROC_MASTER_TYPE    = "e2-standard-2"         # Dùng dòng máy e2 rẻ hơn n1, vẫn dư sức chạy Big Data
DATAPROC_WORKER_TYPE    = "e2-standard-2"         # Thực tế dòng này sẽ bị bỏ qua vì số worker = 0

# Airflow connection id 
GCP_CONN_ID             = "google_cloud_default"
# NOAA Weather API
NOAA_API_TOKEN  = "ystxZEXaKsDWcEwfgFOIZFVpIASQCHNy"   # token từ pull_weather.py
NOAA_STATION_ID = "GHCND:USW00094846"                   # station Chicago
# Chicago Community Area
CHICAGO_COMMUNITY_AREA_NAMES = {
    1: "Rogers Park", 2: "West Ridge", 3: "Uptown", 4: "Lincoln Square",
    5: "North Center", 6: "Lake View", 7: "Lincoln Park", 8: "Near North Side",
    9: "Edison Park", 10: "Norwood Park", 11: "Jefferson Park", 12: "Forest Glen",
    13: "North Park", 14: "Albany Park", 15: "Portage Park", 16: "Irving Park",
    17: "Dunning", 18: "Montclare", 19: "Belmont Cragin", 20: "Hermosa",
    21: "Avondale", 22: "Logan Square", 23: "Humboldt Park", 24: "West Town",
    25: "Austin", 26: "West Garfield Park", 27: "East Garfield Park", 28: "Near West Side",
    29: "North Lawndale", 30: "South Lawndale", 31: "Lower West Side", 32: "Loop",
    33: "Near South Side", 34: "Armour Square", 35: "Douglas", 36: "Oakland",
    37: "Fuller Park", 38: "Grand Boulevard", 39: "Kenwood", 40: "Washington Park",
    41: "Hyde Park", 42: "Woodlawn", 43: "South Shore", 44: "Chatham",
    45: "Avalon Park", 46: "South Chicago", 47: "Burnside", 48: "Calumet Heights",
    49: "Roseland", 50: "Pullman", 51: "South Deering", 52: "East Side",
    53: "West Pullman", 54: "Riverdale", 55: "Hegewisch", 56: "Garfield Ridge",
    57: "Archer Heights", 58: "Brighton Park", 59: "McKinley Park", 60: "Bridgeport",
    61: "New City", 62: "West Elsdon", 63: "Gage Park", 64: "Clearing",
    65: "West Lawn", 66: "Chicago Lawn", 67: "West Englewood", 68: "Englewood",
    69: "Greater Grand Crossing", 70: "Ashburn", 71: "Auburn Gresham",
    72: "Beverly", 73: "Washington Heights", 74: "Mount Greenwood",
    75: "Morgan Park", 76: "O'Hare", 77: "Edgewater",
}
