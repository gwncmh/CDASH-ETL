"""
etl_gis_features.py
===================
PySpark job: Tính H3 index cho từng vụ tội phạm
và ghi lại vào fact_crimes trong BigQuery.

Yêu cầu: pip install h3 trên Dataproc workers
(khai báo trong CLUSTER_CONFIG initialization_actions hoặc
 dùng --pip-packages khi submit job)

Upload lên: gs://YOUR_BUCKET/scripts/etl_gis_features.py
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


def add_h3_index(spark: SparkSession, args):
    # ── Đọc fact_crimes từ BigQuery ─────────────────────────────────────────
    print(f"[GIS] Reading fact_crimes from BigQuery")
    df = (
        spark.read
        .format("bigquery")
        .option("table", f"{args.project}.{args.dataset}.fact_crimes")
        .load()
    )

    # ── Tính H3 index bằng UDF ───────────────────────────────────────────────
    # h3.geo_to_h3(lat, lng, resolution) → string index
    # Resolution 8 ≈ ô lục giác ~0.7 km², phù hợp phân tích khu phố
    resolution = int(args.resolution)

    @F.udf(StringType())
    def geo_to_h3(lat, lng):
        if lat is None or lng is None:
            return None
        try:
            import h3
            return h3.geo_to_h3(float(lat), float(lng), resolution)
        except Exception:
            return None

    df = df.withColumn("h3_index", geo_to_h3(F.col("latitude"), F.col("longitude")))

    # ── Thêm h3_resolution để tiện filter sau này ───────────────────────────
    df = df.withColumn("h3_resolution", F.lit(resolution))

    # ── Ghi lại BigQuery (overwrite) ─────────────────────────────────────────
    bq_table = f"{args.project}:{args.dataset}.fact_crimes"
    print(f"[GIS] Writing h3_index back to {bq_table}")

    (
        df.write
        .format("bigquery")
        .option("table", bq_table)
        .option("temporaryGcsBucket", args.bucket)
        .option("writeMethod", "indirect")
        .mode("overwrite")
        .save()
    )

    # ── Stats ─────────────────────────────────────────────────────────────────
    total = df.count()
    indexed = df.filter(F.col("h3_index").isNotNull()).count()
    print(f"[GIS] ✓ H3 index computed: {indexed}/{total} rows ({100*indexed//total}%)")

    # Top 10 ô nóng nhất
    print("[GIS] Top 10 H3 cells by crime count:")
    (
        df.groupBy("h3_index")
        .count()
        .orderBy(F.desc("count"))
        .show(10)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",    required=True)
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--bucket",     required=True)
    parser.add_argument("--resolution", default="8")
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("CrimeETL-GISFeatures")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    add_h3_index(spark, args)
    spark.stop()
