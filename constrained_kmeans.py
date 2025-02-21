import time
import warnings
import numpy as np

from graphframes import GraphFrame
import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, broadcast, pandas_udf
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField
from pyspark.sql.window import Window

warnings.filterwarnings("ignore", message="DataFrame.sql_ctx is an internal property, and will be removed in future releases")
warnings.filterwarnings("ignore", message="DataFrame constructor is internal. Do not directly use it.")


##########################
# 1) TIỀN XỬ LÝ MUST-LINK
##########################
def preprocess_must_link(
    df_data: DataFrame,
    df_ml: DataFrame,
    id_col: str = "id",
    features_col: str = "features"
):
    """
    Sử dụng GraphFrame để tính connected components từ must-link (df_ml),
    sau đó tính siêu điểm (superpoint) cho mỗi component bằng cách tính trung bình các vector của các điểm
    trong cùng component, không dùng UDF mà dựa trên hàm native của Spark.
    
    Các bước được đo thời gian bằng cách sử dụng Python's time module và các action (count, collect)
    để kích hoạt thực thi tính toán phân tán.
    
    Output:
      - df_superpoints: DataFrame [root, features] với features là array<double> trung bình của các vector.
      - df_cc: DataFrame [id, root] ánh xạ mỗi id đến component (root).
    """
    # --- Step 1: Tạo vertices từ dữ liệu gốc ---
    df_vertices = df_data.select(
        F.col(id_col).alias("id"),
        F.col(features_col).alias("features"))
    df_vertices.cache()
    
    # --- Step 2: Tạo edges từ must-link pairs và thêm cạnh đảo ngược ---
    # Chuẩn hóa các cặp must-link: luôn đặt giá trị nhỏ hơn ở cột 'src'
    df_edges = df_ml.select(
        F.least(F.col("id1"), F.col("id2")).alias("src"),
        F.greatest(F.col("id1"), F.col("id2")).alias("dst"))
    df_edges.cache()
    
    # --- Step 3: Tạo GraphFrame và tính connected components ---
    gf = GraphFrame(df_vertices, df_edges)

    numPartitions = 40
    df_vertices = df_vertices.repartition(numPartitions, F.col("id"))
    df_edges = df_edges.repartition(numPartitions, F.col("src"))

    gf_cc = gf.connectedComponents()  # Trả về DataFrame với cột "id" và "component"
    df_cc = gf_cc.select("id", F.col("component").alias("root"))
    
    # --- Step 4: Join kết quả components với dữ liệu gốc để lấy features ---
    # Sử dụng lại df_vertices đã cache ở bước 1
    df_join = df_cc.join(df_vertices, on="id", how="left")
    df_join.cache()
    
    # --- Step 5: Chuyển DenseVector thành mảng (array<double>) ---
    df_join = df_join.withColumn("features_arr", vector_to_array("features"))
    # Lấy mẫu 1 dòng để đảm bảo cột được tính toán
    sample = df_join.select("features_arr").limit(1).collect()
    
    # --- Step 6: Xác định số chiều của vector từ mẫu ---
    if sample:
        dim = len(sample[0]["features_arr"])
    else:
        raise ValueError("Không có dữ liệu để xác định số chiều của vector")
    
    # --- Step 7: Tách mảng thành các cột riêng cho từng chiều (Cách tối ưu) ---
    # Tạo danh sách các cột từ features_arr cho từng chiều
    feat_columns = [F.col("features_arr")[i].alias(f"feat_{i}") for i in range(dim)]

    # Sử dụng select để thêm các cột mới cùng lúc (giữ lại các cột ban đầu nếu cần)
    df_join = df_join.select("*", *feat_columns)
    # Nếu không cần cột features_arr nữa, bạn có thể loại bỏ nó:
    # df_join = df_join.select(*[col for col in df_join.columns if col != "features_arr"], *feat_columns)

    # Kích hoạt tính toán với một mẫu nhỏ
    _ = df_join.select([f"feat_{i}" for i in range(dim)]).limit(1).collect()

    # --- Step 8: Gom nhóm theo component và tính trung bình cho mỗi chiều ---
    # Phân vùng lại theo key "root" để cân bằng dữ liệu trước khi groupBy
    df_join = df_join.repartition(numPartitions, F.col("root"))

    # Tạo danh sách biểu thức để tính trung bình cho từng cột
    agg_exprs = [F.avg(f"feat_{i}").alias(f"avg_{i}") for i in range(dim)]

    # Thực hiện groupBy và aggregation
    df_grouped = df_join.groupBy("root").agg(*agg_exprs)
    
    # --- Step 9: Ghép lại các giá trị trung bình thành 1 mảng ---
    avg_cols = [F.col(f"avg_{i}") for i in range(dim)]
    df_superpoints = df_grouped.withColumn("avg_features", F.array(*avg_cols)) \
                               .select(F.col("root"), F.col("avg_features").alias("features"))
    
    # Định nghĩa UDF chuyển array thành DenseVector
    to_vector_udf = udf(lambda arr: DenseVector(arr), VectorUDT())

    # Áp dụng UDF để chuyển đổi cột "features" (đã được tạo từ avg_features) sang DenseVector
    df_superpoints = df_superpoints.withColumn("features", to_vector_udf("features"))

    return df_superpoints, df_cc

##########################
# 2) CHẠY K-MEANS TRÊN SUPERPOINTS
##########################
def run_kmeans_on_superpoints(
    df_superpoints: DataFrame,
    k: int,
    seed: int = 42,
    maxIter: int = 20,
    tol: float = 1e-4
):
    """
    df_superpoints: DataFrame chứa các cột [root, features]
    Trả về DataFrame chứa các cột [root, cluster] sau khi áp dụng KMeans.
    """
    # Tạo mô hình KMeans với cấu hình tùy chọn (các tham số mặc định có thể được điều chỉnh)
    kmeans = KMeans(k=k, seed=seed, featuresCol="features", predictionCol="cluster",
                    maxIter=maxIter, tol=tol)
    model = kmeans.fit(df_superpoints)

    # Áp dụng mô hình để gán cluster cho từng superpoint
    df_clustered = model.transform(df_superpoints).select("root", "cluster")
    return df_clustered

##########################
# 2') GÁN CLUSTER VỀ CHO MỖI ID
##########################
def assign_clusters_back(
    df_cc: DataFrame,
    df_clustered_roots: DataFrame,
    df_data: DataFrame,
    id_col: str = "id"
) -> DataFrame:
    """
    df_cc: DataFrame chứa [id, root] (mapping từ id -> root)
    df_clustered_roots: DataFrame chứa [root, cluster]
    df_data: DataFrame gốc, chứa cột id
    
    Trả về: DataFrame cuối cùng gồm df_data cộng thêm cột "final_cluster".
    """
    # Nếu df_cc và df_clustered_roots nhỏ, có thể broadcast:
    df_cc = broadcast(df_cc)
    df_clustered_roots = broadcast(df_clustered_roots)

    # Bước 1: Join df_cc với df_clustered_roots dựa trên "root"
    df_id_cluster = df_cc.join(df_clustered_roots, on="root", how="left")
    
    # Sử dụng alias để phân biệt các cột
    df_id_cluster = df_id_cluster.alias("idc")
    df_data_alias = df_data.alias("data")
    
    # Bước 2: Join kết quả với df_data dựa trên cột id
    # Ở đây, df_id_cluster có cột "id" là id của điểm đã được gán component.
    df_final = df_data_alias.join(
        df_id_cluster,
        F.col("data." + id_col) == F.col("idc.id"),
        how="left"
    )
    
    # Tạo cột final_cluster từ cột "cluster" (từ df_clustered_roots)
    df_final = df_final.withColumn("final_cluster", F.col("idc.cluster"))
    
    # Drop cột "id" từ df_id_cluster nếu cần (tránh nhầm lẫn với cột id của df_data)
    df_final = df_final.drop(F.col("idc.id"))

    # Drop cột "root" và "cluster" từ df_final nếu không cần thiết
    df_final = df_final.drop("root", "cluster")
    
    return df_final

##########################
# 3) HẬU XỬ LÝ CANNOT-LINK
##########################
def compute_centroids_rdd(df, cluster_col="final_cluster", features_col="features"):
    """
    Tính centroid cho mỗi cụm từ DataFrame df sử dụng phương pháp phân tán qua RDD.
    
    Input:
      - df: DataFrame chứa cột cluster (cluster_col) và cột features (VectorUDT).
      - cluster_col: tên cột chứa nhãn cụm.
      - features_col: tên cột chứa vector đặc trưng.
      
    Output:
      - DataFrame với schema (cluster_id: int, centroid: VectorUDT) chứa centroid tính theo trung bình.
      
    Cách hoạt động:
      1. Chuyển DataFrame thành RDD với các cặp (cluster, (features, 1)).
      2. Dùng reduceByKey để cộng các vector và đếm số bản ghi theo cụm.
      3. Tính trung bình vector (centroid) cho mỗi cụm.
      4. Chuyển kết quả về DataFrame.
    """
    spark = SparkSession.builder.getOrCreate()

    # 1. Chuyển thành RDD: (cluster, (features, 1))
    rdd = df.select(cluster_col, features_col).rdd.map(
        lambda row: (row[cluster_col], (row[features_col], 1))
    )

    # 2. Giảm (reduceByKey): tổng hợp vector và đếm số bản ghi theo cluster
    def merge_values(a, b):
        # a, b: tuple (vector, count)
        vec_a, count_a = a
        vec_b, count_b = b
        # Cộng các vector: chuyển sang mảng, cộng, rồi tạo lại DenseVector
        sum_vec = DenseVector(np.add(vec_a.toArray(), vec_b.toArray()))
        return (sum_vec, count_a + count_b)
    
    aggregated = rdd.reduceByKey(merge_values)

    # 3. Tính trung bình vector cho mỗi cluster => centroid
    def compute_average(item):
        cluster, (sum_vec, count) = item
        avg_array = sum_vec.toArray() / count
        return (cluster, DenseVector(avg_array))
    
    centroids_rdd = aggregated.map(compute_average)
    
    # 4. Chuyển kết quả về DataFrame. Ta định nghĩa schema đơn giản.
    schema = StructType([
        StructField("cluster_id", IntegerType(), True),
        StructField("centroid", VectorUDT(), True)
    ])
    
    centroids_df = spark.createDataFrame(centroids_rdd, schema=schema)
    return centroids_df

# Pandas UDF: Tính khoảng cách Euclid vector hóa
@pandas_udf(DoubleType())
def euclid_dist_pd(feat_series: pd.Series, centroid_series: pd.Series) -> pd.Series:
    """
    Tính khoảng cách Euclid giữa các cặp vector từ hai cột.
    Mỗi phần tử của feat_series và centroid_series là list/array số thực.
    """
    return feat_series.combine(centroid_series, lambda f, c: float(np.linalg.norm(np.array(f) - np.array(c))))

def postprocess_cannot_link(
    df_result,
    df_cl,
    max_iter=5,
    id_col="id",
    features_col="features",
    cluster_col="final_cluster"
):
    """
    Hậu xử lý cannot-link bằng heuristic dựa trên khoảng cách.
    
    Các bước triển khai:
      1. Lặp tối đa max_iter vòng:
         a. Tính toán centroid cho mỗi cụm từ df_result.
         b. Xác định các cặp vi phạm cannot-link bằng cách join df_cl với df_result
            (kiểm tra xem các điểm trong cặp có cùng cluster không).
         c. Với mỗi vi phạm, tính khoảng cách từ điểm vi phạm (id2) đến centroid của
            các cụm khác (ngoại trừ cụm hiện tại).
         d. Lựa chọn cụm ứng viên cho điểm vi phạm dựa trên khoảng cách (chọn cụm có
            khoảng cách thấp nhất).
         e. Cập nhật nhãn cụm của điểm đó trong df_result.
         f. Kiểm tra tổng số vi phạm sau khi cập nhật; nếu không giảm, dừng vòng lặp.
    
    Trả về:
      DataFrame df_result đã được cập nhật nhãn cụm nhằm giảm vi phạm cannot-link.
    """
    spark = df_result.sparkSession

    # Khởi tạo biến lưu số vi phạm của vòng trước
    prev_violations = None

    # Cache df_result nếu cần (để tối ưu các phép join lặp đi lặp lại)
    df_result = df_result.cache()

    for it in range(max_iter):
        print(f"\n[Iter {it}] Bắt đầu vòng lặp hậu xử lý cannot-link")

        # --- Bước 1a: Tính centroid cho mỗi cụm từ df_result ---
        # Sử dụng hàm compute_centroids_rdd (đã tối ưu bằng RDD)
        df_centroids = compute_centroids_rdd(df_result, cluster_col, features_col)
        # Broadcast bảng centroid (số dòng nhỏ, giảm chi phí shuffle khi join)
        df_centroids = F.broadcast(df_centroids)
        df_centroids.cache()  # cache nếu cần

        # --- Bước 1b: Xác định các cặp vi phạm cannot-link ---
        # Tạo DataFrame chứa nhãn cụm theo id cho id1 và id2
        df_r1 = df_result.select(F.col(id_col).alias("id1"), F.col(cluster_col).alias("cluster1"))
        df_r2 = df_result.select(F.col(id_col).alias("id2"), F.col(cluster_col).alias("cluster2"))
        # Join với df_cl để bổ sung thông tin cluster cho id1 và id2, sau đó lọc ra các cặp vi phạm (cùng cluster)
        df_violations = (
            df_cl
            .join(df_r1, on="id1")
            .join(df_r2, on="id2")
            .filter(F.col("cluster1") == F.col("cluster2"))
            .select("id1", "id2", F.col("cluster1").alias("the_cluster"))
        )
        violation_count = df_violations.count()
        print(f"Number of cannot-link violations: {violation_count}")

        # Nếu không còn vi phạm, dừng vòng lặp
        if violation_count == 0:
            print("Không còn vi phạm cannot-link. Dừng vòng lặp.")
            break

        # --- Bước 1c: Tính khoảng cách từ điểm vi phạm đến các centroid của các cụm khác ---
        # 1. Lấy thông tin vector (features) của các điểm id2 dưới dạng array
        df_id2_features = df_result.select(
            F.col(id_col).alias("id2"),
            vector_to_array(F.col(features_col)).alias("feat2")
        )
        # 2. Join để lấy feat2 cho các điểm vi phạm từ df_violations
        df_v = df_violations.join(df_id2_features, on="id2", how="inner")
        # 3. Cross join với df_centroids để tính khoảng cách đến tất cả các cụm, loại trừ cụm hiện tại (the_cluster)
        df_candidates = (
            df_v.crossJoin(F.broadcast(df_centroids))
            .filter(F.col("cluster_id") != F.col("the_cluster"))
        )
        # 4. Chuyển centroid sang dạng array để tính khoảng cách
        df_candidates = df_candidates.withColumn("centroid_arr", vector_to_array(F.col("centroid")))
        # 5. Tính khoảng cách Euclid giữa feat2 và centroid_arr bằng Pandas UDF
        df_candidates = df_candidates.withColumn("dist_col", euclid_dist_pd(F.col("feat2"), F.col("centroid_arr")))
        # (Optional) df_candidates.show(5, truncate=False)

        # --- Bước 1d: Lựa chọn cụm ứng viên tốt nhất cho mỗi điểm vi phạm ---
        # Sử dụng Window function để nhóm theo id2 và sắp xếp theo khoảng cách tăng dần, chọn candidate có khoảng cách nhỏ nhất
        w = Window.partitionBy("id2").orderBy(F.col("dist_col").asc())
        df_best_candidates = (
            df_candidates.withColumn("rn", F.row_number().over(w))
            .filter(F.col("rn") == 1)
            .select("id2", F.col("cluster_id").alias("new_cluster"), "dist_col")
        )
        # (Optional) df_best_candidates.show(5, truncate=False)

        # --- Bước 1e: Cập nhật nhãn cụm của các điểm vi phạm (id2) trong df_result ---
        df_result = (
            df_result.alias("A")
            .join(df_best_candidates.alias("B"), F.col("A.id") == F.col("B.id2"), "left")
            .withColumn(
                cluster_col,
                F.when(F.col("B.new_cluster").isNotNull(), F.col("B.new_cluster"))
                 .otherwise(F.col("A." + cluster_col))
            )
            .select("A.id", "A.features", cluster_col)
        )
        print(f"[Iter {it}] Đã cập nhật nhãn cụm cho các điểm vi phạm.")

        # # --- Bước 1f: Kiểm tra tổng số vi phạm cannot-link sau cập nhật ---
        # df_r1_updated = df_result.select(F.col("id").alias("id1"), F.col(cluster_col).alias("cluster1"))
        # df_r2_updated = df_result.select(F.col("id").alias("id2"), F.col(cluster_col).alias("cluster2"))
        # total_violations = (
        #     df_cl
        #     .join(df_r1_updated, on="id1")
        #     .join(df_r2_updated, on="id2")
        #     .filter(F.col("cluster1") == F.col("cluster2"))
        #     .count()
        # )
        # print(f"[Iter {it}] Total violations after update: {total_violations}")

        # # Nếu số vi phạm không giảm, dừng vòng lặp
        # if prev_violations is not None and total_violations >= prev_violations:
        #     print(f"[Iter {it}] Số vi phạm không giảm so với vòng trước, dừng vòng lặp.")
        #     break
        # else:
        #     prev_violations = total_violations

    # Trả về df_result đã được cập nhật sau quá trình postprocessing
    return df_result



##########################
#  TỔNG HỢP: CHẠY PIPELINE
##########################
def constrained_kmeans(
    spark: SparkSession,
    df_data: DataFrame,
    df_ml: DataFrame,
    df_cl: DataFrame,
    k: int,
    id_col="id",
    features_col="features",
    handle_cannot_link=False
):
    """
    Gọi tuần tự:
      1) preprocess_must_link -> (df_superpoints, df_cc)
      2) run_kmeans_on_superpoints -> df_clustered_roots
      3) assign_clusters_back -> df_result
      4) (nếu handle_cannot_link) => postprocess_cannot_link
    Trả về df_result
    """

    # 1) Gom must-link
    start_time = time.time()
    df_superpoints, df_cc = preprocess_must_link(
        df_data, df_ml,
        id_col=id_col,
        features_col=features_col
    )
    end_time = time.time()
    print(f"Step 1 (preprocess_must_link) took {end_time - start_time} seconds")

    # 2) K-Means
    start_time = time.time()
    df_clustered_roots = run_kmeans_on_superpoints(df_superpoints, k)
    end_time = time.time()
    print(f"Step 2 (run_kmeans_on_superpoints) took {end_time - start_time} seconds")

    # 3) Gán cluster
    start_time = time.time()
    df_result = assign_clusters_back(
        df_cc, df_clustered_roots,
        df_data, id_col=id_col
    )
    end_time = time.time()
    print(f"Step 3 (assign_clusters_back) took {end_time - start_time} seconds")

    # ...

    return df_result
