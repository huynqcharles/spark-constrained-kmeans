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


def preprocess_must_link(
    df_data: DataFrame,
    df_ml: DataFrame,
    id_col: str = "id",
    features_col: str = "features"
):
    """
    Uses GraphFrame to compute connected components from must-link constraints,
    then calculates superpoints by averaging vectors within each component.
    
    Returns:
      - df_superpoints: DataFrame [root, features] containing averaged feature vectors
      - df_cc: DataFrame [id, root] mapping each id to its component root
    """
    # Create vertices from original data
    df_vertices = df_data.select(
        F.col(id_col).alias("id"),
        F.col(features_col).alias("features"))
    df_vertices.cache()
    
    # Create edges from must-link pairs (normalized to have smaller ID as src)
    df_edges = df_ml.select(
        F.least(F.col("id1"), F.col("id2")).alias("src"),
        F.greatest(F.col("id1"), F.col("id2")).alias("dst"))
    df_edges.cache()
    
    # Create GraphFrame and compute connected components
    gf = GraphFrame(df_vertices, df_edges)

    numPartitions = 40
    df_vertices = df_vertices.repartition(numPartitions, F.col("id"))
    df_edges = df_edges.repartition(numPartitions, F.col("src"))

    gf_cc = gf.connectedComponents()
    df_cc = gf_cc.select("id", F.col("component").alias("root"))
    
    # Join components with original features
    df_join = df_cc.join(df_vertices, on="id", how="left")
    df_join.cache()
    
    # Convert DenseVector to array
    df_join = df_join.withColumn("features_arr", vector_to_array("features"))
    sample = df_join.select("features_arr").limit(1).collect()
    
    # Get vector dimension
    if sample:
        dim = len(sample[0]["features_arr"])
    else:
        raise ValueError("No data to determine vector dimension")
    
    # Split array into separate dimension columns for optimized aggregation
    feat_columns = [F.col("features_arr")[i].alias(f"feat_{i}") for i in range(dim)]
    df_join = df_join.select("*", *feat_columns)
    _ = df_join.select([f"feat_{i}" for i in range(dim)]).limit(1).collect()

    # Group by component and compute average for each dimension
    df_join = df_join.repartition(numPartitions, F.col("root"))
    agg_exprs = [F.avg(f"feat_{i}").alias(f"avg_{i}") for i in range(dim)]
    df_grouped = df_join.groupBy("root").agg(*agg_exprs)
    
    # Combine averaged values back into array/vector
    avg_cols = [F.col(f"avg_{i}") for i in range(dim)]
    df_superpoints = df_grouped.withColumn("avg_features", F.array(*avg_cols)) \
                               .select(F.col("root"), F.col("avg_features").alias("features"))
    
    # Convert array back to DenseVector
    to_vector_udf = udf(lambda arr: DenseVector(arr), VectorUDT())
    df_superpoints = df_superpoints.withColumn("features", to_vector_udf("features"))

    return df_superpoints, df_cc

def run_kmeans_on_superpoints(
    df_superpoints: DataFrame,
    k: int,
    seed: int = 42,
    maxIter: int = 20,
    tol: float = 1e-4
):
    """
    Run KMeans on superpoints DataFrame and return cluster assignments.
    
    Args:
        df_superpoints: DataFrame with [root, features] columns
        k: Number of clusters
        seed: Random seed
        maxIter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        DataFrame with [root, cluster] columns after KMeans clustering
    """
    # Initialize and fit KMeans model
    kmeans = KMeans(k=k, seed=seed, featuresCol="features", predictionCol="cluster", 
                    maxIter=maxIter, tol=tol)
    model = kmeans.fit(df_superpoints)

    # Get cluster assignments
    df_clustered = model.transform(df_superpoints).select("root", "cluster")
    return df_clustered

def assign_clusters_back(
    df_cc: DataFrame,
    df_clustered_roots: DataFrame,
    df_data: DataFrame,
    id_col: str = "id"
) -> DataFrame:
    """
    Maps cluster assignments back to original data points.

    Args:
        df_cc: DataFrame with [id, root] mapping
        df_clustered_roots: DataFrame with [root, cluster] assignments  
        df_data: Original DataFrame
        id_col: Name of ID column
    
    Returns:
        DataFrame with original data plus "final_cluster" column
    """
    # Broadcast small DataFrames if needed
    df_cc = broadcast(df_cc)
    df_clustered_roots = broadcast(df_clustered_roots)

    # Join to get cluster assignments for each ID
    df_id_cluster = df_cc.join(df_clustered_roots, on="root", how="left")
    
    df_id_cluster = df_id_cluster.alias("idc")
    df_data_alias = df_data.alias("data")
    
    # Join back with original data
    df_final = df_data_alias.join(
        df_id_cluster,
        F.col("data." + id_col) == F.col("idc.id"), 
        how="left"
    )
    
    df_final = df_final.withColumn("final_cluster", F.col("idc.cluster"))
    
    # Clean up temporary columns
    df_final = df_final.drop(F.col("idc.id"))
    df_final = df_final.drop("root", "cluster")
    
    return df_final


def compute_centroids_rdd(df, cluster_col="final_cluster", features_col="features"):
    """
    Compute centroids for each cluster using distributed RDD operations.
    
    Args:
        df: DataFrame with cluster labels and feature vectors
        cluster_col: Name of cluster label column
        features_col: Name of feature vector column
    
    Returns:
        DataFrame with (cluster_id, centroid) for each cluster
    """
    spark = SparkSession.builder.getOrCreate()

    # Convert to RDD of (cluster, (features, 1)) pairs
    rdd = df.select(cluster_col, features_col).rdd.map(
        lambda row: (row[cluster_col], (row[features_col], 1))
    )

    # Reduce by key to sum vectors and counts per cluster
    def merge_values(a, b):
        vec_a, count_a = a
        vec_b, count_b = b
        sum_vec = DenseVector(np.add(vec_a.toArray(), vec_b.toArray()))
        return (sum_vec, count_a + count_b)
    
    aggregated = rdd.reduceByKey(merge_values)

    # Compute average vector (centroid) for each cluster
    def compute_average(item):
        cluster, (sum_vec, count) = item
        avg_array = sum_vec.toArray() / count
        return (cluster, DenseVector(avg_array))
    
    centroids_rdd = aggregated.map(compute_average)
    
    # Convert results to DataFrame
    schema = StructType([
        StructField("cluster_id", IntegerType(), True),
        StructField("centroid", VectorUDT(), True)
    ])
    
    centroids_df = spark.createDataFrame(centroids_rdd, schema=schema)
    return centroids_df

# Pandas UDF: Calculate Euclidean distance between vectors
@pandas_udf(DoubleType())
def euclid_dist_pd(feat_series: pd.Series, centroid_series: pd.Series) -> pd.Series:
    """
    Calculates Euclidean distance between pairs of vectors from two columns.
    Each element of feat_series and centroid_series is a list/array of floats.
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
    Post-process cannot-link constraints using distance-based heuristic.
    
    For each violated constraint pair that is in the same cluster,
    attempts to move one point to the nearest valid cluster.
    
    Args:
        df_result: DataFrame with clustering results
        df_cl: DataFrame with cannot-link constraints
        max_iter: Maximum iterations
        id_col: Name of ID column  
        features_col: Name of features column
        cluster_col: Name of cluster label column
        
    Returns:
        Updated df_result with less cannot-link violations
    """
    spark = df_result.sparkSession
    df_result = df_result.cache()

    for it in range(max_iter):
        print(f"\n[Iter {it}] Starting cannot-link postprocessing")

        # Compute centroids for each cluster
        df_centroids = compute_centroids_rdd(df_result, cluster_col, features_col)
        df_centroids = F.broadcast(df_centroids)
        df_centroids.cache()

        # Find violated cannot-link pairs (points in same cluster)
        df_r1 = df_result.select(F.col(id_col).alias("id1"), F.col(cluster_col).alias("cluster1"))
        df_r2 = df_result.select(F.col(id_col).alias("id2"), F.col(cluster_col).alias("cluster2"))
        df_violations = (
            df_cl
            .join(df_r1, on="id1")
            .join(df_r2, on="id2")
            .filter(F.col("cluster1") == F.col("cluster2"))
            .select("id1", "id2", F.col("cluster1").alias("the_cluster"))
        )
        violation_count = df_violations.count()
        print(f"Number of cannot-link violations: {violation_count}")

        if violation_count == 0:
            print("No more cannot-link violations. Stopping.")
            break

        # Get features vectors for violated points
        df_id2_features = df_result.select(
            F.col(id_col).alias("id2"),
            vector_to_array(F.col(features_col)).alias("feat2")
        )
        df_v = df_violations.join(df_id2_features, on="id2", how="inner")
        
        # Calculate distances to other cluster centroids
        df_candidates = (
            df_v.crossJoin(F.broadcast(df_centroids))
            .filter(F.col("cluster_id") != F.col("the_cluster"))
        )
        df_candidates = df_candidates.withColumn("centroid_arr", vector_to_array(F.col("centroid")))
        df_candidates = df_candidates.withColumn("dist_col", euclid_dist_pd(F.col("feat2"), F.col("centroid_arr")))

        # Select closest valid cluster for each violated point
        w = Window.partitionBy("id2").orderBy(F.col("dist_col").asc())
        df_best_candidates = (
            df_candidates.withColumn("rn", F.row_number().over(w))
            .filter(F.col("rn") == 1)
            .select("id2", F.col("cluster_id").alias("new_cluster"), "dist_col")
        )

        # Update cluster assignments
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
        print(f"[Iter {it}] Updated cluster assignments for violated points")

    return df_result



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
    Pipeline steps:
      1) Group must-linked points into superpoints
      2) Run k-means on superpoints 
      3) Map cluster assignments back to original points
      4) Optional: Handle cannot-link violations via post-processing
    """

    # 1) Process must-link constraints
    start_time = time.time()
    df_superpoints, df_cc = preprocess_must_link(
        df_data, df_ml,
        id_col=id_col,
        features_col=features_col
    )
    end_time = time.time()
    print(f"Step 1 (preprocess_must_link) took {end_time - start_time} seconds")

    # 2) Run k-means clustering
    start_time = time.time()
    df_clustered_roots = run_kmeans_on_superpoints(df_superpoints, k)
    end_time = time.time()
    print(f"Step 2 (run_kmeans_on_superpoints) took {end_time - start_time} seconds")

    # 3) Map clusters back to original points  
    start_time = time.time()
    df_result = assign_clusters_back(
        df_cc, df_clustered_roots,
        df_data, id_col=id_col
    )
    end_time = time.time()
    print(f"Step 3 (assign_clusters_back) took {end_time - start_time} seconds")

    # 4) Handle cannot-link constraints if requested
    if handle_cannot_link:
        start_time = time.time()
        
        df_violations = check_cannot_link_violations(
            df_result, 
            df_cl,
            id_col=id_col,
            cluster_col="final_cluster"
        )
        num_violations = df_violations.count()
        
        if num_violations > 0:
            print(f"Found {num_violations} cannot-link violations, attempting to fix...")
            
            df_result = postprocess_cannot_link(
                df_result,
                df_cl,
                max_iter=5,
                id_col=id_col,
                features_col=features_col,
                cluster_col="final_cluster"
            )
            
            # Check remaining violations
            df_violations_after = check_cannot_link_violations(
                df_result,
                df_cl,
                id_col=id_col,
                cluster_col="final_cluster"
            )
            num_violations_after = df_violations_after.count()
            
            print(f"After postprocessing: {num_violations_after} cannot-link violations remain")
            
        end_time = time.time()
        print(f"Step 4 (postprocess_cannot_link) took {end_time - start_time} seconds")

    # Validate final results
    validate_clustering_result(
        df_result,
        df_ml,
        df_cl,
        k,
        id_col=id_col,
        cluster_col="final_cluster"
    )

    return df_result


def validate_clustering_result(df_result, df_ml, df_cl, k, id_col="id", cluster_col="final_cluster"):
    """
    Validate final clustering results against constraints
    """
    # Check number of clusters
    num_clusters = df_result.select(cluster_col).distinct().count()
    assert num_clusters <= k, f"Found {num_clusters} clusters, expected <= {k}"
    
    # Check constraints
    ml_violations = check_must_link_violations(df_result, df_ml, id_col)
    num_ml_violations = ml_violations.count()
    
    cl_violations = check_cannot_link_violations(df_result, df_cl, id_col)
    num_cl_violations = cl_violations.count()
    
    print(f"\nValidation Results:")
    print(f"Number of clusters: {num_clusters}")
    print(f"Must-link violations: {num_ml_violations}")
    print(f"Cannot-link violations: {num_cl_violations}")

def check_cannot_link_violations(df_clustered, df_cl, id_col="id", cluster_col="final_cluster"):
    """
    Find pairs of points that violate cannot-link constraints
    (points that should not be in same cluster but are)
    """
    return df_cl.join(
        df_clustered.select(
            F.col(id_col).alias("id1"),
            F.col(cluster_col).alias("cluster1")
        ),
        "id1"
    ).join(
        df_clustered.select(
            F.col(id_col).alias("id2"),
            F.col(cluster_col).alias("cluster2")
        ),
        "id2"
    ).filter(F.col("cluster1") == F.col("cluster2"))

def check_must_link_violations(df_clustered, df_ml, id_col="id", cluster_col="final_cluster"):
    """
    Find pairs of points that violate must-link constraints
    (points that should be in same cluster but aren't)
    """
    return df_ml.join(
        df_clustered.select(
            F.col(id_col).alias("id1"),
            F.col(cluster_col).alias("cluster1")
        ),
        "id1"
    ).join(
        df_clustered.select(
            F.col(id_col).alias("id2"),
            F.col(cluster_col).alias("cluster2")
        ),
        "id2"
    ).filter(F.col("cluster1") != F.col("cluster2"))
