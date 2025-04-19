import time
import warnings
import numpy as np
import pandas as pd

from graphframes import GraphFrame
from pyspark.ml.clustering import KMeans
from pyspark.ml.functions import vector_to_array
from pyspark.ml.stat import Summarizer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
import pyspark.sql.functions as F

warnings.filterwarnings("ignore", message="DataFrame.sql_ctx is an internal property, and will be removed in future releases")
warnings.filterwarnings("ignore", message="DataFrame constructor is internal. Do not directly use it.")

def preprocess_must_link(
    df_data: DataFrame,
    df_ml:   DataFrame,
    id_col:       str = "id",
    features_col: str = "features"
) -> tuple[DataFrame, DataFrame]:
    
    spark = SparkSession.builder.getOrCreate()
    # Create vertices & edges
    df_vert = df_data.select(
        F.col(id_col).alias("id"),
        F.col(features_col).alias("features")
    ).cache()
    df_edge = df_ml.select(
        F.least("id1","id2").alias("src"), 
        F.greatest("id1","id2").alias("dst")
    ).cache()

    # Dynamic repartitioning
    npart = spark.sparkContext.defaultParallelism * 2
    df_vert = df_vert.repartition(npart, "id")
    df_edge = df_edge.repartition(npart, "src")

    # Find connected components
    gf_cc = GraphFrame(df_vert, df_edge).connectedComponents()
    df_cc = gf_cc.select(
        F.col("id"),
        F.col("component").alias("root")
    )

    # Join IDs with features and compute mean vector for each component
    df_join = (
      df_cc
      .join(df_vert, on="id", how="inner")
      .repartition(npart, "root")
      .select("root", "features")
    )

    # Use Summarizer to compute centroids (means) for each group
    df_super = (
      df_join
      .groupBy("root")
      .agg(
         Summarizer.mean(F.col("features")).alias("features")
      )
    )

    return df_super, df_cc

def initialize_cluster_on_superpoints(
    df_superpoints: DataFrame,
    k: int,
    seed: int = 42,
    maxIter: int = 20,
    tol: float = 1e-4
):
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
    1) Join df_cc (id → root) with df_clustered_roots (root → cluster) 
    2) Select (id, final_cluster) and broadcast
    3) Direct join with df_data on id
    """

    # Create table mapping id → final_cluster
    df_id_cluster = (
        df_cc
        .join(df_clustered_roots, on="root", how="left")
        .select(
            F.col(id_col).alias("id"), 
            F.col("cluster").alias("final_cluster")
        )
    ).cache()

    # Broadcast id_cluster table
    df_id_cluster = F.broadcast(df_id_cluster)

    # Join with original data
    df_final = (
        df_data
        .join(df_id_cluster, on=id_col, how="left")
    )

    return df_final

def compute_centroids_df(
    df: DataFrame,
    cluster_col: str = "final_cluster",
    features_col: str = "features"
) -> DataFrame:
    # Use Summarizer to get mean vector per cluster
    mean_metric = Summarizer.mean(F.col(features_col))
    df_centroids = (
        df.groupBy(cluster_col)
          .agg(mean_metric.alias("centroid"))
          .withColumnRenamed(cluster_col, "cluster_id")
    )
    return df_centroids

# Pandas UDF: Calculate Euclidean distance between vectors
@pandas_udf(DoubleType())
def euclid_dist_pd(feat_series: pd.Series, centroid_series: pd.Series) -> pd.Series:
    return feat_series.combine(centroid_series, lambda f, c: float(np.linalg.norm(np.array(f) - np.array(c))))

def postprocess_cannot_link(
    df_result,
    df_cl,
    max_iter=5,
    id_col="id",
    features_col="features", 
    cluster_col="final_cluster"
):
    for it in range(max_iter):
        # Compute centroids for each cluster
        df_centroids = compute_centroids_df(
            df_result,
            cluster_col=cluster_col,
            features_col=features_col
        )
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
        df_violations = df_violations.checkpoint()
        violation_count = df_violations.count()

        if violation_count == 0:
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

        df_candidates = df_candidates.checkpoint()
        df_candidates.cache()

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

        df_result = df_result.checkpoint()
        df_result.cache()

    return df_result

def constrained_kmeans(
    df_data: DataFrame,
    df_ml: DataFrame,
    df_cl: DataFrame,
    k: int,
    id_col: str = "id",
    features_col: str = "features",
    verbose: bool = False
) -> DataFrame:
    
    spark = SparkSession.builder.getOrCreate()
    npart = spark.sparkContext.defaultParallelism * 2

    # cache inputs
    df_data = df_data.cache()
    df_ml   = df_ml.cache()
    df_cl   = df_cl.cache()

    try:
        # Step 1: preprocess must‑link
        t0 = time.time()
        df_superpoints, df_cc = preprocess_must_link(
            df_data, df_ml, id_col=id_col, features_col=features_col
        )
        df_superpoints = df_superpoints \
            .repartition(npart, F.col("root")) \
            .cache()
        df_cc = df_cc \
            .repartition(npart, F.col("id")) \
            .cache()
        # checkpoint to cut long DAG
        df_superpoints = df_superpoints.checkpoint()
        if verbose:
            print(f"[Step1] preprocess_must_link: {time.time()-t0:.2f}s")

        # Step 2: initialize on superpoints
        t1 = time.time()
        df_clustered_roots = initialize_cluster_on_superpoints(df_superpoints, k)
        df_clustered_roots = df_clustered_roots.cache()
        if verbose:
            print(f"[Step2] initialize_cluster: {time.time()-t1:.2f}s")

        # Step 3: assign back to original points
        t2 = time.time()
        df_result = assign_clusters_back(
            df_cc, df_clustered_roots, df_data, id_col=id_col
        ).cache()
        # checkpoint once before postprocess
        df_result = df_result.checkpoint()
        if verbose:
            print(f"[Step3] assign_clusters_back: {time.time()-t2:.2f}s")

        # Step 4: postprocess cannot‑link
        t3 = time.time()
        df_result = postprocess_cannot_link(
            df_result,
            df_cl,
            max_iter=5,
            id_col=id_col,
            features_col=features_col,
            cluster_col="final_cluster"
        ).cache()
        # final checkpoint
        df_result = df_result.checkpoint()
        if verbose:
            print(f"[Step4] postprocess_cannot_link: {time.time()-t3:.2f}s")

        return df_result

    finally:
        # clean up caches
        for df in (df_data, df_ml, df_cl,
                   df_superpoints, df_cc, df_clustered_roots, df_result):
            try:
                df.unpersist()
            except:
                pass