import time

from pyspark.sql import SparkSession

from constrained_kmeans_test import constrained_kmeans
from data_processor import prepare_data
from sklearn.metrics import adjusted_rand_score

# Create SparkSession
spark = SparkSession.builder \
    .appName("SparkConstrainedKMeans") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setCheckpointDir("hdfs://localhost:9000/data/checkpoint_dir")

# Prepare data, read data from HDFS
df_data, df_class, df_ml, df_cl = prepare_data()

# Count occurrences per class
class_counts = df_class.groupBy("class").count().orderBy("class")
class_counts.show()

# Get number of unique classes for k
k = df_class.select("class").distinct().count()

start_time = time.time()

# Call constrained kmeans pipeline
# handle_cannot_link = False (cannot-link not handled yet)
df_result = constrained_kmeans(
    df_data,
    df_ml,
    df_cl,
    k=k,
    id_col="id",
    features_col="features",
    verbose=True
)

end_time = time.time()
print(f"Time taken to run constrained kmeans: {end_time - start_time} seconds")

# df_result.show()

# Join the result with the original class labels
df_result = df_result.join(df_class, on="id")

# Collect the results
results = df_result.select("class", "final_cluster").collect()

# Extract the true labels and predicted labels
true_labels = [row["class"] for row in results]
predicted_labels = [row["final_cluster"] for row in results]

# Calculate the Adjusted Rand Index (ARI)
ari = adjusted_rand_score(true_labels, predicted_labels)
print(f"Adjusted Rand Index (ARI): {ari}")

spark.stop()
