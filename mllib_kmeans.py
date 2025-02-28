import time
from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
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

# Khởi tạo KMeans với thuật toán k-means|| (mặc định của Spark)
kmeans = KMeans(k=k, initMode="k-means||", maxIter=10, seed=42)
model = kmeans.fit(df_data)

end_time = time.time()
print(f"Time taken to mllib kmeans: {end_time - start_time} seconds")

# Lấy dự đoán
predictions = model.transform(df_data)

# Giả sử DataFrame df_class có cột "id" chung với df_class để thực hiện join
pred_with_label = predictions.join(df_class, on="id")

# Thu thập nhãn dự đoán và nhãn thực về driver để tính ARI
preds = [row.prediction for row in pred_with_label.select("prediction").collect()]
true_labels = [row["class"] for row in pred_with_label.select("class").collect()]

# Tính ARI bằng hàm adjusted_rand_score từ scikit-learn
ari = adjusted_rand_score(true_labels, preds)
print("Adjusted Rand Index (ARI):", ari)