from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

def prepare_data():
    spark = SparkSession.builder.getOrCreate()

    # Read data from HDFS
    # SMALL
    # dataset_path = "hdfs://localhost:9000/data/small/movement_libras_data.csv"
    # dataset_path = "hdfs://localhost:9000/data/small/breast_cancer_data.csv"
    # dataset_path = "hdfs://localhost:9000/data/small/vehicle_data.csv"
    # dataset_path = "hdfs://localhost:9000/data/small/sonar_data.csv"
    dataset_path = "hdfs://localhost:9000/data/small/spectfheart_data.csv"

    # LARGE
    # dataset_path = "hdfs://localhost:9000/data/large/mnist_data.csv"
    # dataset_path = "hdfs://localhost:9000/data/large/shuttle_data.csv"

    df_data = spark.read.csv(dataset_path, header=True, inferSchema=True)

    # Create id column
    df_data = df_data.withColumn("id", F.monotonically_increasing_id())

    # Create "features" column
    ignore_cols = ["class", "id"]
    feature_cols = [col for col in df_data.columns if col not in ignore_cols]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_class = df_data.select("id", "class")
    df_data = assembler.transform(df_data).select("id", "features")

    # Read constraints from HDFS
    # SMALL
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/movement_libras_constraints_0.2.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/breast_cancer_constraints_0.2.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/vehicle_constraints_0.2.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/sonar_constraints_0.2.json"
    constraints_path = "hdfs://localhost:9000/data/small/constraint sets/spectfheart_constraints_0.2.json"

    # LARGE
    # constraints_path = "hdfs://localhost:9000/data/large/constraint sets/mnist_constraints_0.005.json"
    # constraints_path = "hdfs://localhost:9000/data/large/constraint sets/shuttle_constraints_0.01.json"

    df_json = spark.read.json(constraints_path)

    df_ml = df_json.selectExpr(
        "inline(transform(ml, x -> struct(x[0] as id1, x[1] as id2)))"
    )

    df_cl = df_json.selectExpr(
        "inline(transform(cl, x -> struct(x[0] as id1, x[1] as id2)))"
    )

    return df_data, df_class, df_ml, df_cl