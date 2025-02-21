from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

def prepare_data():
    spark = SparkSession.builder.getOrCreate()

    # Read data from HDFS
    dataset_path = "hdfs://localhost:9000/data/data.csv"
    # dataset_path = "hdfs://localhost:9000/data/movement_libras_data.csv"
    dataset_path = "hdfs://localhost:9000/data/breast_cancer_data.csv"
    # dataset_path = "hdfs://localhost:9000/data/vehicle_data.csv"
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
    constraints_path = "hdfs://localhost:9000/data/constraints.json"
    # constraints_path = "hdfs://localhost:9000/data/movement_libras_constraints_0.2.json"
    constraints_path = "hdfs://localhost:9000/data/breast_cancer_constraints_0.2.json"
    # constraints_path = "hdfs://localhost:9000/data/vehicle_constraints_0.2.json"
    df_json = spark.read.json(constraints_path)

    df_ml = df_json.selectExpr(
        "inline(transform(ml, x -> struct(x[0] as id1, x[1] as id2)))"
    )

    df_cl = df_json.selectExpr(
        "inline(transform(cl, x -> struct(x[0] as id1, x[1] as id2)))"
    )

    return df_data, df_class, df_ml, df_cl