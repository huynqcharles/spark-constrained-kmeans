from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

def prepare_data():
    spark = SparkSession.builder.getOrCreate()

    # Read data from HDFS
    # dataset_path = "hdfs://localhost:9000/data/small/breast_cancer_data.csv"

    dataset_path = "hdfs://localhost:9000/data/small/ecoli_data.csv"

    # dataset_path = "hdfs://localhost:9000/data/small/ionosphere_data.csv"

    # dataset_path = "hdfs://localhost:9000/data/small/monk-2_data.csv"

    # dataset_path = "hdfs://localhost:9000/data/small/moons_data.csv"

    # dataset_path = "hdfs://localhost:9000/data/small/newthyroid_data.csv"

    # dataset_path = "hdfs://localhost:9000/data/small/sonar_data.csv"

    # dataset_path = "hdfs://localhost:9000/data/small/spectfheart_data.csv"

    # dataset_path = "hdfs://localhost:9000/data/small/spiral_data.csv"

    # dataset_path = "hdfs://localhost:9000/data/small/vehicle_data.csv"

    df_data = spark.read.csv(dataset_path, header=True, inferSchema=True)

    # Print dataset information
    num_instances = df_data.count()
    num_attributes = len(df_data.columns) - 1
    num_classes = df_data.select('class').distinct().count()
    print(f"Number of instances: {num_instances}")
    print(f"Number of attributes: {num_attributes}")
    print(f"Number of classes: {num_classes}")

    # Create id column
    df_data = df_data.withColumn("id", F.monotonically_increasing_id())

    # Create "features" column
    ignore_cols = ["class", "id"]
    feature_cols = [col for col in df_data.columns if col not in ignore_cols]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_class = df_data.select("id", "class")
    df_data = assembler.transform(df_data).select("id", "features")

    # Read constraints from HDFS
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/breast_cancer_constraints_0.05.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/breast_cancer_constraints_0.1.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/breast_cancer_constraints_0.15.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/breast_cancer_constraints_0.2.json"

    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/ecoli_constraints_0.05.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/ecoli_constraints_0.1.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/ecoli_constraints_0.15.json"
    constraints_path = "hdfs://localhost:9000/data/small/constraint sets/ecoli_constraints_0.2.json"

    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/ionosphere_constraints_0.05.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/ionosphere_constraints_0.1.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/ionosphere_constraints_0.15.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/ionosphere_constraints_0.2.json"

    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/monk-2_constraints_0.05.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/monk-2_constraints_0.1.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/monk-2_constraints_0.15.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/monk-2_constraints_0.2.json"

    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/moons_constraints_0.05.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/moons_constraints_0.1.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/moons_constraints_0.15.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/moons_constraints_0.2.json"

    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/newthyroid_constraints_0.05.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/newthyroid_constraints_0.1.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/newthyroid_constraints_0.15.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/newthyroid_constraints_0.2.json"

    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/sonar_constraints_0.05.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/sonar_constraints_0.1.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/sonar_constraints_0.15.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/sonar_constraints_0.2.json"

    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/spectfheart_constraints_0.05.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/spectfheart_constraints_0.1.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/spectfheart_constraints_0.15.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/spectfheart_constraints_0.2.json"

    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/spiral_constraints_0.05.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/spiral_constraints_0.1.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/spiral_constraints_0.15.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/spiral_constraints_0.2.json"

    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/vehicle_constraints_0.05.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/vehicle_constraints_0.1.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/vehicle_constraints_0.15.json"
    # constraints_path = "hdfs://localhost:9000/data/small/constraint sets/vehicle_constraints_0.2.json"

    df_json = spark.read.json(constraints_path)

    df_ml = df_json.selectExpr(
        "inline(transform(ml, x -> struct(x[0] as id1, x[1] as id2)))"
    )

    df_cl = df_json.selectExpr(
        "inline(transform(cl, x -> struct(x[0] as id1, x[1] as id2)))"
    )



    return df_data, df_class, df_ml, df_cl