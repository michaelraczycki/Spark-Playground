import json

import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format, udf, window
from pyspark.sql.types import DoubleType

# Load the configuration file
with open('config.json') as config_file:
    config = json.load(config_file)

# Get the data path from the configuration
dataPath = config['data_path']

# Initialize Spark session
spark = SparkSession.builder.master("local").appName("sparkStreaming").getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", 5)

# Load data
staticDataFrame = spark.read.format("csv")\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .load(f"{dataPath}/retail-data/by-day/*.csv")

# Create temporary view and use schema
staticDataFrame.createOrReplaceTempView("retailData")
staticSchema = staticDataFrame.schema

# Perform select and group by operations
staticDataFrame.selectExpr("CustomerId", "(UnitPrice * Quantity) as totalCost", "InvoiceDate")\
    .groupBy(col("CustomerId"), window(col("InvoiceDate"), "1 day"))\
    .sum("totalCost")\
    .show(5)

# Print schema and prepare the data
staticDataFrame.printSchema()
preppedDataFrame = staticDataFrame.na.fill(0)\
    .withColumn("dayOfWeek", date_format(col("InvoiceDate"), "EEEE")).coalesce(5)
print(preppedDataFrame.take(2))

# Prepare the pipeline:

trainDataFrame = preppedDataFrame.where("InvoiceDate < '2011-07-01'")
testDataFrame = preppedDataFrame.where("InvoiceDate >= '2011-07-01'")

indexer = StringIndexer()\
    .setInputCol("dayOfWeek")\
    .setOutputCol("dayOfWeekIndex")

encoder = OneHotEncoder()\
    .setInputCol("dayOfWeekIndex")\
    .setOutputCol("dayOfWeekEncoded")

vectorAssembler = VectorAssembler()\
    .setInputCols(["UnitPrice", "Quantity", "dayOfWeekEncoded"])\
    .setOutputCol("features")

transformationPipeline = Pipeline()\
    .setStages([indexer, encoder, vectorAssembler])

fittedPipeline = transformationPipeline.fit(trainDataFrame)
transformedTraining = fittedPipeline.transform(trainDataFrame)

transformedTraining.cache()

kmeans = KMeans()\
    .setK(20)\
    .setSeed(1)

kmModel = kmeans.fit(transformedTraining)
print(kmModel.summary.trainingCost)
transformedTraining.unpersist()
transformedTest = fittedPipeline.transform(testDataFrame)

testPredictions = kmModel.transform(transformedTest)

# Compute test cost manually (as training cost is available only for the training set)
# Use the cluster centers from the trained model
centers = kmModel.clusterCenters()


# UDF to compute squared distance between a point and its assigned cluster center
def squared_distance(point, center):
    return float(np.sum((np.array(point) - np.array(center)) ** 2))


# Register UDF
squared_distance_udf = udf(lambda point, cluster_idx: squared_distance(point, centers[cluster_idx]), DoubleType())

# Add squared distance column to the test predictions
testPredictionsWithCost = testPredictions.withColumn(
    "squaredDistance",
    squared_distance_udf(col("features"), col("prediction"))
)

# Sum the squared distances to get the test set cost (WSSSE for test data)
testCost = testPredictionsWithCost.agg({"squaredDistance": "sum"}).collect()[0][0]

print(f"Test Set Sum of Squared Errors (WSSSE) = {testCost}")
