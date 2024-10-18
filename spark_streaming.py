from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format, window

spark = SparkSession.builder.master("local").appName("spark-streaming").getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", 5)


staticDataFrame = (
    spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("../Spark-The-Definitive-Guide/data/retail-data/by-day/*.csv")
)

staticDataFrame.createOrReplaceTempView("retail_data")
staticSchema = staticDataFrame.schema

staticDataFrame.selectExpr(
    "CustomerId", "(UnitPrice * Quantity) as total_cost", "InvoiceDate"
).groupBy(col("CustomerId"), window(col("InvoiceDate"), "1 day")).sum(
    "total_cost"
).show(
    5
)

staticDataFrame.printSchema()
prepped_data_frame = (
    staticDataFrame.na.fill(0)
    .withColumn("day_of_week", date_format(col("InvoiceDate"), "EEEE"))
    .coalesce(5)
)
print(prepped_data_frame.take(2))

# prepare the pipeline:

trainDataFrame = prepped_data_frame.where("InvoiceDate < '2011-07-01'")
testDataFrame = prepped_data_frame.where("InvoiceDate >= '2011-07-01'")

indexer = StringIndexer().setInputCol("day_of_week").setOutputCol("day_of_week_index")

encoder = (
    OneHotEncoder().setInputCol("day_of_week_index").setOutputCol("day_of_week_encoded")
)


vectorAssembler = (
    VectorAssembler()
    .setInputCols(["UnitPrice", "Quantity", "day_of_week_encoded"])
    .setOutputCol("features")
)

transformationPipeline = Pipeline().setStages([indexer, encoder, vectorAssembler])

fittedPipeline = transformationPipeline.fit(trainDataFrame)
transformedTraining = fittedPipeline.transform(trainDataFrame)

transformedTraining.cache()
