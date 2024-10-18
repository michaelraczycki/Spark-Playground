from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local").appName("demo").getOrCreate()

flight_data = (
    spark.read.option("inferSchema", "true")
    .option("header", "true")
    .csv("../Spark-The-Definitive-Guide/data/flight-data/csv/2015-summary.csv")
)
"""
print(flight_data.take(3))

flight_data.sort("count").explain()"""

spark.conf.set("spark.sql.shuffle.partitions", 5)
print(flight_data.sort("count").take(2))
