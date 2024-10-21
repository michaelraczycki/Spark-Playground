import json

from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import col, column, expr
from pyspark.sql.types import StringType, StructField, StructType

# Load the configuration file
with open('config.json') as config_file:
    config = json.load(config_file)

# Get the data path from the configuration
dataPath = config['data_path']

spark = SparkSession.builder.master("local").appName("spark_operations").getOrCreate()

df = spark.read.format("json").load(f"{dataPath}/flight-data/json/2015-summary.json")
df.printSchema()

print(spark.read.format("json").load(f"{dataPath}/flight-data/json/2015-summary.json").schema)

# example of manually setting schema
myManualSchema = StructType([
    StructField("DEST_COUNTRY_NAME", StringType(), True),
    StructField("ORIGIN_COUNTRY_NAME", StringType(), False, metadata={"hello": "world"})
])
df = spark.read.format("json").schema(myManualSchema).load(f"{dataPath}/flight-data/json/2015-summary.json")
df.printSchema()

# Row creation:
myRow = Row("Hello", None, 1, False)  # rows do have a schema

print(myRow[0])
print(myRow[2])


df = spark.read.format("json").load(f"{dataPath}/flight-data/json/2015-summary.json")
df.createOrReplaceTempView("dfTable")

print(df.select("DEST_COUNTRY_NAME").show(2))

print(df.select("DEST_COUNTRY_NAME", "ORIGIN_COUNTRY_NAME").show(2))

# There are different ways to refer to column, stick to one:

df.select(
    expr("DEST_COUNTRY_NAME"),
    col("DEST_COUNTRY_NAME"),
    column("DEST_COUNTRY_NAME")
).show(2)

# When selecting, you can either use column objects in select or strings, even mix and match (but not recommended for clarity)
df.select(col("DEST_COUNTRY_NAME"), "DEST_COUNTRY_NAME").show(2)

# if you want to select result of an expression, simply use selectExpr instead of select(expr(<someexpression>)):
df.selectExpr("DEST_COUNTRY_NAME as newColumnName", "DEST_COUNTRY_NAME").show(2)

# this can be also used for creating new columns based on column values:
df.selectExpr("*", "(DEST_COUNTRY_NAME = ORIGIN_COUNTRY_NAME) as withinCountry")\
    .show(2)

# or aggregations:
df.selectExpr("avg(count)", "count(distinct(ORIGIN_COUNTRY_NAME)) as num_of_origins").show(2)
