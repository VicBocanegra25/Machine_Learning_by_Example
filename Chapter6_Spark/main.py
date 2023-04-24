"""
# Allocating memory and all processing power
./bin/pyspark --master local[*]  --driver-memory 20G

# Building an app in spark
spark = SparkSession.builder.appName("CTR").getOrCreate()

from pyspark.sql.types import StructField, StringType, StructType, IntegerType

# Creating a schema for the dataframe
schema = StructType([
    StructField("id", StringType(), True),
    StructField("click", IntegerType(), True),
    StructField("hour", IntegerType(), True),
    StructField("C1", StringType(), True),
    StructField("banner_pos", StringType(), True),
    StructField("site_id", StringType(), True),
    StructField("site_domain", StringType(), True),
    StructField("site_category", StringType(), True),
    StructField("app_id", StringType(), True),
    StructField("app_domain", StringType(), True),
    StructField("app_category", StringType(), True),
    StructField("device_id", StringType(), True),
    StructField("device_ip", StringType(), True),
    StructField("device_model", StringType(), True),
    StructField("device_type", StringType(), True),
    StructField("device_conn_type", StringType(), True),
    StructField("C14", StringType(), True),
    StructField("C15", StringType(), True),
    StructField("C16", StringType(), True),
    StructField("C17", StringType(), True),
    StructField("C18", StringType(), True),
    StructField("C19", StringType(), True),
    StructField("C20", StringType(), True),
    StructField("C21", StringType(), True),
])

# Loading the dataframe
df = spark.read.csv("/home/vicbocanegra/PycharmProjects/MLExample/Packt/Machine_Learning_by_Example/Chapter6_Spark/dataset/train.csv", schema=schema, header=True)

# Dropping columns that provide little information
df = df.drop('id').drop('hour').drop('device_id').drop('device_ip')

# We rename the column from click to label, as this will be consumed more often in the downstream operations:
df = df.withColumnRenamed("click", "label")

# Let's look at the current columns in the DataFrame object:
df.columns['label', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']


# Splitting and caching the data
df_train, df_test = df.randomSplit([0.7, 0.3], 42)

# Applying caching and persistence so we don't have to recalculate the dataframe again (if required)
df_train.cache()
DataFrame[label: int, C1: string, banner_pos: string, site_id: string, site_domain: string, site_category: string, app_id: string, app_domain: string, app_category: string, device_model: string, device_type: string, device_conn_type: string, C14: string, C15: string, C16: string, C17: string, C18: string, C19: string, C20: string, C21: string]

df_test.cache()
DataFrame[label: int, C1: string, banner_pos: string, site_id: string, site_domain: string, site_category: string, app_id: string, app_domain: string, app_category: string, device_model: string, device_type: string, device_conn_type: string, C14: string, C15: string, C16: string, C17: string, C18: string, C19: string, C20: string, C21: string]

# One-hot encoding categorical features
categorical = df_train.columns
categorical.remove('label')

# In PySpark, one-hot encoding is not as direct as it is in scikit-learn (specifically, with the OneHotEncoder module).
# We need to index each categorical column using the StringIndexer module:
from pyspark.ml.feature import StringIndexer
indexers = [StringIndexer(inputCol=c, outputCol = "{0}_indexed".format(c)).setHandleInvalid("keep") for c in categorical]

#Then, we perform one-hot encoding on each individual indexed categorical column using the OneHotEncoderEstimator module:
from pyspark.ml.feature import OneHotEncoder
encoder = OneHotEncoder(
    inputCols = [indexer.getOutputCol() for indexer in indexers],
    outputCols = ["{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers])

# Next, we concatenate all sets of generated binary vectors into a single one using the VectorAssembler module:
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(
    inputCols = encoder.getOutputCols(),
    outputCol = "features")

# We chain all these three stages together into a pipeline with the Pipeline module in PySpark, which better organizes our one-hot encoding workflow:
stages = indexers+ [encoder, assembler]
from pyspark.ml import Pipeline
pipeline = Pipeline(stages = stages)
# The variable stages is a list of operations needed for encoding.

# Finally, we can fit the pipeline one-hot encoding model over the training set:
one_hot_encoder = pipeline.fit(df_train)

# Once this is done, we use the trained encoder to transform both the training and testing sets. For the training set, we use the following code:
df_train_encoded = one_hot_encoder.transform(df_train)
df_train_encoded.show()

# We only select the features column, which contains the one-hot encoded results.
df_train_encoded = df_train_encoded.select(["label", "features"])

# Don't forget to cache df_train_encoded, as we will be using it to iteratively train our classification model:
df_train_encoded.cache()

# To release some space, we uncache df_train, since we will no longer need it:
df_train.unpersist()

# Now, we repeat the preceding steps for the testing set:
df_test_encoded = one_hot_encoder.transform(df_test)
df_test_encoded = df_test_encoded.select(["label", "features"])
df_test_encoded.cache()
df_test.unpersist()

# We'll now train our model for Logistic regression, but there are several other models:
# https://spark.apache.org/docs/latest/ml-classification-regression.html#classification

# We first import the logistic regression module and initialize a model:
from pyspark.ml.classification import LogisticRegression
classifier = LogisticRegression(maxIter = 20, regParam = 0.001, elasticNetParam = 0.001)

# Now, we fit the model on the encoded training set:
lr_model = classifier.fit(df_train_encoded)

# After all iterations, we apply the trained model on the testing set:
predictions = lr_model.transform(df_test_encoded)

# We cache the prediction results, as we will compute the prediction's performance:
predictions.cache()

# We evaluate the Area Under Curve (AUC) of the Receiver Operating Characteristics (ROC) on the testing set using the BinaryClassificationEvaluator function with the areaUnderROC evaluation metric:
from pyspark.ml.evaluation import BinaryClassificationEvaluator
ev = BinaryClassificationEvaluator(rawPredictionCol = "rawPRediction", metricName = "areaUnderROC")
print(ev.evaluate(predictions))



"""