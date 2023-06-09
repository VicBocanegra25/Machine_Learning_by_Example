"""Chapter 6: Using Apache Spark to load the entire Ad Click-Through dataset
(40 millions)

1.- In the one-hot encoding solution, can you use different
classifiers supported in PySpark instead of logistic regression, such as
decision trees, random forests, or linear SVM?

# Allocating memory and all processing power. This needs to be run in a terminal
pyspark --master local[*]  --driver-memory 30G

# Building an app in spark
spark = SparkSession.builder.appName("CTR").getOrCreate()

# Importing the pyspark types
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

# Limiting the number of rows:
n_rows = 5_000_000
# Loading the dataframe
df = spark.read.csv("C:/Users/52556/Documents_outside_ODB/Python_Excercises/Machine_Learning/Machine_Learning_by_Example/dataset/train.csv", schema=schema, header=True)

# Dropping columns that provide little information
df = df.drop('id').drop('hour').drop('device_id').drop('device_ip')

# We rename the column from click to label, as this will be consumed more often in the downstream operations:
df = df.withColumnRenamed("click", "label")

# Let's look at the current columns in the DataFrame object:
# df.columns['label', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']

# Splitting and caching the data
df_train, df_test, _ = df.randomSplit([0.175, 0.075, 0.75], 42)

# Applying caching and persistence so we don't have to recalculate the dataframe again (if required)
df_train.cache()
# DataFrame[label: int, C1: string, banner_pos: string, site_id: string, site_domain: string, site_category: string, app_id: string, app_domain: string, app_category: string, device_model: string, device_type: string, device_conn_type: string, C14: string, C15: string, C16: string, C17: string, C18: string, C19: string, C20: string, C21: string]

df_test.cache()
# DataFrame[label: int, C1: string, banner_pos: string, site_id: string, site_domain: string, site_category: string, app_id: string, app_domain: string, app_category: string, device_model: string, device_type: string, device_conn_type: string, C14: string, C15: string, C16: string, C17: string, C18: string, C19: string, C20: string, C21: string]

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
stages = indexers + [encoder, assembler]
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

# We'll now train our model for Decision Trees, Random Trees and linear SVM models:
# https://spark.apache.org/docs/latest/ml-classification-regression.html#classification

# A) We first import the Decision tree classifier module and initialize a model:
from pyspark.ml.classification import DecisionTreeClassifier
classifier = DecisionTreeClassifier(labelCol = "label", featuresCol = "features")

# Now, we fit the model on the encoded training set:
tree_model = classifier.fit(df_train_encoded)

# After all iterations, we apply the trained model on the testing set:
predictions = tree_model.transform(df_test_encoded)

# We cache the prediction results, as we will compute the prediction's performance:
predictions.cache()

# We evaluate the Area Under Curve (AUC) of the Receiver Operating Characteristics (ROC) on the testing set using the BinaryClassificationEvaluator function with the areaUnderROC evaluation metric:
from pyspark.ml.evaluation import BinaryClassificationEvaluator
ev = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction", metricName = "areaUnderROC")
print(ev.evaluate(predictions))
predictions.unpersist()

# B) Trying with Random Forest Trees:
from pyspark.ml.classification import RandomForestClassifier
classifier_rf = RandomForestClassifier(labelCol = "label", featuresCol =
"features", numTrees = 10)

# Training the model on the encoded training set
random_forest_model = classifier_rf.fit(df_train_encoded)

# Applying the trained model on the testing set
predictions_rf = model.transform(df_test_encoded)

# Cache prediction results as we'll compute the prediction's performance:
predictions_rf.cache()

# We evaluate the Area Under Curve (AUC) of the Receiver Operating Characteristics (ROC) on the testing set using the BinaryClassificationEvaluator function with the areaUnderROC evaluation metric:
from pyspark.ml.evaluation import BinaryClassificationEvaluator
ev = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction", metricName = "areaUnderROC")
print(ev.evaluate(predictions_rf))
predictions_rf.unpersist()

# C) Using Linear SVM to process the dataset
from pyspark.ml.classification import LinearSVC

classifier_lsvc = LinearSVC(maxIter = 10, regParam = 0.1)
model_lsvc = classifier_lsvc.fit(df_train_encoded)

# Applying the trained svc model on the testing set
predictions_svc = model_lsvc.transform(df_test_encoded)

# Cache prediction results as we'll compute the prediction's performance:
predictions_svc.cache()

# We evaluate the Area Under Curve (AUC) of the Receiver Operating Characteristics (ROC) on the testing set using the BinaryClassificationEvaluator function with the areaUnderROC evaluation metric:
from pyspark.ml.evaluation import BinaryClassificationEvaluator
ev = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction", metricName = "areaUnderROC")
print(ev.evaluate(predictions_svc))
predictions_svc2.unpersist()


# Performing feature hashing. Importing the module and initializing a feature hasher with an output size of 10_000
from pyspark.ml.feature import FeatureHasher
hasher = FeatureHasher(numFeatures = 10_000, inputCols = categorical,
outputCol = "features")

# We use the defined hasher to convert the input DataFrame:
hasher.transform(df_train).select("features").show()

# For better organization of the entire workflow, we chain the hasher and classification model together into a pipeline:
classifier = LogisticRegression(maxIter = 20, regParam = 0.001,
elasticNetParam = 0.001)
stages = [hasher, classifier]
pipeline = Pipeline(stages = stages)

# We fit the pipelined model on the training set as follows:
model = pipeline.fit(df_train)

# We apply the trained model on the testing set and record the prediction results:
predictions = model.transform(df_test)
predictions.cache()

# We apply the trained model on the testing set and record the prediction results:
ev = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction",
metricName = "areaUnderROC")
print(ev.evaluate(predictions))

# Performing feature interaction
from pyspark.ml.feature import RFormula

# We need to define an interaction formula accordingly:
cat_inter = ["C14", "C15"]
cat_no_inter = [c for c in categorical if c not in cat_inter]
concat = "+".join(categorical)
interaction = ":".join(cat_inter)
formula = "label ~ " + concat + "+" + interaction
print(formula)

# Now, we can initialize a feature interactor with this formula:
interactor = RFormula(
	formula = formula,
	featuresCol = "features",
	labelCol="label").setHandleInvalid("keep") # Making sure it won't crash if new values occurs

# We use the defined feature interactor to fit and transform the input DataFrame:
interactor.fit(df_train).transform(df_train).select("features").show()


# Again, we chain the feature interactor and classification model together into a pipeline to organize the entire workflow:
classifier = LogisticRegression(maxIter = 20, regParam = 0.001,
elasticNetParam = 0.001)

stages = [interactor, classifier]
pipeline = Pipeline(stages = stages)
model = pipeline.fit(df_train)

predictions = model.transform(df_test)
predictions.cache()

ev = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction",
metricName = "areaUnderROC")
print(ev.evaluate(predictions))
"""