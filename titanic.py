from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName('Titanic Data') \
    .getOrCreate()


df = (spark.read.format("csv").option('header', 'true').load("arquivos/titanic.csv"))

from pyspark.sql.functions import col
dataset = df.select(col('Survived').cast('float'),
                         col('Pclass').cast('float'),
                         col('Sex'),
                         col('Age').cast('float'),
                         col('Fare').cast('float'),
                         col('Embarked')
                        )

dataset = dataset.replace('null', None).dropna(how='any')

from pyspark.ml.feature import StringIndexer
dataset = StringIndexer(inputCol='Sex', outputCol='Gender',handleInvalid='keep').fit(dataset).transform(dataset)
dataset = StringIndexer(inputCol='Embarked',outputCol='Boarded',handleInvalid='keep').fit(dataset).transform(dataset)
dataset = dataset.drop('Sex','Embarked')


required_features = ['Pclass','Age','Fare','Gender','Boarded']

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=required_features, outputCol='features')
transformed_data = assembler.transform(dataset)

(training_data, test_data) = transformed_data.randomSplit([0.8,0.2])

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol='Survived',featuresCol='features',maxDepth=3)
model = rf.fit(training_data)

predictions = model.transform(test_data)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol='Survived',predictionCol='prediction',metricName='accuracy')


accuracy = evaluator.evaluate(predictions)
print('Test Accuracy = ', accuracy)
