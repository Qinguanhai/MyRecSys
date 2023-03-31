from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml.feature import  OneHotEncoder, StringIndexer, QuantileDiscretizer, MinMaxScaler
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.ml import Pipeline

def read_movie_data(spark, file_path):
    # 读取csv文件为dataframe
    movie_raw_data = spark.read.format('csv').option('header', 'true').load(file_path)

    # 展示读取到原始数据
    print("Raw Movie Samples:")
    movie_raw_data.show(10)
    movie_raw_data.printSchema()
    return movie_raw_data

def read_rating_data(spark, file_path):
    # 读取csv文件为dataframe
    rating_data = spark.read.format('csv').option('header', 'true').load(file_path)
    rating_data.show(10)
    rating_data.printSchema()
    return rating_data


def OneHotEncoding(movie_raw_data):
    # col()函数是用来提取 dataframe中的某一列，并且可以应用于计算
    # lit()函数是用来将常量添加到dataframe里面，withColumn("res_", lit(res))
    # withColumn() 函数 是用来对列进行操作 withColumn(res_col, col)
    samplesWithIdNumber = movie_raw_data.withColumn("movieIdNumber", F.col("movieId").cast(IntegerType()))
    encoder =  OneHotEncoder(inputCols=["movieIdNumber"], outputCols=['movieIdVector'], dropLast=False)
    oneHotEncoderSamples = encoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
    oneHotEncoderSamples.show(10)
    oneHotEncoderSamples.printSchema()
    return oneHotEncoderSamples

def array2vec(genreIndexes, indexSize):
    # Vectors.sparse() 创建稀疏向量 
    # Vectors.sparse(向量长度，索引数组，数值数组)
    genreIndexes.sort()
    fill_list = [1.0 for _ in range(len(genreIndexes))]
    return Vectors.sparse(indexSize, genreIndexes, fill_list)

def MultiHotEncoding(movie_raw_data):
    # 将genres展开，并重新命名为genre
    samplesWithGenre = movie_raw_data.select("movieId", "title", explode(
        split(F.col("genres"), "\\|").cast(ArrayType(StringType()))).alias('genre'))
    
    # 使用 StringIndexer 转换器将一列类别型特征进行编码，使其数值化
    # 索引的范围从0开始，该过程可以使得相应的特征索引化，使得某些无法接受类别型特征的算法可以使用
    # 索引构建的顺序为标签的频率，优先编码频率较大的标签，所以出现频率最高的标签为0号
    # 如果输入的是数值型的，我们会把它转化成字符型，然后再对其进行编码
    genreIndexer = StringIndexer(inputCol="genre", outputCol="genreIndex")
    StringIndexerModel = genreIndexer.fit(samplesWithGenre)
    genreIndexSamples = StringIndexerModel.transform(samplesWithGenre).withColumn("genreIndex", F.col("genreIndex").cast(IntegerType()))

    # 开始制作 multi hot vector
    # agg 作为聚合函数，可以与groupby函数一起使用，表示对分组后的数据进行聚合操作
    # 如果没有 groupby函数，则默认对整个dataframe进行聚合操作
    # collect_list 把某一列值聚合成为一个列表
    indexSize = genreIndexSamples.agg(max(F.col("genreIndex"))).head()[0] + 1
    processedSamples = genreIndexSamples.groupBy("movieId").agg(F.collect_list("genreIndex").alias("genreIndexes")).withColumn("indexSize", F.lit(indexSize))

    # udf() 将一般函数注册为spark dataframe可以使用的 udf函数
    # udf(要注册的函数，返回的类型) => 函数
    finalSample = processedSamples.withColumn("vector",
                                udf(array2vec, VectorUDT())(F.col("genreIndexes"), F.col("indexSize")))
    finalSample.show(10)
    finalSample.printSchema()

    return finalSample
                                             
def ratingFeatures(rating_data):
    movieFeatures = rating_data.groupBy('movieId').agg(F.count(F.lit(1)).alias('ratingCount'),
                                                         F.avg("rating").alias("avgRating"),
                                                         F.variance('rating').alias('ratingVar')) \
        .withColumn('avgRatingVec', udf(lambda x: Vectors.dense(x), VectorUDT())(F.col('avgRating')))
    movieFeatures.show(10)
    movieFeatures.printSchema()
    
    # QuantileDiscretizer(numBuckets=2*, inputCol=None, outputCol=None, relativeError=0.001, handleInvalid=‘error’)
    # 采用具有连续特征的列，并输出具有分箱分类特征的列
    # QuantileDiscretizer 在数据集中找到 NaN 值时会引发错误，
    # 可以通过设置 handleInvalid 参数来选择保留或删除数据集中的 NaN 值。
    ratingCountDiscretizer = QuantileDiscretizer(numBuckets=100, inputCol="ratingCount", outputCol="ratingCountBucket")

    # MinMaxScaler(min=0.0, max=1.0, inputCol=None, outputCol=None)
    # 输入和输出都是 DenseVector
    ratingScaler = MinMaxScaler(inputCol="avgRatingVec", outputCol="scaleAvgRating")
    pipelineStage = [ratingCountDiscretizer, ratingScaler]
    # Pipeline
    #
    featurePipeline = Pipeline(stages=pipelineStage)
    movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)
    movieProcessedFeatures.show(10)

    


if __name__ == "__main__":
    # 新建一个spark程序入口
    conf = SparkConf().setAppName('featureEngineering').setMaster('local[*]')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    data_path = "/Users/huangqiushi/Desktop/LearningRS/MyRecSys/Datas/"
    movie_resource_path = data_path + "movies.csv"
    rating_resource_path = data_path + "ratings.csv"

    movie_data = read_movie_data(spark=spark, file_path= movie_resource_path)
    one_hot_data = OneHotEncoding(movie_raw_data=movie_data)
    multi_hot_data = MultiHotEncoding(movie_raw_data=movie_data)

    rating_data = read_rating_data(spark=spark, file_path=rating_resource_path)
    rating_feature = ratingFeatures(rating_data)






