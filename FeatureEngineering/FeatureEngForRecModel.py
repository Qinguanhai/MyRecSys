# 一般来说，我们不会人为预设哪个特征有用，哪个特征无用，而是让模型自己去判断，如果一个特征的加入没有提升模型效果，我们再去除这个特征。
# 就像我刚才虽然提取了不少特征，但并不是说每个模型都会使用全部的特征，而是根据模型结构、模型效果有针对性地部分使用它们。

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.functions import *
from pyspark.sql.types import *
from collections import defaultdict
from pyspark.sql import functions as F

NUMBER_PRECISION = 2

def addSampleLabel(ratingSamples):
    ratingSamples.show(5, truncate=False)
    ratingSamples.printSchema()
    sampleCount = ratingSamples.count()
    # 统计每个打分区间的占比
    ratingSamples.groupBy('rating').count().orderBy('rating').withColumn('percentage',
                                                                         F.col('count') / sampleCount).show()
    # 生成label
    ratingSamples = ratingSamples.withColumn('label', when(F.col('rating') >= 3.5, 1).otherwise(0))
    return ratingSamples


def extractReleaseYearUdf(title):
    # 从标题中提取发布的年份，默认为1990
    if not title or len(title.strip()) < 6:
        return 1990
    else:
        yearStr = title.strip()[-5:-1]
    return int(yearStr)


def addMovieFeatures(movieSamples, ratingSamplesWithLabel):
    # # # SPARROW REC 中原始逻辑，较为繁琐，无用操作过多
    # # 将movie的基础信息和raw label join 在一起
    # samplesWithMovies1 = ratingSamplesWithLabel.join(movieSamples, on=['movieId'], how='left')

    # # 只提取movie的年份，title信息需要使用nlp处理后才能使用
    # samplesWithMovies2 = samplesWithMovies1.withColumn('releaseYear',
    #                                                    udf(extractReleaseYearUdf, IntegerType())('title')) \
    #     .withColumn('title', udf(lambda x: x.strip()[:-6].strip(), StringType())('title')) \
    #     .drop('title')
    
    # # 只保留最多3个电影的genre 
    # samplesWithMovies3 = samplesWithMovies2.withColumn('movieGenre1', split(F.col('genres'), "\\|")[0]) \
    #     .withColumn('movieGenre2', split(F.col('genres'), "\\|")[1]) \
    #     .withColumn('movieGenre3', split(F.col('genres'), "\\|")[2])
    
    # # 添加电影的统计特征（非滑动窗口）
    # movieRatingFeatures = samplesWithMovies3.groupBy('movieId').agg(F.count(F.lit(1)).alias('movieRatingCount'),
    #                                                                 format_number(F.avg(F.col('rating')),
    #                                                                               NUMBER_PRECISION).alias(
    #                                                                     'movieAvgRating'),
    #                                                                 F.stddev(F.col('rating')).alias(
    #                                                                     'movieRatingStddev')).fillna(0) \
    #     .withColumn('movieRatingStddev', format_number(F.col('movieRatingStddev'), NUMBER_PRECISION))
    
    # # join movie rating features
    # samplesWithMovies4 = samplesWithMovies3.join(movieRatingFeatures, on=['movieId'], how='left')
    # samplesWithMovies4.printSchema()
    # samplesWithMovies4.show(5, truncate=False)
    # return samplesWithMovies4
    
    # 更新后的处理逻辑，相对直接
    # Step 1: 处理基础信息特征
    processedMovieSamples = movieSamples.withColumn('releaseYear', udf(extractReleaseYearUdf, IntegerType())('title')) \
            .drop('title') \
            .withColumn('movieGenre1', split(F.col('genres'), "\\|")[0]) \
            .withColumn('movieGenre2', split(F.col('genres'), "\\|")[1]) \
            .withColumn('movieGenre3', split(F.col('genres'), "\\|")[2])
    
    # Step 2: 处理统计特征, 添加电影的统计特征（滑动窗口）
    processedRatingSamplesWithLabel = ratingSamplesWithLabel.withColumn("movieRatingCount", 
        F.count(F.lit(1)).over(sql.Window.partitionBy('movieId').orderBy('timestamp').rowsBetween(-100, -1))) \
            .withColumn("movieAvgRating", 
        format_number(F.avg(F.col('rating')).over(sql.Window.partitionBy('movieId').orderBy('timestamp').rowsBetween(-100, -1)), NUMBER_PRECISION)) \
            .withColumn("movieRatingStddev",
        format_number(F.stddev(F.col('rating')).over(sql.Window.partitionBy('movieId').orderBy('timestamp').rowsBetween(-100, -1)), NUMBER_PRECISION)) \
        .na.fill("0.00")
    
    # Step 3: 组合基础信息特征和统计特征
    processedRatingSamplesWithLabel = processedRatingSamplesWithLabel.join(processedMovieSamples, on=['movieId'], how='left')
    processedRatingSamplesWithLabel.printSchema()
    processedRatingSamplesWithLabel.show(5, truncate=False)
    return processedRatingSamplesWithLabel
    


def extractGenres(genres_list):
    '''
    pass in a list which format like ["Action|Adventure|Sci-Fi|Thriller", "Crime|Horror|Thriller"]
    count by each genre, return genre_list in reverse order
    eg:
    (('Thriller',2),('Action',1),('Sci-Fi',1),('Horror', 1), ('Adventure',1),('Crime',1))
    return:['Thriller','Action','Sci-Fi','Horror','Adventure','Crime']
    '''
    genres_dict = defaultdict(int)
    for genres in genres_list:
        for genre in genres.split('|'):
            genres_dict[genre] += 1
    sortedGenres = sorted(genres_dict.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sortedGenres]


def addUserFeatures(samplesWithMovieFeatures):

    extractGenresUdf = udf(extractGenres, ArrayType(StringType()))

    # 1) F.when(): when in pyspark multiple conditions can be built using &  (for and) and | (for or), 
    #   it is important to enclose every expressions within parenthesis that combine to form the condition
    #
    # 2) F.when() 可以通过 "." 连用:
    # dataDF.withColumn("new_column",
    #   when((col("code") == "a") | (col("code") == "d"), "A")
    #  .when((col("code") == "b") & (col("amt") == "4"), "B")
    #  .otherwise("A1")).show()

    # 1）userPositiveHistory: 用户最近100条的有正反馈的movieId
    # 2）userRatedMoive1-5: 用户最近5个有正反馈的movieId
    # 3）userRatingCount: 用户至今评价过了视频数，100条截断
    # 4）userAvgReleaseYear, userReleaseYearStddev: 用户最近100条评价电影的发布年份和标准差 -> 冷启是否可以添加 用户最近消费正向的视频的新鲜度？来判断是否是一个适合发冷启的用户
    # 5）userAvgRating, userRatingStddev: 用户最近100条评价的均值和方差 -> 我们现有的特征体系中是不是很缺少方差？
    # 6) userGenres:  用户最近100条评价中好评的genre，按照genre出现的次数降序排序
    # 7）userGenre1-5: 用户最近100条评价中好评的genre TOP 5
    # 8）同时过滤除此评价的用户特征数据
    
    samplesWithUserFeatures = samplesWithMovieFeatures \
        .withColumn('userPositiveHistory',
                    F.collect_list(when(F.col('label') == 1, F.col('movieId')).otherwise(F.lit(None))).over(
                        sql.Window.partitionBy("userId").orderBy(F.col("timestamp")).rowsBetween(-100, -1))) \
        .withColumn("userPositiveHistory", reverse(F.col("userPositiveHistory"))) \
        .withColumn('userRatedMovie1', F.col('userPositiveHistory')[0]) \
        .withColumn('userRatedMovie2', F.col('userPositiveHistory')[1]) \
        .withColumn('userRatedMovie3', F.col('userPositiveHistory')[2]) \
        .withColumn('userRatedMovie4', F.col('userPositiveHistory')[3]) \
        .withColumn('userRatedMovie5', F.col('userPositiveHistory')[4]) \
        .withColumn('userRatingCount',
                    F.count(F.lit(1)).over(sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn('userAvgReleaseYear', F.avg(F.col('releaseYear')).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)).cast(IntegerType())) \
        .withColumn('userReleaseYearStddev', F.stddev(F.col("releaseYear")).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn("userAvgRating", format_number(
        F.avg(F.col("rating")).over(sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)),
        NUMBER_PRECISION)) \
        .withColumn("userRatingStddev", F.stddev(F.col("rating")).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn("userGenres", extractGenresUdf(
        F.collect_list(when(F.col('label') == 1, F.col('genres')).otherwise(F.lit(None))).over(
            sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)))) \
        .withColumn("userRatingStddev", format_number(F.col("userRatingStddev"), NUMBER_PRECISION)) \
        .withColumn("userReleaseYearStddev", format_number(F.col("userReleaseYearStddev"), NUMBER_PRECISION)) \
        .withColumn("userGenre1", F.col("userGenres")[0]) \
        .withColumn("userGenre2", F.col("userGenres")[1]) \
        .withColumn("userGenre3", F.col("userGenres")[2]) \
        .withColumn("userGenre4", F.col("userGenres")[3]) \
        .withColumn("userGenre5", F.col("userGenres")[4]) \
        .drop("genres", "userGenres", "userPositiveHistory") \
        .filter(F.col("userRatingCount") > 1)
    samplesWithUserFeatures.printSchema()
    samplesWithUserFeatures.show(10)
    samplesWithUserFeatures.filter(samplesWithMovieFeatures['userId'] == 1).orderBy(F.col('timestamp').asc()).show(
        truncate=False)
    return samplesWithUserFeatures


def splitAndSaveTrainingTestSamples(samplesWithUserFeatures, file_path):
    smallSamples = samplesWithUserFeatures.sample(0.1)
    training, test = smallSamples.randomSplit((0.8, 0.2))
    trainingSavePath = file_path + '/trainingSamples'
    testSavePath = file_path + '/testSamples'
    training.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(trainingSavePath)
    test.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(testSavePath)


def splitAndSaveTrainingTestSamplesByTimeStamp(samplesWithUserFeatures, file_path):
    smallSamples = samplesWithUserFeatures.sample(0.1).withColumn("timestampLong", F.col("timestamp").cast(LongType()))
    quantile = smallSamples.stat.approxQuantile("timestampLong", [0.8], 0.05)
    splitTimestamp = quantile[0]
    training = smallSamples.where(F.col("timestampLong") <= splitTimestamp).drop("timestampLong")
    test = smallSamples.where(F.col("timestampLong") > splitTimestamp).drop("timestampLong")
    trainingSavePath = file_path + '/trainingSamples'
    testSavePath = file_path + '/testSamples'
    training.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(trainingSavePath)
    test.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(testSavePath)


if __name__ == '__main__':
    conf = SparkConf().setAppName('featureEngineering').setMaster('local[*]')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    file_path = '/Users/huangqiushi/Desktop/LearningRS/MyRecSys/Datas/'
    movieResourcesPath = file_path + "movies.csv"
    ratingsResourcesPath = file_path + "ratings.csv"

    movieSamples = spark.read.format('csv').option('header', 'true').load(movieResourcesPath)
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingsResourcesPath)

    # Step1: 生成label
    ratingSamplesWithLabel = addSampleLabel(ratingSamples)
    ratingSamplesWithLabel.show(10, truncate=False)

    # Step2: 拼接movie信息
    samplesWithMovieFeatures = addMovieFeatures(movieSamples, ratingSamplesWithLabel)
    
    # Step3: 拼接user信息
    samplesWithUserFeatures = addUserFeatures(samplesWithMovieFeatures)

    # save samples as csv format
    # splitAndSaveTrainingTestSamples(samplesWithUserFeatures, file_path + "sampledata")

    splitAndSaveTrainingTestSamplesByTimeStamp(samplesWithUserFeatures, file_path + "sampledata")
