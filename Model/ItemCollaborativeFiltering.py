from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql.types import *
import math


def generateItemRatingPairs(movieRatingPairs):
    rawPairs = []
    for pairString in movieRatingPairs:
        movieId, rating = pairString.split(":")
        rawPairs.append((movieId, rating))
    
    pairNum = len(rawPairs)
    processedPair = []

    for i in range(pairNum-1):
        movieIdA = rawPairs[i][0]
        ratingA = float(rawPairs[i][1])
        for j in range(i+1, pairNum):
            movieIdCache = [movieIdA]
            movieIdB = rawPairs[j][0]
            ratingB = float(rawPairs[j][1])
            movieIdCache.append(movieIdB)
            movieIdCache.sort()
            Product = ratingA*ratingB
            processedPair.append("_".join(movieIdCache) + ":" + str(Product))
    return processedPair

def generateItemRatingNormSquare(ratings):
    itemNormSquare = 0
    for rating in ratings:
        itemNormSquare += float(rating)*float(rating)
    return itemNormSquare

def calCosValue(product, norm1, norm2):
    return product / (math.sqrt(norm1) * math.sqrt(norm2))

# def summarizeItemSim(itemIdList, simList):
#     itemSim = []
#     for itemId, sim in zip(itemIdList, simList):
#         itemSim.append({itemId : sim})
#     return itemSim

def summarizeItemSimList(simLists):
    itemSim = []
    for simList in simLists:
        itemSim += simList
    return itemSim


if __name__ == "__main__":
    conf = SparkConf().setAppName("UserCF").setMaster("local[*]")
    spark = SparkSession.builder.config(conf = conf).getOrCreate()

    ratingDataPath = "/Users/huangqiushi/Desktop/LearningRS/MyRecSys/Datas/ratings.csv"
    ratingRawSamples = spark.read.format("csv").option('header', 'true').load(ratingDataPath)
    training, test = ratingRawSamples.randomSplit((0.8, 0.2))
    # training.printSchema()

    itemRatingPairs = training.withColumn("movieRatingPair", concat_ws(":", F.col("movieId"), F.col("rating"))) \
            .groupBy("userId") \
            .agg(udf(generateItemRatingPairs, ArrayType(StringType()))(F.collect_list("movieRatingPair")).alias("ItemPairs")) \
            .select(explode(F.col("itemPairs")).alias("itemPair")) \
            .withColumn("itemIdPair", split(F.col("itemPair"), ":")[0]) \
            .withColumn("itemPairProduct", split(F.col("itemPair"), ":")[1]) \
            .select(F.col("itemIdPair"), F.col("itemPairProduct")) \
            .groupBy("itemIdPair") \
            .agg(sum(F.col("itemPairProduct")).alias("itemPairProductSum"))
    
    itemRatingPairs.show()
    itemRatingPairs.printSchema()

    itemNorms = training.groupBy("movieId") \
            .agg(udf(generateItemRatingNormSquare, FloatType())(F.collect_list("rating")).alias("itemNorms"))
    
    itemNormsJoin1 = itemNorms.withColumnRenamed("movieId", "movieId1") \
                    .withColumnRenamed("itemNorms", "itemNorms1")
    
    itemNormsJoin2 = itemNorms.withColumnRenamed("movieId", "movieId2") \
                    .withColumnRenamed("itemNorms", "itemNorms2")
    
    itemNorms.show()

    itemRatingPairsJoin = itemRatingPairs.withColumn("movieIdHead", split(F.col("itemIdPair"), "_")[0]) \
                    .withColumn("movieIdTail", split(F.col("itemIdPair"), "_")[1])
    
    itemPairCosRaw =  itemRatingPairsJoin.join(itemNormsJoin1, itemRatingPairsJoin.movieIdHead == itemNormsJoin1.movieId1, "left") \
                    .join(itemNormsJoin2, itemRatingPairsJoin.movieIdTail == itemNormsJoin2.movieId2, "left") \
                    .withColumn("cos", udf(calCosValue, FloatType())(F.col("itemPairProductSum"), F.col("itemNorms1"), F.col("itemNorms2"))) \
                    .select(F.col("movieIdHead"), F.col("movieIdTail"), F.col("cos")) \
                    .filter(F.col("cos") >= 0.3)
    
    itemPairCosRaw.show()
    itemPairCosRaw.printSchema()
    # spark中同时使用多个 collect_list() 时， 不同保证两个List 是同样的顺序，因此不可以像如下方法使用： 
    # itemPairCosHead = itemPairCosRaw.groupBy("movieIdHead") \
    #                     .agg(udf(summarizeItemSim, ArrayType(MapType(StringType(), FloatType())))(F.collect_list("movieIdTail"), F.collect_list("cos")).alias("simList"))
    # itemPairCosHead.show()

    itemPairCosHead = itemPairCosRaw.groupBy("movieIdHead") \
                        .agg(F.collect_list(struct(F.col("movieIdTail"), F.col("cos"))).alias("simList"))
    
    itemPairCosHead.show()

    itemPairCosTail = itemPairCosRaw.groupBy("movieIdTail") \
                        .agg(F.collect_list(struct(F.col("movieIdHead"), F.col("cos"))).alias("simList"))      
    
    itemPairCosTail.show()

    itemPairCosProcessed = itemPairCosHead.union(itemPairCosTail) \
                            .withColumnRenamed("movieIdHead", "movieId") \
                            .groupBy("movieId") \
                            .agg(udf(summarizeItemSimList, ArrayType(StructType([StructField("movieIdTail", StringType(), True), StructField("cos", FloatType(), True)]))) \
                                 (F.collect_list("simList")))
    itemPairCosProcessed.show()
    itemPairCosProcessed.printSchema()


    

