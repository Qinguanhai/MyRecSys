import os
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.mllib.feature import Word2Vec
from pyspark.ml.linalg import Vectors
import random
from collections import defaultdict
import numpy as np
from pyspark.sql import functions as F
from tqdm import tqdm

class UdfFunction:
    @staticmethod
    def sortF(movie_list, timestamp_list):
        """
        sort by time and return the corresponding movie sequence
        eg:
            input: movie_list:[1,2,3]
                   timestamp_list:[1112486027,1212546032,1012486033]
            return [3,1,2]
        """
        pairs = []
        for m, t in zip(movie_list, timestamp_list):
            pairs.append((m, t))
        # sort by time
        pairs = sorted(pairs, key=lambda x: x[1])
        return [x[0] for x in pairs]


def processItemSequence(spark, rawSampleDataPath):
    # rating data to item pairs
    ratingSamples = spark.read.format("csv").option("header", "true").load(rawSampleDataPath)
    ratingSamples.show(5)
    ratingSamples.printSchema()
    # ArrayType(StringType())
    # array_join()
    sortUdf = udf(UdfFunction.sortF, ArrayType(StringType()))
    userSeq = ratingSamples \
        .where(F.col("rating") >= 3.5) \
        .withColumn("movieId", concat(F.lit("movieId:"), F.col("movieId"))) \
        .groupBy("userId") \
        .agg(sortUdf(F.collect_list("movieId"), F.collect_list("timestamp")).alias('movieIds')) \
        .withColumn("movieIdStr", array_join(F.col("movieIds"), " "))
    itemPairs = userSeq.select('movieIdStr').rdd.map(lambda x: x[0].split(' ')).flatMap(lambda x: generate_pair(x))
    return itemPairs

def processUserItemSequence(spark, rawSampleDataPath):
    # rating data to user item pairs
    ratingSamples = spark.read.format("csv").option("header", "true").load(rawSampleDataPath)
    userItemSeq = ratingSamples \
        .where(F.col("rating") >= 3.5) \
        .select(concat(F.lit("userId:"), F.col("userId")).alias("userId"), concat(F.lit("movieId:"), F.col("movieId")).alias("movieId"))
    userItemSeq.show(5)
    userItemSeq.printSchema()
    return userItemSeq.rdd.map(lambda x : (x[0], x[1]))

def processItemGenreInfo(spark, rawSampleDataPath, itemInfoPath):
    # movie infos to movieId genre pairs
    ratingSamples = spark.read.format("csv").option("header", "true").load(rawSampleDataPath)
    movieInfos = spark.read.format("csv").option("header", "true").load(itemInfoPath)
    movieInfos.show(5)
    movieInfos.printSchema()

    movieInfos = movieInfos.withColumn("movieId", concat(F.lit("movieId:"), F.col("movieId")))
    seenMovies = ratingSamples.select(concat(F.lit("movieId:"), F.col("movieId")).alias("movieId_")).distinct()
    movieInfos = movieInfos.join(seenMovies, movieInfos.movieId == seenMovies.movieId_, "inner") \
            .select("movieId", "genres") \
            .rdd \
            .flatMap(lambda x :generate_item_genre_pair(x)) 
    return movieInfos

def generate_pair(x):
    # eg:
    # watch sequence:['858', '50', '593', '457']
    # return:[['858', '50'],['50', '593'],['593', '457']]
    pairSeq = []
    previousItem = ''
    for item in x:
        if not previousItem:
            previousItem = item
        else:
            pairSeq.append((previousItem, item))
            previousItem = item
    return pairSeq

def generate_item_genre_pair(x):
    # eg:
    # in: ["movieId:12", "Action:Comedy"]
    # return: [("moiveId:12", "Action"), ("movieId:12", "Comedy")]
    pairSeq = []
    item = x[0]
    genres = x[1].split("|")
    for genre in genres:
        pairSeq.append((item, "genre:" + genre))
    return pairSeq

def trainItem2vec(spark, samples, embLength, embOutputPath, saveToRedis, redisKeyPrefix):
    word2vec = Word2Vec().setVectorSize(embLength).setWindowSize(5).setNumIterations(10)
    model = word2vec.fit(samples)
    synonyms = model.findSynonyms("movieId:158", 20)
    for synonym, cosineSimilarity in synonyms:
        print(synonym, cosineSimilarity)
    
    print("start saving embeddings ...")
    embOutputDir = '/'.join(embOutputPath.split('/')[:-1])
    if not os.path.exists(embOutputDir):
        os.makedirs(embOutputDir)
    with open(embOutputPath, 'w') as f:
        for movie_id in model.getVectors():
            vectors = " ".join([str(emb) for emb in model.getVectors()[movie_id]])
            f.write(movie_id + ":" + vectors + "\n")
    embeddingLSH(spark, model.getVectors())
    return model

def embeddingLSH(spark, movieEmbMap):
    movieEmbSeq = []
    for key, embedding_list in movieEmbMap.items():
        embedding_list = [np.float64(embedding) for embedding in embedding_list]
        movieEmbSeq.append((key, Vectors.dense(embedding_list)))
    movieEmbDF = spark.createDataFrame(movieEmbSeq).toDF("movieId", "emb")
    bucketProjectionLSH = BucketedRandomProjectionLSH(inputCol="emb", outputCol="bucketId", bucketLength=0.1,
                                                      numHashTables=3)
    bucketModel = bucketProjectionLSH.fit(movieEmbDF)
    embBucketResult = bucketModel.transform(movieEmbDF)
    print("movieId, emb, bucketId schema:")
    embBucketResult.printSchema()
    print("movieId, emb, bucketId data result:")
    embBucketResult.show(10, truncate=False)
    print("Approximately searching for 5 nearest neighbors of the sample embedding:")
    sampleEmb = Vectors.dense(0.795, 0.583, 1.120, 0.850, 0.174, -0.839, -0.0633, 0.249, 0.673, -0.237)
    bucketModel.approxNearestNeighbors(movieEmbDF, sampleEmb, 5).show(truncate=False)

def generateTransitionMatrix(item2ItemSamples, user2ItemSamples, item2GenreSamples):
    # 从 item - item, user - item, item -genre边中形成转移矩阵
    # 由于metapath随机游走的特性，只需要在同类节点中的转移概率进行归一化

    # 先定义转移矩阵，节点矩阵
    transitionCountMatrix = defaultdict(dict)
    itemDistribution = defaultdict(dict)
    userDistribution = defaultdict(dict)

    # 1）处理item - item 信息 
    pairSamples = item2ItemSamples

    # 将rdd中每一个值作为key 统计出现的次数 返回的值为一个 defaultdict
    pairCountMap = pairSamples.countByValue()
    itemPairTotalCount = 0

    # 给每一个字典设置好默认值，防止取不到的情况报错
    itemCountMap = defaultdict(int)
    for key, cnt in pairCountMap.items():
        key1, key2 = key
        if transitionCountMatrix[key1].get("item") == None:
            transitionCountMatrix[key1]["item"] = {key2:cnt}
        else:
            transitionCountMatrix[key1]["item"][key2] = cnt
        itemCountMap[key1] += cnt
        itemPairTotalCount += cnt

    # 2）处理 user - item 信息
    pairSamples = user2ItemSamples

    pairCountMap = pairSamples.countByValue()
    userPairTotalCount = 0
    userCountMap = defaultdict(int)
    userDisCountMap = defaultdict(int)

    for key, cnt in pairCountMap.items():
        key1, key2 = key

        # user node 中加入 item 链接信息
        if transitionCountMatrix[key1].get("item") == None:
            transitionCountMatrix[key1]["item"] = {key2:cnt}
        else:
            transitionCountMatrix[key1]["item"][key2] = cnt
        
        # item node 中加入 user 链接信息
        if transitionCountMatrix[key2].get("user") == None:
            transitionCountMatrix[key2]["user"] = {key1:cnt}
        else:
            transitionCountMatrix[key2]["user"][key1] = cnt
        userCountMap[key1] += cnt
        userCountMap[key2] += cnt
        userDisCountMap[key1] += cnt
        userPairTotalCount += cnt

    # 3）处理 item - genre 信息
    pairSamples = item2GenreSamples

    pairCountMap = pairSamples.countByValue()
    genrePairTotalCount = 0
    genreCountMap = defaultdict(int)

    for key, cnt in pairCountMap.items():
        key1, key2 = key

        # item node 中加入 genre 链接信息
        if transitionCountMatrix[key1].get("genre") == None:
            transitionCountMatrix[key1]["genre"] = {key2:cnt}
        else:
            transitionCountMatrix[key1]["genre"][key2] = cnt
        
        # genre node 中加入 item 链接信息
        if transitionCountMatrix[key2].get("item") == None:
            transitionCountMatrix[key2]["item"] = {key1:cnt}
        else:
            transitionCountMatrix[key2]["item"][key1] = cnt
        genreCountMap[key1] += cnt
        genreCountMap[key2] += cnt
        genrePairTotalCount += cnt
    
    # 4）分类归一化生成 transition Matrix
    transitionMatrix = defaultdict(dict)
    
    for key1, transitionMap in transitionCountMatrix.items():
        # 如果节点是 item 
        if key1.split(":")[0] == "movieId":

            if transitionMap.get("item") != None:
                for key2, cnt in transitionMap["item"].items():
                    if transitionMatrix[key1].get("item") == None:
                        transitionMatrix[key1]["item"] = {key2: transitionCountMatrix[key1]["item"][key2] / itemCountMap[key1]}
                    else:
                        transitionMatrix[key1]["item"][key2] = transitionCountMatrix[key1]["item"][key2] / itemCountMap[key1]

            if transitionMap.get("user") != None:
                for key2, cnt in transitionMap["user"].items():
                    if transitionMatrix[key1].get("user") == None:
                        transitionMatrix[key1]["user"] = {key2: transitionCountMatrix[key1]["user"][key2] / userCountMap[key1]}
                    else:
                        transitionMatrix[key1]["user"][key2] = transitionCountMatrix[key1]["user"][key2] / userCountMap[key1]
            
            if transitionMap.get("genre") != None:
                for key2, cnt in transitionMap["genre"].items():
                    if transitionMatrix[key1].get("genre") == None:
                        transitionMatrix[key1]["genre"] = {key2: transitionCountMatrix[key1]["genre"][key2] / genreCountMap[key1]}
                    else:
                        transitionMatrix[key1]["genre"][key2] = transitionCountMatrix[key1]["genre"][key2] / genreCountMap[key1]
            
        # 如果节点是 user
        elif key1.split(":")[0] == "userId":
            if transitionMap.get("item") != None:
                for key2, cnt in transitionMap["item"].items():
                    if transitionMatrix[key1].get("item") == None:
                        transitionMatrix[key1]["item"] = {key2: transitionCountMatrix[key1]["item"][key2] / userCountMap[key1]}
                    else:
                        transitionMatrix[key1]["item"][key2] = transitionCountMatrix[key1]["item"][key2] / userCountMap[key1]

        #如果节点是 genre
        elif key1.split(":")[0] == "genre":
            if transitionMap.get("item") != None:
                for key2, cnt in transitionMap["item"].items():
                    if transitionMatrix[key1].get("item") == None:
                        transitionMatrix[key1]["item"] = {key2: transitionCountMatrix[key1]["item"][key2] / genreCountMap[key1]}
                    else:
                        transitionMatrix[key1]["item"][key2] = transitionCountMatrix[key1]["item"][key2] / genreCountMap[key1]

        else:
            continue
        
    for itemid, cnt in itemCountMap.items():
        itemDistribution[itemid] = cnt / itemPairTotalCount
    
    for userid, cnt in userDisCountMap.items():
        userDistribution[userid] = cnt / userPairTotalCount

    return transitionMatrix, itemDistribution, userDistribution

def oneMetaPathWalk(transitionMatrix, itemDistribution, userDistribution, metapath):
    sample = []

    # pick the first element
    if metapath[0] == "i":
        firstNodeDistribution = itemDistribution
    else:
        firstNodeDistribution = userDistribution
    
    randomDouble = random.random()
    firstItem = ""
    accumulateProb = 0.0
    for item, prob in firstNodeDistribution.items():
        accumulateProb += prob
        if accumulateProb >= randomDouble:
            firstItem = item
            break
    sample.append(firstItem)
    curElement = firstItem

    #pick other elements    
    for node in metapath[1:]:
        if curElement not in transitionMatrix:
            break

        if node == "i":
            meta = "item"
        elif node == "u":
            meta = "user"
        elif node == "g":
            meta = "genre"
        else:
            break

        metaDistribution = transitionMatrix[curElement].get(meta)
        if metaDistribution == None:
            break

        randomDouble = random.random()
        accumulateProb = 0.0
        for item, prob in metaDistribution.items():
            accumulateProb += prob
            if accumulateProb >= randomDouble:
                curElement = item
                break
        sample.append(curElement)

    return sample


def metaPathRandomWalk(transitionMatrix, itemDistribution, userDistribution, sampleCounts, metapaths):
    samples = []
    for sampleCount, metapath in zip(sampleCounts, metapaths):
        print("metapath: " + metapath + " is generating")
        for i in tqdm(range(sampleCount)):
            samples.append(oneMetaPathWalk(transitionMatrix, itemDistribution, userDistribution, metapath))
    return samples

def metaPathGraphEmb(itemPairSamples, userItemPairSamples, itemGenrePairSamples, spark, embLength, embOutputFilename, saveToRedis, redisKeyPrefix):
    # 从边的信息中生成转移矩阵
    transitionMatrix, itemDistribution, userDistribution = generateTransitionMatrix(itemPairSamples, userItemPairSamples, itemGenrePairSamples)
    
    # 转移矩阵点检
    # print("item check:")
    # print(transitionMatrix["movieId:2"])
    # print(itemDistribution["movieId:2"])
    # print("user check:")
    # print(transitionMatrix["userId:1"])
    # print(userDistribution["userId:1"])

    # 开始metapath随机游走生成corpus
    sampleCounts = [20000, 10000, 10000, 30000, 30000]
    metapaths = ["uiiuiiuiiu", "igiuigiuigi", "iiuiigiiuii", "iiiiiiiiiii", "iiiuiiiuiii"]
    newSamples = metaPathRandomWalk(transitionMatrix, itemDistribution, userDistribution, sampleCounts, metapaths)

    #corpus点检
    print("corpus length:", len(newSamples))
    print(newSamples[0])

    rddSamples = spark.sparkContext.parallelize(newSamples)
    trainItem2vec(spark, rddSamples, embLength, embOutputFilename, saveToRedis, redisKeyPrefix)

if __name__ == "__main__":
    embLength = 10
    
    conf = SparkConf().setAppName("embedding").setMaster("local[*]")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    out_path = "/Users/huangqiushi/Desktop/LearningRS/MyRecSys/Embedding/result"
    rating_data_path = "/Users/huangqiushi/Desktop/LearningRS/MyRecSys/Datas/ratings.csv"
    movie_info_path = "/Users/huangqiushi/Desktop/LearningRS/MyRecSys/Datas/movies.csv"

    # 从用户的打分序列信息中生产item-item pair数据
    itemPairSamples = processItemSequence(spark, rating_data_path)
    print("check 5 lines of the item-item pair samples:")
    print(itemPairSamples.take(5))

    # 从用户的打分序列信息中生产user-item pair数据
    userItemPairSamples = processUserItemSequence(spark, rating_data_path)
    print("check 5 lines of the user-item pair samples:")
    print(userItemPairSamples.take(5))

    # 从用户的打分序列和电影的作品信息中生产item-genre pair数据
    itemGenrePairSamples = processItemGenreInfo(spark, rating_data_path, movie_info_path)
    print("check 5 lines of the item-genre pair samples:")
    print(itemGenrePairSamples.take(5))

    # Metapath2embedding
    metaPathGraphEmb(itemPairSamples, userItemPairSamples, itemGenrePairSamples, spark, embLength, 
                     embOutputFilename=out_path + "/itemMetaPathGraphEmb.csv", saveToRedis=True, redisKeyPrefix="metaPathEmb")
    