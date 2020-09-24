# -*- coding: utf-8 -*-
from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.sql import Row
import pyspark.sql.functions as f
from pyspark.sql.functions import udf
from pyspark.sql import Row
from pyspark.sql.types import *
import time
import datetime
import os
import sys
import logging
import json
import operator
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, ArrayType
import redis
from collections import defaultdict

total_cores = 128
executor_cores = 2
executor_instances = total_cores / executor_cores
memory_per_core = 1
app_name = 'abc'
master = 'yarn'
builder = SparkSession.builder.appName(app_name)
builder.master(master)

spark = builder.enableHiveSupport().getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('ERROR')
logger.info('Spark version: %s.', spark.version)

redis_conn = None
def RConn():
    global redis_conn
    if not redis_conn:
        redis_conn = redis.Redis(host=, password=, port=,db=)
    return redis_conn

def to_redis(data_iter):
    r = RConn()
    for ele in data_iter:
        uid = str(ele[0])
        readlist_str = str(ele[1])
        r.set(uid, readlist_str)
        r.expire(uid, 86400 * 1)

def filter_docid(uid, info):
    doc_time_list = info.split("#")
    docid_time_dict = {}
    for doc_time in doc_time_list:
        doc_itme_split = doc_time.split("_")
        if len(doc_itme_split) != 2:
            continue
        docid, time = doc_itme_split
        docid_time_dict[docid] = int(time)
    sort_list = sorted(docid_time_dict.items(), key = operator.itemgetter(1), reverse = False)
    doc_sort = [ele[0] for ele in sort_list]
    doc_sort_len = len(doc_sort)
    return (uid, ",".join(doc_sort), doc_sort_len)
    

def process():
    today =  datetime.datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.datetime.now()-datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    ago_30_day = (datetime.datetime.now()-datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    save_file = "hdfs://inner-di-hdfs.1sapp.com/algo/user/zhangqianjin/test/day=%s"%(yesterday)
    os.system("hadoop fs -rm -r " + save_file)
    
    sql = """
    df = spark.sql(sql)
    rdd = df.rdd.map(lambda x:filter_docid(x.member_id,x.docid_time)).filter( lambda x:x[2]>5)
    rdd.foreachPartition(lambda x:to_redis(x))
    # rdd.saveAsTextFile(save_file)

if __name__ == '__main__':
    process()
    sc.stop()
