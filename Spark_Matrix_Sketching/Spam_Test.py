#import findspark
#findspark.init('/home/duynguyen/spark-master')

import numpy as np
from scipy.io import arff
from matrix_sketching import MatrixSketching
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("PysparkSVDTest")
sc = SparkContext(conf=conf)


data, meta = arff.loadarff('/home/duynguyen/Test_Spark/Spark_Matrix_Sketching/spam_data.arff')
A = []
for i in range(len(data)):
    A.append(np.hstack(data[i]).astype(np.integer))

B = np.zeros((9324,500))
for i in range(len(A)):
    B[i] = A[i]

A = np.array(B)
del(B)

approxCovarianceMatrixA = np.dot(A.transpose(),A)
Norm_A = np.linalg.norm(A,ord='fro')

# isvd Test
Test1 = np.zeros(9)
for j in range(9):
    l = 20 + 10*j
    ms =  MatrixSketching(sc=sc,rows=l,columns=500,op='isvd')
    for i in range(9324):
        row = A[i,:]
        ms.add(row)
    B = ms.getLocalSketchMatrix()
    approxCovarianceMatrixB = np.dot(B.transpose(),B)
    testMatrix = approxCovarianceMatrixA - approxCovarianceMatrixB
    Test1[j] = np.linalg.norm(testMatrix,ord=2)/(Norm_A**2)
    del(ms)

np.savetxt("spam_Test1.csv", Test1, delimiter=",")

# FD Test
Test2 = np.zeros(9)
for j in range(9):
    l = 20 + 10*j
    ms =  MatrixSketching(sc=sc,rows=l,columns=500,op='fd')
    for i in range(9324):
        row = A[i,:]
        ms.add(row)
    B = ms.getLocalSketchMatrix()
    approxCovarianceMatrixB = np.dot(B.transpose(),B)
    testMatrix = approxCovarianceMatrixA - approxCovarianceMatrixB
    Test2[j] = np.linalg.norm(testMatrix,ord=2)/(Norm_A**2)
    del(ms)

np.savetxt("spam_Test2.csv", Test2, delimiter=",")

# SSD Test
Test3 = np.zeros(9)
for j in range(9):
    l = 20 + 10*j
    ms =  MatrixSketching(sc=sc,rows=l,columns=500,op='ssd')
    for i in range(9324):
        row = A[i,:]
        ms.add(row)
    B = ms.getLocalSketchMatrix()
    approxCovarianceMatrixB = np.dot(B.transpose(),B)
    testMatrix = approxCovarianceMatrixA - approxCovarianceMatrixB
    Test3[j] = np.linalg.norm(testMatrix,ord=2)/(Norm_A**2)
    del(ms)

np.savetxt("spam_Test3.csv", Test3, delimiter=",")

# PFD 0.2 Test
Test4 = np.zeros(9)
for j in range(9):
    l = 20 + 10*j
    ms =  MatrixSketching(sc=sc,rows=l,columns=500,op=0.2)
    for i in range(9324):
        row = A[i,:]
        ms.add(row)
    B = ms.getLocalSketchMatrix()
    approxCovarianceMatrixB = np.dot(B.transpose(),B)
    testMatrix = approxCovarianceMatrixA - approxCovarianceMatrixB
    Test4[j] = np.linalg.norm(testMatrix,ord=2)/(Norm_A**2)
    del(ms)

np.savetxt("spam_Test4.csv", Test4, delimiter=",")

# PFD 0.8 Test
Test5 = np.zeros(9)
for j in range(9):
    l = 20 + 10*j
    ms =  MatrixSketching(sc=sc,rows=l,columns=500,op=0.8)
    for i in range(9324):
        row = A[i,:]
        ms.add(row)
    B = ms.getLocalSketchMatrix()
    approxCovarianceMatrixB = np.dot(B.transpose(),B)
    testMatrix = approxCovarianceMatrixA - approxCovarianceMatrixB
    Test5[j] = np.linalg.norm(testMatrix,ord=2)/(Norm_A**2)
    del(ms)

np.savetxt("spam_Test5.csv", Test5, delimiter=",")
