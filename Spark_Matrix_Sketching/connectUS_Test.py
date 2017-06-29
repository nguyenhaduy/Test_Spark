import findspark
findspark.init('/home/duynguyen/spark-master')

import numpy as np
from scipy.io import mmread
from matrix_sketching import MatrixSketching
from pyspark import SparkContext
sc = SparkContext(appName="PysparkSVDTest")


data = mmread('connectus.mtx')
temp = data.toarray()
A = temp.transpose()

del(data)
del(temp)

approxCovarianceMatrixA = np.dot(A.transpose(),A)
Norm_A = np.linalg.norm(A,ord='fro')

# isvd Test
Test1 = np.zeros(9)
for j in range(9):
    l = 20 + 10*j
    ms =  MatrixSketching(sc=sc,rows=l,columns=512,op='isvd')
    for i in range(394792):
        row = A[i,:]
        ms.add(row)
    B = ms.getLocalSketchMatrix()
    approxCovarianceMatrixB = np.dot(B.transpose(),B)
    testMatrix = approxCovarianceMatrixA - approxCovarianceMatrixB
    Test1[j] = np.linalg.norm(testMatrix,ord=2)/(Norm_A**2)
    del(ms)

np.savetxt("connectUS_Test1.csv", Test1, delimiter=",")

# FD Test
Test2 = np.zeros(9)
for j in range(9):
    l = 20 + 10*j
    ms =  MatrixSketching(sc=sc,rows=l,columns=512,op='fd')
    for i in range(394792):
        row = A[i,:]
        ms.add(row)
    B = ms.getLocalSketchMatrix()
    approxCovarianceMatrixB = np.dot(B.transpose(),B)
    testMatrix = approxCovarianceMatrixA - approxCovarianceMatrixB
    Test2[j] = np.linalg.norm(testMatrix,ord=2)/(Norm_A**2)
    del(ms)

np.savetxt("connectUS_Test2.csv", Test2, delimiter=",")

# SSD Test
Test3 = np.zeros(9)
for j in range(9):
    l = 20 + 10*j
    ms =  MatrixSketching(sc=sc,rows=l,columns=512,op='ssd')
    for i in range(394792):
        row = A[i,:]
        ms.add(row)
    B = ms.getLocalSketchMatrix()
    approxCovarianceMatrixB = np.dot(B.transpose(),B)
    testMatrix = approxCovarianceMatrixA - approxCovarianceMatrixB
    Test3[j] = np.linalg.norm(testMatrix,ord=2)/(Norm_A**2)
    del(ms)

np.savetxt("connectUS_Test3.csv", Test3, delimiter=",")

# PFD 0.2 Test
Test4 = np.zeros(9)
for j in range(9):
    l = 20 + 10*j
    ms =  MatrixSketching(sc=sc,rows=l,columns=512,op=0.2)
    for i in range(394792):
        row = A[i,:]
        ms.add(row)
    B = ms.getLocalSketchMatrix()
    approxCovarianceMatrixB = np.dot(B.transpose(),B)
    testMatrix = approxCovarianceMatrixA - approxCovarianceMatrixB
    Test4[j] = np.linalg.norm(testMatrix,ord=2)/(Norm_A**2)
    del(ms)

np.savetxt("connectUS_Test4.csv", Test4, delimiter=",")

# PFD 0.8 Test
Test5 = np.zeros(9)
for j in range(9):
    l = 20 + 10*j
    ms =  MatrixSketching(sc=sc,rows=l,columns=512,op=0.8)
    for i in range(394792):
        row = A[i,:]
        ms.add(row)
    B = ms.getLocalSketchMatrix()
    approxCovarianceMatrixB = np.dot(B.transpose(),B)
    testMatrix = approxCovarianceMatrixA - approxCovarianceMatrixB
    Test5[j] = np.linalg.norm(testMatrix,ord=2)/(Norm_A**2)
    del(ms)

np.savetxt("connectUS_Test5.csv", Test5, delimiter=",")