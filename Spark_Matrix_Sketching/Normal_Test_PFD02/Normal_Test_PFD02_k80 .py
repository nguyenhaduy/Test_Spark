#import findspark
#findspark.init('/home/duynguyen/spark-master')

import numpy as np
from matrix_sketching import MatrixSketching
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("PysparkSVDTest")
sc = SparkContext(conf=conf)


filename = '/home/hduser/Test_Spark/Spark_Matrix_Sketching/matrix.csv'
raw_data = open(filename, 'rt')
A = np.loadtxt(raw_data, delimiter=",")

approxCovarianceMatrixA = np.dot(A.transpose(),A)
Norm_A = np.linalg.norm(A,ord='fro')

# isvd Test
l = 80
ms =  MatrixSketching(sc=sc,rows=l,columns=500,op=0.2)
for i in range(10000):
    row = A[i,:]
    ms.add(row)
B = ms.getLocalSketchMatrix()
approxCovarianceMatrixB = np.dot(B.transpose(),B)
testMatrix = approxCovarianceMatrixA - approxCovarianceMatrixB
Test1 = np.linalg.norm(testMatrix,ord=2)/(Norm_A**2)
print(Test1)
del(ms)

np.savetxt("Test_PFD02_80.csv", Test1, delimiter=",")

sc.stop()

# # FD Test
# Test2 = np.zeros(9)
# for j in range(9):
#     l = 20 + 10*j
#     ms =  MatrixSketching(sc=sc,rows=l,columns=500,op='fd')
#     for i in range(10000):
#         row = A[i,:]
#         ms.add(row)
#     B = ms.getLocalSketchMatrix()
#     approxCovarianceMatrixB = np.dot(B.transpose(),B)
#     testMatrix = approxCovarianceMatrixA - approxCovarianceMatrixB
#     Test2[j] = np.linalg.norm(testMatrix,ord=2)/(Norm_A**2)
#     del(ms)

# np.savetxt("random_Test2.csv", Test2, delimiter=",")

# # SSD Test
# Test3 = np.zeros(9)
# for j in range(9):
#     l = 20 + 10*j
#     ms =  MatrixSketching(sc=sc,rows=l,columns=500,op='ssd')
#     for i in range(10000):
#         row = A[i,:]
#         ms.add(row)
#     B = ms.getLocalSketchMatrix()
#     approxCovarianceMatrixB = np.dot(B.transpose(),B)
#     testMatrix = approxCovarianceMatrixA - approxCovarianceMatrixB
#     Test3[j] = np.linalg.norm(testMatrix,ord=2)/(Norm_A**2)
#     del(ms)

# np.savetxt("random_Test3.csv", Test3, delimiter=",")

# # PFD 0.2 Test
# Test4 = np.zeros(9)
# for j in range(9):
#     l = 20 + 10*j
#     ms =  MatrixSketching(sc=sc,rows=l,columns=500,op=0.2)
#     for i in range(10000):
#         row = A[i,:]
#         ms.add(row)
#     B = ms.getLocalSketchMatrix()
#     approxCovarianceMatrixB = np.dot(B.transpose(),B)
#     testMatrix = approxCovarianceMatrixA - approxCovarianceMatrixB
#     Test4[j] = np.linalg.norm(testMatrix,ord=2)/(Norm_A**2)
#     del(ms)

# np.savetxt("random_Test4.csv", Test4, delimiter=",")

# # PFD 0.8 Test
# Test5 = np.zeros(9)
# for j in range(9):
#     l = 20 + 10*j
#     ms =  MatrixSketching(sc=sc,rows=l,columns=500,op=0.8)
#     for i in range(10000):
#         row = A[i,:]
#         ms.add(row)
#     B = ms.getLocalSketchMatrix()
#     approxCovarianceMatrixB = np.dot(B.transpose(),B)
#     testMatrix = approxCovarianceMatrixA - approxCovarianceMatrixB
#     Test5[j] = np.linalg.norm(testMatrix,ord=2)/(Norm_A**2)
#     del(ms)

# np.savetxt("random_Test5.csv", Test5, delimiter=",")
