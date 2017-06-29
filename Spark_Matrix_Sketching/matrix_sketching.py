from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
from numpy import zeros, max, sqrt, isnan, isinf, dot, diag, count_nonzero, where


class MatrixSketching(object):
    
    def __init__(self ,sc , rows, columns, op='fd'):
        """
        Matrix Sketching using Frequent Direction.
        Choose 'fd' for normal Frequent Direction, 'ssd' for Space Saving Direction, 'cfd' for Compensative Frequent Direction, 'isvd' for iterative SVD, and a number between 0 and 1 for Parameterized Frequent Direction
        """
        self.class_name = 'MatrixSketching'
        self.sc = sc
        self.op = op
        self.columns = columns
        self.rows = rows
        self.localSketchMatrix = zeros((self.rows, self.columns)) 
        self.distributedSketchMatrix = RowMatrix(self.sc.parallelize(self.localSketchMatrix))
        self.S = zeros(self.rows)
        self.U = []
        self.V = []
        self.step = 1
        self.nextZeroRow = 0
        self.emptyRows = self.rows
        

        # Parsing the operation parameter
        if self.op == 'fd':
            print("Matrix Sketching Using Frequent Direction")
            self.reduceRank = self.__FDOperate__
        elif self.op == 'ssd':
            print("Matrix Sketching Using Space Saving Direction")
            self.op = 2
            self.reduceRank = self.__SSDOperate__
        elif self.op == 'cfd':
            print("Matrix Sketching Using Compensative Frequent Direction")
            self.reduceRank = self.__CFDperate__
        elif self.op == 'isvd':
            print("Matrix Sketching Using iSVD")
            self.reduceRank = self.__iSVDOperate__
        elif type(self.op) != str and self.op > 0 and self.op < 1:
            print("Matrix Sketching Using Parameterized Frequent Direction")
            self.reduceRank = self.__PFDOperate__
            self.DELTA = 0
        else:
            print("Type of Reduce Rank algorithm is not correct")
            raise ValueError
    
    # Add new vector to the sketch matrix
    def add(self,vector):     
        if count_nonzero(vector) == 0:
            return
        
        # If the approximate matrix is full, call the operate method to free half of the columns
        if self.emptyRows <= 0:
            self.svd = self.distributedSketchMatrix.computeSVD(self.rows, computeU=True)
            self.U = self.svd.U       # The U factor is a distributed RowMatrix.
            self.S[:] = self.svd.s[:]      # The singular values are stored in a local dense vector.
            self.V = self.svd.V       # The V factor is a local dense matrix.
            self.reduceRank()

        # Push the new vector to the next zero row and increase the next zero row index
        self.localSketchMatrix[self.nextZeroRow,:] = vector
        del(self.distributedSketchMatrix)
        self.distributedSketchMatrix = RowMatrix(self.sc.parallelize(self.localSketchMatrix))
        self.nextZeroRow += 1
        self.emptyRows -= 1


    # Shrink the approximate matrix using Frequent Direction
    def __FDOperate__(self):
        # Calculating matrix s
        delta = sqrt(self.S[:]**2 - self.S[len(self.S)-1]**2)
        self.S = delta
        self.S[len(self.S)-1] = 0
        #Shrink the sketch matrix
        self.localSketchMatrix[:,:] = dot(diag(self.S), self.V.toArray().transpose()[:len(self.S),:])
        self.nextZeroRow = (len(self.S)-1)
        self.emptyRows += 1
            
        
    # Shrink the approximate matrix using iterative SVD
    def __iSVDOperate__(self):
        # Calculating matrix s
        self.S[len(self.S)-1] = 0
        #Shrink the sketch matrix
        self.localSketchMatrix[:,:] = dot(diag(self.S), self.V.toArray().transpose()[:len(self.S),:])
        self.nextZeroRow = (len(self.S)-1)
        self.emptyRows += 1


   	# Shrink the approximate matrix using Parameterized FD
    def __PFDOperate__(self):
        #Shrink the sketch matrix
        # Calculating matrix s
        delta = self.S[-1]**2
        self.S[round(len(self.S)*(1-self.op)):] = sqrt(self.S[round(len(self.S)*(1-self.op)):]**2 - self.S[-1]**2)
        #Shrink the sketch matrix
        self.localSketchMatrix[:,:] = dot(diag(self.S), self.V.toArray().transpose()[:len(self.S),:])
        self.nextZeroRow = len(self.S) - 1
        self.emptyRows += 1
    

    # Shrink the approximate matrix using Space Saving Direction
    def __SSDOperate__(self):
        # Calculating matrix s
        self.S[-1] = sqrt(self.S[-1]**2 + self.S[-2]**2)
        self.S[len(self.S)-2] = 0
        #Shrink the sketch matrix
        self.localSketchMatrix[:,:] = dot(diag(self.S), self.V.toArray().transpose()[:len(self.S),:])
        self.nextZeroRow = len(self.S)-2
        self.emptyRows += 1
        
        
    # Shrink the approximate matrix using Compensative Direction
    def __CFDperate__(self):
        # Calculating SVD
        try:
            [U,self.S,Vt] = svd(self.sketchMatrix, full_matrices=False)
        except LinAlgError as err:
            [U,self.S,Vt] = scipy_svd(self.sketchMatrix, full_matrices = False)
        # Calculating matrix s
        self.DELTA += self.S[-1]**2
        self.S = sqrt(self.S[:len(self.S)]**2 + self.DELTA)
        delta = sqrt(self.S[:]**2 - self.S[len(self.S)-1]**2)
        self.S = delta
        self.S[len(self.S)-1] = 0
        #Shrink the sketch matrix
        self.localSketchMatrix[:,:] = dot(diag(self.S), self.V.toArray().transpose()[:len(self.S),:])
        self.nextZeroRow = len(self.S) - 1
        self.emptyRows += 1


    # Return the local sketch matrix
    def getLocalSketchMatrix(self):
        return self.localSketchMatrix
    
    # Return the distributed sketch matrix
    def getDistributedSketchMatrix(self):
        return self.distributedSketchMatrix
    
    # Return the S matrix
    def getS(self):
        return self.S
    
    # Return the U matrix
    def getU(self):
        return self.U
    
    # Return the Vt matrix
    def getV(self):
        return self.V