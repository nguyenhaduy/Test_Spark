from numpy import zeros, max, sqrt, isnan, isinf, dot, diag, count_nonzero, where
from numpy.linalg import svd, linalg, LinAlgError, norm
from scipy.linalg import svd as scipy_svd


class FrequentDirections(object):
    
    def __init__(self , rows, columns, op='fd'):
        """
		Matrix Sketching using Frequent Direction. Choose:
		'fd' for normal Frequent Direction;
		'ssd' for Space Saving Direction; 
		'isvd' for iterative SVD;
		and a number between 0 and 1 for Parameterized Frequent Direction
        """
        self.class_name = 'FrequentDirections'
        self.op = op
        self.columns = columns
        self.rows = rows
        self.sketchMatrix = zeros((self.rows, self.columns)) 
        self.S = zeros(self.columns)
        self.U = []
        self.Vt = []
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
            try:
                [self.U,self.S,self.Vt] = svd(self.sketchMatrix , full_matrices=True)
            except LinAlgError as err:
                [self.U,self.S,self.Vt] = scipy_svd(self.sketchMatrix , full_matrices = True)
            self.reduceRank()

        # Push the new vector to the next zero row and increase the next zero row index
        self.sketchMatrix[self.nextZeroRow,:] = vector
        self.nextZeroRow += 1
        self.emptyRows -= 1


    # Shrink the approximate matrix using Frequent Direction
    def __FDOperate__(self):
        # Calculating matrix s
        delta = sqrt(self.S[:]**2 - self.S[len(self.S)-1]**2)
        self.S = delta
        self.S[len(self.S)-1] = 0
        #Shrink the sketch matrix
        self.sketchMatrix[:,:] = dot(diag(self.S), self.Vt[:self.rows,:])
        self.nextZeroRow = (len(self.S)-1)
        self.emptyRows += 1
            
        
    # Shrink the approximate matrix using iterative SVD
    def __iSVDOperate__(self):
        # Calculating matrix s
        self.S[len(self.S)-1] = 0
        #Shrink the sketch matrix
        self.sketchMatrix[:len(self.S),:] = dot(diag(self.S), self.Vt[:len(self.S),:])
        self.nextZeroRow = (len(self.S)-1)
        self.emptyRows += 1


   	# Shrink the approximate matrix using Parameterized FD
    def __PFDOperate__(self):
        # Calculating matrix s
        delta = self.S[-1]**2
        self.S[round(len(self.S)*(1-self.op)):] = sqrt(self.S[round(len(self.S)*(1-self.op)):]**2 - self.S[-1]**2)
        #Shrink the sketch matrix
        self.sketchMatrix[:,:] = dot(diag(self.S), self.Vt[:self.rows,:])
        self.nextZeroRow = len(self.S) - 1
        self.emptyRows += 1

    # Shrink the approximate matrix using Space Saving Direction
    def __SSDOperate__(self):
        # Calculating matrix s
        self.S[-1] = sqrt(self.S[-1]**2 + self.S[-2]**2)
        self.S[len(self.S)-2] = 0
        #Shrink the sketch matrix
        self.sketchMatrix[:len(self.S),:] = dot(diag(self.S), self.Vt[:len(self.S),:])
        self.nextZeroRow = len(self.S)-2
        self.emptyRows += 1
        

    # Return the sketch matrix
    def getSketchMatrix(self):
        return self.sketchMatrix[:self.rows,:]
    
    # Return the S matrix
    def getS(self):
        return self.S
    
    # Return the U matrix
    def getU(self):
        return self.U
    
    # Return the Vt matrix
    def getVt(self):
        return self.Vt
