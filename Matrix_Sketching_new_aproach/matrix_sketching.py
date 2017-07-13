from numpy import zeros, max, sqrt, isnan, isinf, dot, diag, count_nonzero, where, random
from sklearn.utils.extmath import randomized_svd


class MatrixSketching(object):
    
    def __init__(self , rows, columns, op='fd'):
        """
		Matrix Sketching library. Choose:
		'fd' for normal Frequent Direction;
		'ssd' for Space Saving Direction; 
		'isvd' for iterative SVD;
		'random' for Random Reduce
		and a number between 0 and 1 for Parameterized Frequent Direction
        """
        self.class_name = 'MatrixSketching'
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
        elif self.op == 'random':
            print("Matrix Sketching Using Random Reduce")
            self.random_matrix = random.randn(int(self.rows/2),self.rows)*(1/(self.rows)**2)
            self.reduceRank = self.__randomOperate__
        else:
            print("Type of Reduce Rank algorithm is not correct")
            raise ValueError
    
    # Add new vector to the sketch matrix
    def add(self,vector):     
        if count_nonzero(vector) == 0:
            return
        
        # If the approximate matrix is full, call the operate method to free half of the columns
        if self.emptyRows <= 0:            
            self.reduceRank()

        # Push the new vector to the next zero row and increase the next zero row index
        self.sketchMatrix[self.nextZeroRow,:] = vector
        self.nextZeroRow += 1
        self.emptyRows -= 1


    # Shrink the approximate matrix using Frequent Direction
    def __FDOperate__(self):
        [self.U,self.S,self.Vt] = randomized_svd(self.sketchMatrix,n_components=self.rows,n_iter=1,random_state=None)
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
        [self.U,self.S,self.Vt] = randomized_svd(self.sketchMatrix,n_components=self.rows,n_iter=1,random_state=None)
        # Calculating matrix s
        self.S[len(self.S)-1] = 0
        #Shrink the sketch matrix
        self.sketchMatrix[:len(self.S),:] = dot(diag(self.S), self.Vt[:len(self.S),:])
        self.nextZeroRow = (len(self.S)-1)
        self.emptyRows += 1


   	# Shrink the approximate matrix using Parameterized FD
    def __PFDOperate__(self):
        [self.U,self.S,self.Vt] = randomized_svd(self.sketchMatrix,n_components=self.rows,n_iter=1,random_state=None)
        # Calculating matrix s
        delta = self.S[-1]**2
        self.S[round(len(self.S)*(1-self.op)):] = sqrt(self.S[round(len(self.S)*(1-self.op)):]**2 - self.S[-1]**2)
        #Shrink the sketch matrix
        self.sketchMatrix[:,:] = dot(diag(self.S), self.Vt[:self.rows,:])
        self.nextZeroRow = len(self.S) - 1
        self.emptyRows += 1

    # Shrink the approximate matrix using Space Saving Direction
    def __SSDOperate__(self):
        [self.U,self.S,self.Vt] = randomized_svd(self.sketchMatrix,n_components=self.rows,n_iter=1,random_state=None)
        # Calculating matrix s
        self.S[-1] = sqrt(self.S[-1]**2 + self.S[-2]**2)
        self.S[len(self.S)-2] = 0
        #Shrink the sketch matrix
        self.sketchMatrix[:len(self.S),:] = dot(diag(self.S), self.Vt[:len(self.S),:])
        self.nextZeroRow = len(self.S)-2
        self.emptyRows += 1

	# Shrink the approximate matrix using Parameterized FD
    def __randomOperate__(self):
        # Creating a random matrix
        #random_matrix = random.randn(int(self.rows/2),self.rows)*(1/(self.rows)**2)
        #Shrink the sketch matrix
        self.sketchMatrix[:int(self.rows/2),:] = self.random_matrix.dot(self.sketchMatrix)
        self.nextZeroRow = self.rows - int(self.rows/2)
        self.emptyRows += int(self.rows/2)
        

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
