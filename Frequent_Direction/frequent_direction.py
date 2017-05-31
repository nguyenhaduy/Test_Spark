
# coding: utf-8

# In[1]:


from numpy import zeros, max, sqrt, isnan, isinf, dot, diag, count_nonzero
from numpy.linalg import svd, linalg, LinAlgError, norm
from scipy.linalg import svd as scipy_svd
#from scipy.sparse.linalg import svds as scipy_svds


# In[2]:


class FrequentDirections():

    def __init__(self , rows, columns):
        self.class_name = 'FrequentDirections'
        self.columns = columns
        self.rows = rows
        self.sketchMatrix = zeros((self.rows, self.columns)) 
        self.nextZeroRow = 0
    
    # Add new vector to the sketch matrix
    def add(self,vector):     
        if count_nonzero(vector) == 0:
            return
        
        # If the approximate matrix is full, call the operate method to free half of the columns
        if self.nextZeroRow >= self.rows:
            self.__operate__()

        # Push the new vector to the next zero row and increase the next zero row index
        self.sketchMatrix[self.nextZeroRow,:] = vector 
        self.nextZeroRow += 1


    # Shrink the approximate matrix
    def __operate__(self):
        # Calculating SVD
        [U,s,Vt] = svd(self.sketchMatrix , full_matrices=False)
        #Shrink the sketch matrix
        self.sketchMatrix[:len(s),:] = dot(diag(s), Vt[:len(s),:])
        self.sketchMatrix[int(len(s)/2):,:] = 0
        self.nextZeroRow = int(len(s)/2)
        
    # Return the sketch matrix
    def get(self):
        return self.sketchMatrix[:self.rows,:]

