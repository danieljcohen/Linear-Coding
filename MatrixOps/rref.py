from elemRowOps import *
from print import *

def GaussJordan(matrix):
    """
    Turns a matrix into its rref using
    the GaussJordan algorithm
    Returns a matrix
    """
    numRows = len(matrix)
    numColumns = len(matrix[0])
    i = 0    #row pos
    j = 0    #col pos
    while i < numRows and j < numColumns:
        if matrix[i][j] == 0:              #if current val = 0, check if ones below it aren't 0
            tempi = i
            while (tempi < numRows):            #checks if there is a swap to make, then makes it, new row is now in pos i
                if matrix[tempi][j] != 0:
                    swapRow(i, tempi, matrix)
                    break
                else:
                    tempi += 1 
            if matrix[i][j] != 0:                           #if the row isnt 0, scales it
                 scaleRow(1 / (matrix[i][j]), i, matrix)
        else: 
            scaleRow(1 / (matrix[i][j]), i, matrix)        #scales row if there was no swap needed
        
        tempi = 0                          #scales row below it by multiples of 1
        while (tempi < numRows):
            if (matrix[tempi][j] != 0 and tempi != i):
                addRow(i, tempi, -matrix[tempi][j], matrix)
            tempi += 1
        
        if matrix[i][j] == 1:
            i += 1
        j += 1
        

    return matrix

        




