from elemRowOps import *
from print import printMatrix

def inverse(matrix):
    """
    Finds an inverse
    """
    numRows = len(matrix)
    numColumns = len(matrix[0])

    if numRows != numColumns:
        print("This matrix is not square and does not have an inverse")
        return
    
    ident = identityMatrix(numRows)

    i = 0    #row pos
    j = 0    #col pos
    while i < numRows and j < numColumns:
        if matrix[i][j] == 0:              #if current val = 0, check if ones below it aren't 0
            tempi = i
            while (tempi < numRows):            #checks if there is a swap to make, then makes it, new row is now in pos i
                if matrix[tempi][j] != 0:
                    swapRow(i, tempi, matrix)
                    swapRow(i, tempi, ident)
                    break
                else:
                    tempi += 1 
            if matrix[i][j] != 0:                           #if the row isnt 0, scales it
                 scaleRow(1 / (matrix[i][j]), i, matrix)
                 scaleRow(1 / (matrix[i][j]), i, ident)
        else: 
            scaleRow(1 / (matrix[i][j]), i, matrix) 
            scaleRow(1 / (matrix[i][j]), i, ident)       #scales row if there was no swap needed
        
        tempi = i + 1                          #scales row below it by multiples of 1
        while (tempi < numRows):
            if (matrix[tempi][j] != 0):
                addRow(i, tempi, -matrix[tempi][j], matrix)
                addRow(i, tempi, -matrix[tempi][j], ident)
            tempi += 1
        
        if matrix[i][j] == 1:
            i += 1
        j += 1

    if matrix.equals(identityMatrix(numRows)):
        return ident
    
    else:
        print("No inverse")
        return