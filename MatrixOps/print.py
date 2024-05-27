def printMatrix(matrix):
    """
    matrix: matrix
    Prints a matrix with is the input
    Void function, returns nothing
    """
    for row in matrix:
        try:
            temp = [str(int(num)) for num in row]  #Just makes it look better
        except ValueError:
            temp = [str((num)) for num in row]
            print("Are you sure you typed in your matrix correctly, there are decimals.")
        print("[" + ' '.join(temp) + "]")