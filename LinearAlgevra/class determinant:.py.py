class determinant:
    # TODO
    # invertible matrix theorem: A is invertible <=> det(A) != 0
    # det(AB) = det(A) * det(B)
    # det(A^T) = det(A)
    # det(A^-1) = 1/det(A)

    def __init__(self, matrix):
        self.matrix = matrix
        self.n = len(matrix)
        self.m = len(matrix[0])
        if self.n != self.m:
            raise ValueError("Matrix must be square")

    def calculate(self):
        if self.n == 1:
            return self.matrix[0][0]
        if self.n == 2:
            return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]
        det = 0
        for c in range(self.n):
            minor = [row[:c] + row[c+1:] for row in self.matrix[1:]]
            det += ((-1) ** c) * self.matrix[0][c] * determinant(minor).calculate()
        return det
    

class inverse:
    # TODO 
    # A is invertible <=> det(A) != 0
    # (AB)^-1 = B^-1 * A^-1
    # (A^T)^-1 = (A^-1)^T
    # (A^-1)^-1 = A
    # A * A^-1 = I
    def __init__(self, matrix):
        self.matrix = matrix
        self.n = len(matrix)
        self.m = len(matrix[0])
        if self.n != self.m:
            raise ValueError("Matrix must be square")
        if determinant(matrix).calculate() == 0:
            raise ValueError("Matrix is not invertible")
    def inverse_2x2(self):
        if self.n != 2:
            raise ValueError("Matrix must be 2x2")
        det = determinant(self.matrix).calculate()
        return [[self.matrix[1][1]/det, -self.matrix[0][1]/det],
                [-self.matrix[1][0]/det, self.matrix[0][0]/det]]
    def inverse_3x3(self):
        if self.n != 3:
            raise ValueError("Matrix must be 3x3")
        det = determinant(self.matrix).calculate()
        if det == 0:
            raise ValueError("Matrix is not invertible")

        # Build the cofactor matrix
        cofactors = []
        for r in range(3):
            cof_row = []
            for c in range(3):
                minor = [row[:c] + row[c+1:] for i, row in enumerate(self.matrix) if i != r]
                cofactor = ((-1) ** (r + c)) * determinant(minor).calculate()
                cof_row.append(cofactor)
            cofactors.append(cof_row)

        # Adjugate is the transpose of the cofactor matrix
        adjugate = [[cofactors[r][c] for r in range(3)] for c in range(3)]

        # Divide by determinant to get the inverse
        inverse_matrix = [[adjugate[r][c] / det for c in range(3)] for r in range(3)]
        return inverse_matrix
        

class matrix: 
    # TODO 
    # A is n x m, n rows by m columns 
    # the m columns of A are vectors in R^n
    # the n rows of A are vectors in R^m
    # the nullity of A is a subpace of R^m
    # the column space of A is a subspace of R^n
    # the row space of A is a subspace of R^m
    # rank(A) + nullity(A) = m
    def __init__(self, matrix):
        self.matrix = matrix
        self.n = len(matrix)
        self.m = len(matrix[0])
        if self.n != self.m:
            raise ValueError("Matrix must be square")

class main:
    def __init__(self):
        self.matrix = None

    def print_menu(self):
        print("1 - input matrix size")
        print("2 - input matrix values:")
        print("3 - calculate determinant")
        print("4 - calculate inverse")
        print("5 - exit")

    def run(self):
        while True:
            self.print_menu()
            choice = input("Enter your choice: ")
            if choice == '1':
                n = int(input("Enter number of rows: "))
                m = int(input("Enter number of columns: "))
                self.matrix = [[0 for _ in range(m)] for _ in range(n)]
            elif choice == '2':
                if self.matrix is None:
                    print("Please input matrix size first.")
                    continue
                for i in range(len(self.matrix)):
                    row = list(map(float, input(f"Enter row {i+1} values separated by space: ").split()))
                    if len(row) != len(self.matrix[0]):
                        print("Incorrect number of values. Please try again.")
                        break
                    self.matrix[i] = row
            elif choice == '3':
                if self.matrix is None:
                    print("Please input matrix size and values first.")
                    continue
                try:
                    det = determinant(self.matrix).calculate()
                    print(f"Determinant: {det}")
                except ValueError as e:
                    print(e)
            elif choice == '4':
                if self.matrix is None:
                    print("Please input matrix size and values first.")
                    continue
                try:
                    inv = inverse(self.matrix)
                    if inv.n == 2:
                        inv_matrix = inv.inverse_2x2()
                    elif inv.n == 3:
                        inv_matrix = inv.inverse_3x3()
                    else:
                        print("Inverse calculation only implemented for 2x2 and 3x3 matrices.")
                        continue
                    print("Inverse:")
                    for row in inv_matrix:
                        print(row)
                except ValueError as e:
                    print(e)
            elif choice == '5':
                break
            else:
                print("Invalid choice. Please try again.")

        print("Exiting program.")
        
if __name__ == "__main__":
    main().run()