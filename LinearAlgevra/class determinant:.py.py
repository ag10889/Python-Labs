#*
#made by Alexander George
#6/12/2024 (10/15/25)
#scary how you knows my name (sorry broski)
#can you make it so this is multiline comment? 

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
    
class adjugate:
    # TODO
    # adj(A) = det(A) * A^-1
    def __init__(self, matrix):
        self.matrix = matrix
        self.n = len(matrix)
        self.m = len(matrix[0])
        if self.n != self.m:
            raise ValueError("Matrix must be square")
        if determinant(matrix).calculate() == 0:
            raise ValueError("Matrix is not invertible")
    def calculate(self):
        cofactors = []
        for r in range(self.n):
            cof_row = []
            for c in range(self.n):
                minor = [row[:c] + row[c+1:] for i, row in enumerate(self.matrix) if i != r]
                cofactor = ((-1) ** (r + c)) * determinant(minor).calculate()
                cof_row.append(cofactor)
            cofactors.append(cof_row)
        adjugate = [[cofactors[r][c] for r in range(self.n)] for c in range(self.n)]
        return adjugate
        
class cramers_rule:
    # TODO
    # only for n x n matrices
    # Ax = b, where A is n x n, x is n x 1, b is n x 1
    # x_i = det(A_i) / det(A), where A_i is A with column i replaced by b
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.n = len(A)
        self.m = len(A[0])
        if self.n != self.m:
            raise ValueError("Matrix A must be square")
        if len(b) != self.n:
            raise ValueError("Vector b must have the same number of rows as A")
        if determinant(A).calculate() == 0:
            raise ValueError("Matrix A is not invertible")

    def solve(self):
        det_A = determinant(self.A).calculate()
        solution = []
        for i in range(self.n):
            A_i = [row[:] for row in self.A]
            for j in range(self.n):
                A_i[j][i] = self.b[j]
            det_A_i = determinant(A_i).calculate()
            solution.append(det_A_i / det_A)
        return solution

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
        self.matrices = {}  # Dictionary to store matrices by name
        self.current_matrix = None  # Name of the currently selected matrix

    def print_menu(self):
        print("1 - create new matrix")
        print("2 - input matrix values")
        print("3 - list saved matrices")
        print("4 - select matrix for calculation")
        print("5 - calculate determinant of selected matrix")
        print("6 - calculate inverse of selected matrix")
        print("7 - calculate adjugate of selected matrix")
        print("8 - import vector b for Ax = b (selected matrix)")
        print("9 - solve linear system (Cramer's Rule) (selected matrix)")
        print("10 - exit")

    def run(self):
        while True:
            self.print_menu()
            choice = input("Enter your choice: ")
            if choice == '1':
                name = input("Enter a name for this matrix: ")
                n = int(input("Enter number of rows: "))
                m = int(input("Enter number of columns: "))
                self.matrices[name] = [[0 for _ in range(m)] for _ in range(n)]
                print(f"Matrix '{name}' created. Use option 2 to input values.")
            elif choice == '2':
                name = input("Enter the name of the matrix to input values for: ")
                if name not in self.matrices:
                    print("Matrix not found. Please create it first.")
                    continue
                for i in range(len(self.matrices[name])):
                    row = list(map(float, input(f"Enter row {i+1} values separated by space: ").split()))
                    if len(row) != len(self.matrices[name][0]):
                        print("Incorrect number of values. Please try again.")
                        break
                    self.matrices[name][i] = row
                print(f"Values updated for matrix '{name}'.")
            elif choice == '3':
                if not self.matrices:
                    print("No matrices saved.")
                else:
                    print("Saved matrices:")
                    for name in self.matrices:
                        print(f"- {name}")
            elif choice == '4':
                name = input("Enter the name of the matrix to select: ")
                if name not in self.matrices:
                    print("Matrix not found.")
                    continue
                self.current_matrix = name
                print(f"Matrix '{name}' is now selected for calculations.")
            elif choice == '5':
                if self.current_matrix is None:
                    print("No matrix selected. Use option 4 to select a matrix.")
                    continue
                matrix = self.matrices[self.current_matrix]
                try:
                    det = determinant(matrix).calculate()
                    print(f"Determinant of '{self.current_matrix}': {det}")
                except ValueError as e:
                    print(e)
            elif choice == '6':
                if self.current_matrix is None:
                    print("No matrix selected. Use option 4 to select a matrix.")
                    continue
                matrix = self.matrices[self.current_matrix]
                try:
                    inv = inverse(matrix)
                    if inv.n == 2:
                        inv_matrix = inv.inverse_2x2()
                    elif inv.n == 3:
                        inv_matrix = inv.inverse_3x3()
                    else:
                        print("Inverse calculation only implemented for 2x2 and 3x3 matrices.")
                        continue
                    print(f"Inverse of '{self.current_matrix}':")
                    for row in inv_matrix:
                        print(row)
                except ValueError as e:
                    print(e)
            elif choice == '7':
                if self.current_matrix is None:
                    print("No matrix selected. Use option 4 to select a matrix.")
                    continue
                matrix = self.matrices[self.current_matrix]
                try:
                    adj = adjugate(matrix).calculate()
                    print(f"Adjugate of '{self.current_matrix}':")
                    for row in adj:
                        print(row)
                except ValueError as e:
                    print(e)
            elif choice == '8':
                if self.current_matrix is None:
                    print("No matrix selected. Use option 4 to select a matrix.")
                    continue
                matrix = self.matrices[self.current_matrix]
                b = list(map(float, input("Enter vector b values separated by space: ").split()))
                try:
                    solution = cramers_rule(matrix, b).solve()
                    print("Solution:")
                    for i, x in enumerate(solution):
                        print(f"x_{i+1} = {x}")
                except ValueError as e:
                    print(e)
            elif choice == '9':
                if self.current_matrix is None:
                    print("No matrix selected. Use option 4 to select a matrix.")
                    continue
                matrix = self.matrices[self.current_matrix]
                b = list(map(float, input("Enter vector b values separated by space: ").split()))
                try:
                    solution = cramers_rule(matrix, b).solve()
                    print("Solution:")
                    for i, x in enumerate(solution):
                        print(f"x_{i+1} = {x}")
                except ValueError as e:
                    print(e)
            elif choice == '10':
                print("Exiting program.")
                break
            else:
                print("Invalid choice. Please try again.")
        
if __name__ == "__main__":
    main().run()