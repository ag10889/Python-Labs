
#*
#made by Alexander George



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

class subspace(matrix):
    # TODO
    # basis: a set of linearly independent vectors that span the subspace
    # dimension: the number of vectors in a basis for the subspace
    # rank - nullity theorem: rank(A) + nullity(A) = m
    # rank(A): dimension of the column space of A
    # nullity(A): dimension of the null space of A
    # rref: reduced row echelon form of A via Gaussian (Gauss-Jordan/Row Reduction) elimination
    def __init__(self, matrix):
        super().__init__(matrix)

    def rref(self):
        # Return a deep-copied RREF of the matrix using Gauss-Jordan elimination
        import copy
        A = copy.deepcopy(self.matrix)
        n = len(A)
        m = len(A[0])
        row = 0
        eps = 1e-12
        for col in range(m):
            if row >= n:
                break
            # find pivot
            pivot = None
            for r in range(row, n):
                if abs(A[r][col]) > eps:
                    pivot = r
                    break
            if pivot is None:
                continue
            # swap current row with pivot row
            A[row], A[pivot] = A[pivot], A[row]
            # normalize pivot row
            pv = A[row][col]
            A[row] = [val / pv for val in A[row]]
            # eliminate other rows
            for r in range(n):
                if r != row:
                    factor = A[r][col]
                    if abs(factor) > eps:
                        A[r] = [a - factor * b for a, b in zip(A[r], A[row])]
            row += 1
        # zero out near-zero entries
        for r in range(n):
            for c in range(m):
                if abs(A[r][c]) < eps:
                    A[r][c] = 0.0
        return A

    def rank(self):
        # Rank is the number of non-zero rows in RREF
        R = self.rref()
        eps = 1e-12
        rank = 0
        for row in R:
            if any(abs(x) > eps for x in row):
                rank += 1
        return rank

    def nullity(self):
        # Nullity = number of columns - rank
        return self.m - self.rank()

    
class Eigenvalues(matrix):
    # TODO 
    #eigenvalues are scalars λ such that Ax = λx for some non-zero vector x
    #eigenvalues are found by solving det(A - λI) = 0
    #eigenvalues are constant 
    #eigenvalues are used to find eigenvectors 
    #eigenvectors are found by solving (A - λI)x = 0
    #eigenvectors form a basis for the eigenspace corresponding to λ

    def calculate_eigenvalues(self):
        import cmath
        # Implement analytic eigenvalue calculation for 1x1, 2x2 and 3x3 matrices
        if self.n == 1:
            return [self.matrix[0][0]]
        if self.n == 2:
            a = self.matrix[0][0]
            b = self.matrix[0][1]
            c = self.matrix[1][0]
            d = self.matrix[1][1]
            trace = a + d
            det = a * d - b * c
            disc = trace * trace - 4 * det
            sqrt_disc = cmath.sqrt(disc)
            e1 = (trace + sqrt_disc) / 2
            e2 = (trace - sqrt_disc) / 2
            return [e1, e2]
        if self.n == 3:
            # coefficients of characteristic polynomial λ^3 - T λ^2 + S λ - D = 0
            a, b, c = self.matrix[0]
            d, e, f = self.matrix[1]
            g, h, i = self.matrix[2]
            T = a + e + i  # trace
            # sum of principal 2x2 minors
            S = (a*e - b*d) + (a*i - c*g) + (e*i - f*h)
            D = determinant(self.matrix).calculate()
            # convert to depressed cubic y^3 + p y + q = 0 with λ = y + T/3
            p = S - (T*T)/3
            q = (T*S)/3 - (2*(T**3))/27 - D
            disc = (q/2)**2 + (p/3)**3

            def croot(z):
                # real cube root for reals, principal for complex
                import cmath
                if isinstance(z, complex):
                    return z**(1/3)
                if z >= 0:
                    return z**(1/3)
                else:
                    return -((-z)**(1/3))

            roots = []
            if abs(disc) < 1e-14:
                disc = 0.0
            if disc > 0:
                # one real root and two complex
                A = -q/2 + cmath.sqrt(disc)
                B = -q/2 - cmath.sqrt(disc)
                u = croot(A)
                v = croot(B)
                y1 = u + v
                lam1 = y1 + T/3
                # other two
                omega = complex(-0.5, (3**0.5)/2)
                y2 = u*omega + v*omega.conjugate()
                y3 = u*omega.conjugate() + v*omega
                lam2 = y2 + T/3
                lam3 = y3 + T/3
                roots = [lam1, lam2, lam3]
            elif disc == 0:
                # multiple real roots
                A = croot(-q/2)
                y1 = 2*A
                y2 = -A
                lam1 = y1 + T/3
                lam2 = y2 + T/3
                lam3 = lam2
                roots = [lam1, lam2, lam3]
            else:
                # three distinct real roots
                import math
                rho = math.sqrt(- (p**3) / 27)
                # compute angle safely
                phi = math.acos(max(-1.0, min(1.0, (-q / 2) / rho)))
                m = 2 * math.sqrt(-p/3)
                y1 = m * math.cos(phi / 3)
                y2 = m * math.cos((phi + 2*math.pi) / 3)
                y3 = m * math.cos((phi + 4*math.pi) / 3)
                lam1 = y1 + T/3
                lam2 = y2 + T/3
                lam3 = y3 + T/3
                roots = [lam1, lam2, lam3]
            return roots
        raise NotImplementedError("Eigenvalue calculation only implemented for 1x1, 2x2 and 3x3 matrices")

    def calculate_eigenvectors(self):
        # Return eigenvectors corresponding to self.eigenvalues (implemented for 1x1, 2x2 and 3x3)
        import copy, cmath
        eps = 1e-10
        if self.n == 1:
            return [[1]]
        if self.n == 2:
            a = self.matrix[0][0]
            b = self.matrix[0][1]
            c = self.matrix[1][0]
            d = self.matrix[1][1]
            vectors = []
            for lam in self.eigenvalues:
                m11 = a - lam
                m12 = b
                m21 = c
                m22 = d - lam
                # Solve (m11 m12; m21 m22) [x y]^T = 0 by picking a nontrivial solution
                if abs(m12) > eps or abs(m11) > eps:
                    if abs(m12) > eps:
                        x = 1
                        y = -m11 / m12
                    else:
                        x = -m12 / m11 if abs(m11) > eps else 1
                        y = 1
                elif abs(m22) > eps or abs(m21) > eps:
                    if abs(m22) > eps:
                        x = -m12 / m22 if abs(m22) > eps else 1
                        y = 1
                    else:
                        x = 1
                        y = -m11 / m21 if abs(m21) > eps else 0
                else:
                    x, y = 1, 0
                vectors.append([x, y])
            return vectors
        if self.n == 3:
            # For each eigenvalue solve (A - λI) x = 0 using RREF to pick a nontrivial null vector
            vectors = []
            for lam in self.eigenvalues:
                # build M = A - lam*I
                M = [[self.matrix[r][c] - (lam if r == c else 0) for c in range(3)] for r in range(3)]
                # use subspace rref to row-reduce
                R = subspace(M).rref()
                # find pivot columns
                pivot_cols = {}
                for r in range(3):
                    for c in range(3):
                        if abs(R[r][c]) > eps:
                            pivot_cols[r] = c
                            break
                pivots = set(pivot_cols.values())
                free_cols = [c for c in range(3) if c not in pivots]
                if not free_cols:
                    # numerical edge: pick last column as free
                    free_cols = [2]
                free_col = free_cols[0]
                # construct solution vector x with free variable = 1
                x = [0+0j, 0+0j, 0+0j]
                x[free_col] = 1+0j
                # back-substitute rows from bottom up
                for r in reversed(range(3)):
                    if r in pivot_cols:
                        pc = pivot_cols[r]
                        s = 0+0j
                        for c in range(pc+1, 3):
                            s += R[r][c] * x[c]
                        # R[r][pc] should be 1 in RREF; but guard
                        denom = R[r][pc] if abs(R[r][pc]) > eps else 1+0j
                        x_val = -s / denom
                        x[pc] = x_val
                # convert small reals to float if possible
                def simplify(z):
                    if isinstance(z, complex) and abs(z.imag) < 1e-12:
                        return float(z.real)
                    return z
                vectors.append([simplify(val) for val in x])
            return vectors
        raise NotImplementedError("Eigenvector calculation only implemented for 1x1, 2x2 and 3x3 matrices")

    def __init__(self, matrix):
        self.matrix = matrix
        self.n = len(matrix)
        self.m = len(matrix[0])
        if self.n != self.m:
            raise ValueError("Matrix must be square")
        self.eigenvalues = self.calculate_eigenvalues()
        self.eigenvectors = self.calculate_eigenvectors() 

class diagonalizeMatrix(Eigenvalues):
    # TODO
    # A is diagonalizable if there exists an invertible matrix P and a diagonal matrix D such that A = PDP^-1
    # A is diagonalizable if it has n linearly independent eigenvectors
    def __init__(self, matrix):
        # initialize parent to compute eigenvalues and eigenvectors
        super().__init__(matrix)

        # only implemented for 2x2 and 3x3 in this codebase
        if self.n not in (2, 3):
            raise NotImplementedError("Diagonalization only implemented for 2x2 and 3x3 matrices")

        # Build P matrix whose columns are the eigenvectors
        # eigenvectors are returned as list-of-lists, one per eigenvalue
        P = [[self.eigenvectors[col][row] for col in range(self.n)] for row in range(self.n)]

        # Check that P is invertible (i.e. columns are linearly independent)
        if subspace(P).rank() != self.n:
            raise ValueError("Matrix is not diagonalizable: eigenvectors are not linearly independent")

        # Build diagonal matrix D of eigenvalues
        D = [[self.eigenvalues[i] if i == j else 0 for j in range(self.n)] for i in range(self.n)]

        # helper: matrix multiplication
        def matmul(A, B):
            rA = len(A)
            cA = len(A[0])
            rB = len(B)
            cB = len(B[0])
            if cA != rB:
                raise ValueError("Incompatible dimensions for multiplication")
            C = [[0 for _ in range(cB)] for _ in range(rA)]
            for i in range(rA):
                for j in range(cB):
                    s = 0
                    for k in range(cA):
                        s += A[i][k] * B[k][j]
                    C[i][j] = s
            return C

        # compute P^-1 (using inverse class which supports 2x2 and 3x3)
        if self.n == 2:
            P_inv = inverse(P).inverse_2x2()
        else:
            P_inv = inverse(P).inverse_3x3()

        # Optionally verify reconstruction A == P * D * P_inv (numerical errors possible)
        # reconstructed = matmul(matmul(P, D), P_inv)

        # Store diagonal matrix as the representation of the diagonalized form
        self.P = P
        self.P_inv = P_inv
        self.D = D
        self.matrix = D
       
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
        print("10 - Diagonalize Matrix (selected matrix)")
        print("11 - Find Eigenvalues and Eigenvectors (selected matrix)")
        print("12 - exit")

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
                if self.current_matrix is None:
                    print("No matrix selected. Use option 4 to select a matrix.")
                    continue
                matrix = self.matrices[self.current_matrix]
                try:
                    diag_matrix = diagonalizeMatrix(matrix)
                    print(f"Diagonalized form of '{self.current_matrix}':")
                    for row in diag_matrix.matrix:
                        print(row)
                except ValueError as e:
                    print(e)
            elif choice == '11':
                if self.current_matrix is None:
                    print("No matrix selected. Use option 4 to select a matrix.")
                    continue
                matrix = self.matrices[self.current_matrix]
                try:
                    eigen = Eigenvalues(matrix)
                    print(f"Eigenvalues of '{self.current_matrix}': {eigen.eigenvalues}")
                    print(f"Eigenvectors of '{self.current_matrix}':")
                    for row in eigen.eigenvectors:
                        print(row)
                except ValueError as e:
                    print(e)
            elif choice == '12':
                print("Exiting program.")
                break
            else:
                print("Invalid choice. Please try again.")
        
if __name__ == "__main__":
    main().run()
