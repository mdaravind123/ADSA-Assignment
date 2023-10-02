import sys

def matrix_chain_order(dimensions):
    n = len(dimensions) - 1  
    m = [[0] * n for _ in range(n)]  
    s = [[0] * n for _ in range(n)]  

    for chain_length in range(2, n + 1):
        for i in range(n - chain_length + 1):
            j = i + chain_length - 1
            m[i][j] = sys.maxsize  

            for k in range(i, j):
                cost = m[i][k] + m[k + 1][j] + dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k

    return m, s

def print_optimal_parenthesization(s, i, j):
    if i == j:
        print(f'M{str(i)}', end='')
    else:
        print('(', end='')
        print_optimal_parenthesization(s, i, s[i][j])
        print_optimal_parenthesization(s, s[i][j] + 1, j)
        print(')', end='')


matrix_dimensions = [10, 30, 5, 60]
m_table, s_table = matrix_chain_order(matrix_dimensions)
print(f"Minimum scalar multiplications: {m_table[0][-1]}")
print("Optimal Parenthesization: ", end='')
print_optimal_parenthesization(s_table, 0, len(matrix_dimensions) - 2)


#Time and Space Complexity:
#The time complexity of this algorithm is O(n^3), where n is the number of matrices. The space complexity is O(n^2) due to the m and s matrices.