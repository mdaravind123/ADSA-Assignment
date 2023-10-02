def solve_n_queens(n):
    def is_safe(board, row, col):
        for i in range(col):
            if board[row][i] == 1:
                return False
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        for i, j in zip(range(row, n), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        
        return True

    def solve(board, col):
        if col == n:
            solutions.append(["".join("Q" if cell == 1 else "." for cell in row) for row in board])
            return

        for i in range(n):
            if is_safe(board, i, col):
                board[i][col] = 1
                solve(board, col + 1)
                board[i][col] = 0  

    solutions = []
    chessboard = [[0] * n for _ in range(n)]
    solve(chessboard, 0)
    return solutions
  
n = 4
queens_solutions = solve_n_queens(n)


for i, solution in enumerate(queens_solutions):
    print(f"Solution {i + 1}:")
    for row in solution:
        print(row)
    print("\n")
