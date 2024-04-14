"""
回溯法求解八皇后问题
"""


class SolveNQueen(object):
    def __init__(self, n, filename):
        self.count = 0
        self.file = open(filename, "a")
        self.backtracking([-1] * n, 0, n)
        self.file.close()

    def backtracking(self, columnPositions, rowIndex, n):  #  回溯函数
        if rowIndex == n:  # 已经放置完成了，需要打印结果
            self.printSolution(columnPositions, n)
            self.count += 1
            return
        for column in range(n):  # 当前行每一列都尝试是否能摆放棋子
            columnPositions[rowIndex] = column
            if self.isValid(columnPositions, rowIndex):
                self.backtracking(columnPositions, rowIndex + 1, n)

    def isValid(self, columnPositions, rowIndex):
        for i in range(rowIndex):
            if columnPositions[i] == columnPositions[rowIndex]:
                return False
            elif abs(columnPositions[i] - columnPositions[rowIndex]) == rowIndex - i:
                return False
        return True

    def printSolution(self, columnPositions, n):  # 绘图
        self.file.write(f"{self.getCount()+1}\n")
        for row in range(n):
            line = ""
            for column in range(n):
                if columnPositions[row] == column:
                    line += "# "
                else:
                    line += "0 "
            self.file.write(line + "\n")  # 写入当前行并在末尾添加换行符
        self.file.write("\n")

    def getCount(self):
        return self.count


if __name__ == "__main__":
    solve = SolveNQueen(8, filename="./solve8queen.txt")
    print(f"一共有{solve.getCount()}种解法")
