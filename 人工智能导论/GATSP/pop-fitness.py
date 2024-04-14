"""
    本文件用于绘制种群数量-种群适应度曲线
"""
from matplotlib import pyplot as plt

from GATSP import TSP

if __name__ == "__main__":
    c_rate = 0.3986  # 交叉阈值     0.4075 0.3986
    m_rate = 0.253  # 突变阈值    0.4345 0.253
    iteration = 1200  # 迭代次数
    seed = 2023  # 随机种子
    pop_size = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    fitness = []

    for item in pop_size:
        tsp = TSP(c_rate, m_rate, item)
        fitness.append(tsp.best_dist)

    # 绘制种群数量-种群适应度曲线
    plt.plot(pop_size, fitness)
    plt.xlabel("pop_size")
    plt.ylabel("fitness")
    plt.title("pop_size-fitness")
    plt.savefig("img/种群数量-种群适应度曲线.png")
