from bayes_opt import BayesianOptimization
from GATSP2 import TSP

def _TSP(c_rate, m_rate, pop_size):     # 优化目标函数
    tsp = TSP(c_rate, m_rate, pop_size)
    return -tsp.best_dist

if __name__ == "__main__":
    # c_rate = 0.3986     # 交叉阈值     0.4075 0.3986 
    # m_rate = 0.253    # 突变阈值    0.4345 0.253
    # pop_size = 83.73  # 种群大小    84.64 83.73 
    iteration = 1200 # 迭代次数
    seed = 2023     # 随机种子
    # tsp = TSP(c_rate, m_rate, pop_size, iteration, seed)
    # print(tsp.best_gene)

    rf_bo = BayesianOptimization(
        _TSP,
            {
                'c_rate': (0.33,0.42),
                'm_rate': (0.18,0.45),
                'pop_size': (80,88),
            }
    )

    best_params = rf_bo.maximize()  # 获取最优参数组合
    print("Best parameters found:", best_params)