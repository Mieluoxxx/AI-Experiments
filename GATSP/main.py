# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt

class TSP(object):
    def __init__(self, c_rate, m_rate, pop_size, iteration=500, seed=2023):
        self.cities = np.array([])  # 城市数组
        self.cities_name = np.array([])
        self.city_size = -1  # 标记城市数目
        self.pop_size = int(pop_size)  # 种群大小
        self.fitness = np.zeros(self.pop_size)  # 种群适应度
        self.c_rate = c_rate  # 交叉阈值
        self.m_rate = m_rate  # 突变阈值
        self.iteration = iteration  # 迭代次数
        self.best_dist = -1  # 最优距离
        self.best_gene = [] # 最优路径
        np.random.seed(seed)  # 随机种子

        self.init()  # 初始化
        self.evolution()  # 进化
        self.draw()  # 绘制

    def init(self):
        self.data = pd.read_csv("eil51.txt", delimiter=" ", header=None).values
        self.cities = self.data[:, 1:]
        self.cities_name = self.data[:, 0]
        self.city_size = self.data.shape[0]
        self.pop = self.create_pop(self.pop_size)  # 创建种群
        self.fitness = self.get_fitness(self.pop)  # 计算初始种群适应度

    def evolution(self):
        # 主程序：迭代进化种群
        writer = SummaryWriter()
        for i in range(self.iteration):
            best_f_index = np.argmax(self.fitness)
            worst_f_index = np.argmin(self.fitness)
            local_best_gene = self.pop[best_f_index]
            local_best_dist = self.gen_distance(local_best_gene)
            if i == 0:
                self.best_gene = local_best_gene
                self.best_dist = self.gen_distance(local_best_gene)

            if local_best_dist < self.best_dist:
                self.best_dist = local_best_dist  # 记录最优值
                self.best_gene = local_best_gene  # 记录最个体基因

            else:
                self.pop[worst_f_index] = self.best_gene
            print("gen:%d evo,best dist :%s" % (i, self.best_dist))

            self.pop = self.select_pop4(self.pop)  # 选择淘汰种群
            self.fitness = self.get_fitness(self.pop)  # 计算种群适应度
            for j in range(self.pop_size):
                r = np.random.randint(0, self.pop_size - 1)
                if j != r:
                    self.pop[j] = self.cross(self.pop[j], self.pop[r])  # 交叉种群中第j,r个体的基因
                    self.pop[j] = self.mutate(self.pop[j])  # 突变种群中第j个体的基因
            self.best_gene = self.EO(self.best_gene)  # 极值优化，防止收敛局部最优
            self.best_dist = self.gen_distance(self.best_gene)  # 记录最优值
            writer.add_scalar("fitness", self.best_dist, i)
        writer.close()

    def create_pop(self, size):
        pop = [np.random.permutation(self.city_size) for _ in range(size)]
        return np.array(pop)

    def get_fitness(self, pop):
        d = np.array([])  # 适应度记录数组
        for i in range(pop.shape[0]):
            gen = pop[i]  # 取其中一条基因（编码解，个体）
            dis = self.gen_distance(gen)  # 计算此基因优劣（距离长短）
            dis = self.best_dist / dis  # 当前最优距离除以当前pop[i]（个体）距离；越近适应度越高，最优适应度为1
            d = np.append(d, dis)  # 保存适应度pop[i]
        return d

    def get_local_fitness(self, gen, i):
        """
        计算地i个城市的邻域
        交换基因数组中任意两个值组成的解集：称为邻域。计算领域内所有可能的适应度
        :param gen:城市路径
        :param i:第i城市
        :return:第i城市的局部适应度
        """
        di = 0
        fi = 0
        if i == 0:
            di = self.ct_distance(self.cities[gen[0]], self.cities[gen[-1]])
        else:
            di = self.ct_distance(self.cities[gen[i]], self.cities[gen[i - 1]])
        od = []
        for j in range(self.city_size):
            if i != j:
                od.append(
                    self.ct_distance(self.cities[gen[i]], self.cities[gen[i - 1]])
                )
        mind = np.min(od)
        fi = di - mind
        return fi

    def EO(self, gen):
        # 极值优化，传统遗传算法性能不好，这里混合EO
        # 其会在整个基因的领域内，寻找一个最佳变换以更新基因
        local_fitness = np.zeros(self.city_size)
        for g in range(self.city_size):
            local_fitness[g] = self.get_local_fitness(gen, g)
        max_city_i = np.argmax(local_fitness)
        maxgen = np.copy(gen)
        if 1 < max_city_i < self.city_size - 1:
            for j in range(max_city_i):
                maxgen = np.copy(gen)
                jj = max_city_i
                while jj < self.city_size:
                    gen1 = self.exechange_gen(maxgen, j, jj)
                    d = self.gen_distance(maxgen)
                    d1 = self.gen_distance(gen1)
                    if d > d1:
                        maxgen = gen1[:]
                    jj += 1
        gen = maxgen
        return gen

    def select_pop(self, pop):
        # 选择种群，优胜劣汰，策略1：低于平均的要替换改变
        best_f_index = np.argmax(self.fitness)
        av = np.median(self.fitness, axis=0)
        for i in range(self.pop_size):
            if i != best_f_index and self.fitness[i] < av:
                pi = self.cross(pop[best_f_index], pop[i])
                pi = self.mutate(pi)
                pop[i, :] = pi[:]
        return pop

    def select_pop2(self, pop):
        # 选择种群，优胜劣汰，策略2：轮盘赌，适应度低的替换的阈值大
        probility = self.fitness / self.fitness.sum()
        idx = np.random.choice(
            np.arange(self.pop_size), size=self.pop_size, replace=True, p=probility
        )
        n_pop = pop[idx, :]
        return n_pop

    def select_pop3(self, pop):
        # 选择种群，优胜劣汰，锦标赛选择
        tournament_size = 3  # 锦标赛的大小，即每次选择的个体数量
        selected_pop = []

        for _ in range(self.pop_size):
            tournament = np.random.choice(
                range(self.pop_size), size=tournament_size, replace=False
            )  # 随机选择tournament_size个个体作为锦标赛参与者
            best_f_index = max(
                tournament, key=lambda x: self.fitness[x]
            )  # 选择适应度最高的个体作为胜者
            selected_pop.append(pop[best_f_index].copy())  # 将胜者加入选中的个体中
        return np.array(selected_pop)

    def select_pop4(self, pop):
        # 选择种群，优胜劣汰，锦标赛选择与精英保留策略的结合
        tournament_size = 3  # 锦标赛的大小，即每次选择的个体数量
        elite_index = np.argmax(self.fitness)  # 最优个体的索引
        elite = pop[elite_index].copy()  # 复制最优个体作为精英个体
        selected_pop = [elite]  # 将精英个体加入选中的个体中

        for _ in range(self.pop_size - 1):
            tournament = np.random.choice(
                range(self.pop_size), size=tournament_size, replace=False
            )  # 随机选择tournament_size个个体作为锦标赛参与者
            best_f_index = max(
                tournament, key=lambda x: self.fitness[x]
            )  # 选择适应度最高的个体作为胜者
            selected_pop.append(pop[best_f_index].copy())  # 将胜者加入选中的个体中

        return np.array(selected_pop)

    def cross(self, part1, part2):
        """交叉p1,p2的部分基因片段"""
        if np.random.rand() > self.c_rate:
            return part1
        index1 = np.random.randint(0, self.city_size - 1)
        index2 = np.random.randint(index1, self.city_size - 1)
        tempGene = part2[index1:index2]  # 交叉的基因片段
        newGene = []
        p1len = 0
        for g in part1:
            if p1len == index1:
                newGene.extend(tempGene)  # 插入基因片段
            if g not in tempGene:
                newGene.append(g)
            p1len += 1
        newGene = np.array(newGene)

        if newGene.shape[0] != self.city_size:
            print("c error")
            return self.creat_pop(1)
        return newGene

    def mutate(self, gene):
        """突变"""
        if np.random.rand() > self.m_rate:
            return gene
        index1 = np.random.randint(0, self.city_size - 1)
        index2 = np.random.randint(index1, self.city_size - 1)
        newGene = self.reverse_gen(gene, index1, index2)
        if newGene.shape[0] != self.city_size:
            print("m error")
            return self.creat_pop(1)
        return newGene

    def reverse_gen(self, gen, i, j):
        # 函数：翻转基因中i到j之间的基因片段
        if i >= j:
            return gen
        if j > self.city_size - 1:
            return gen
        part1 = np.copy(gen)
        tempGene = part1[i:j]
        newGene = []
        p1len = 0
        for g in part1:
            if p1len == i:
                newGene.extend(tempGene[::-1])  # 插入基因片段
            if g not in tempGene:
                newGene.append(g)
            p1len += 1
        return np.array(newGene)

    def exechange_gen(self, gen, i, j):
        # 函数：交换基因中i,j值
        c = gen[j]
        gen[j] = gen[i]
        gen[i] = c
        return gen

    def gen_distance(self, gen):
        # 计算基因所代表的总旅行距离
        distance = 0.0
        for i in range(-1, len(self.cities) - 1):
            index1, index2 = gen[i], gen[i + 1]
            city1, city2 = self.cities[index1], self.cities[index2]
            distance += self.ct_distance(city1, city2)
        return distance

    def ct_distance(self, city1, city2):
        # 计算2城市之间的欧氏距离
        diff = city1 - city2
        squared_diff = np.power(diff, 2)
        sum_squared_diff = np.sum(squared_diff)
        distance = np.sqrt(sum_squared_diff)
        return distance

    def draw(self):
        x = [city[1] for city in self.data]
        y = [city[2] for city in self.data]

        # 绘制散点图
        plt.scatter(x, y)

        # 添加城市编号标签
        for city in self.data:
            plt.annotate(
                city[0],
                (city[1], city[2]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        # 绘制连线
        for i in range(len(self.best_gene) - 1):
            city1 = self.best_gene[i]
            city2 = self.best_gene[i + 1]
            x1, y1 = self.data[city1][1], self.data[city1][2]
            x2, y2 = self.data[city2][1], self.data[city2][2]
            plt.plot([x1, x2], [y1, y2], "r-")

        # 连接首尾城市
        city1 = self.best_gene[-1]
        city2 = self.best_gene[0]
        x1, y1 = self.data[city1][1], self.data[city1][2]
        x2, y2 = self.data[city2][1], self.data[city2][2]
        plt.plot([x1, x2], [y1, y2], "r-")

        # 设置图形标题和坐标轴标签
        plt.title("City Visualization")
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")

        # 保存图像
        plt.savefig(f'img/City_{self.pop_size}.png')

        # 显示图形
        plt.show()


if __name__ == "__main__":
    c_rate = 0.3986  # 交叉阈值     0.4075 0.3986
    m_rate = 0.253  # 突变阈值    0.4345 0.253
    pop_size = 30  # 种群大小    84.64 83.73
    iteration = 1200  # 迭代次数
    seed = 2023  # 随机种子
    tsp = TSP(c_rate, m_rate, pop_size, iteration, seed)

with open(f'results/result_{pop_size}.txt', 'w') as f:
    count = 0
    for item in tsp.best_gene:
        f.write("%s " % item)
        count += 1
        if count % 10 == 0:
            f.write("\n")