import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter


class TSP(object):
    def __init__(self, c_rate, m_rate, pop_size, iteration=500, seed=2023):
        """
        初始化旅行商问题求解器
        :param c_rate: 交叉概率概率
        :param m_rate: 突变概率概率 
        :param pop_size: 种群大小
        :param iteration: 迭代次数，默认为500
        :param seed: 随机种子，默认为2023
        """
        self.cities = np.array([])  # 城市数组
        self.cities_name = np.array([])
        self.city_size = -1  # 标记城市数目
        self.pop_size = int(pop_size)  # 种群大小
        self.fitness = np.zeros(self.pop_size)  # 种群适应度
        self.c_rate = c_rate  # 交叉概率
        self.m_rate = m_rate  # 突变概率
        self.iteration = iteration  # 迭代次数
        self.best_dist = -1  # 最优距离
        self.best_gene = []  # 最优路径
        np.random.seed(seed)  # 随机种子

        self.init()  # 初始化
        self.evolution()  # 进化
        self.draw()  # 绘制

    def init(self):
        """
        初始化函数，用于读取数据、创建种群并计算初始种群适应度。

        :return: None
        """
        # 读取数据文件 "eil51.txt"，以空格为分隔符，不包含表头
        self.data = pd.read_csv("eil51.txt", delimiter=" ", header=None).values
        # 提取城市坐标信息
        self.cities = self.data[:, 1:]
        # 提取城市名称信息
        self.cities_name = self.data[:, 0]
        # 获取城市数量
        self.city_size = self.data.shape[0]
        # 创建种群
        self.pop = self.create_pop(self.pop_size)
        # 计算初始种群适应度
        self.fitness = self.get_fitness(self.pop)

    def evolution(self):
        """
        进化函数，用于进行遗传算法的迭代过程。

        :return: None
        """
        writer = SummaryWriter()  # 创建一个SummaryWriter对象，用于记录训练过程中的数据
        for i in range(self.iteration):  # 进行指定次数的迭代
            best_f_index = np.argmax(self.fitness)  # 找到当前种群中适应度最高的个体的索引
            worst_f_index = np.argmin(self.fitness)  # 找到当前种群中适应度最低的个体的索引
            local_best_gene = self.pop[best_f_index]  # 获取当前种群中适应度最高的个体的基因
            local_best_dist = self.gen_distance(local_best_gene)  # 计算当前种群中适应度最高的个体的适应度值
            if i == 0:  # 如果是第一次迭代，将当前种群中适应度最高的个体作为最优解
                self.best_gene = local_best_gene
                self.best_dist = self.gen_distance(local_best_gene)

            if (
                local_best_dist < self.best_dist
            ):  # 如果当前种群中适应度最高的个体的适应度值小于之前的最优解，更新最优解和最优适应度值
                self.best_dist = local_best_dist  # 记录最优值
                self.best_gene = local_best_gene  # 记录最个体基因

            else:
                self.pop[
                    worst_f_index
                ] = self.best_gene  # 如果当前种群中适应度最高的个体的适应度值不小于之前的最优解，将最优解替换为当前种群中适应度最低的个体
            print("gen:%d evo,best dist :%s" % (i, self.best_dist))  # 打印当前迭代次数和最优适应度值

            self.pop = self.select_pop4(self.pop)  # 选择淘汰种群
            self.fitness = self.get_fitness(self.pop)  # 计算种群适应度
            for j in range(self.pop_size):  # 对种群中的每个个体进行交叉和突变操作
                r = np.random.randint(0, self.pop_size - 1)
                if j != r:
                    self.pop[j] = self.cross(self.pop[j], self.pop[r])  # 交叉种群中第j,r个体的基因
                    self.pop[j] = self.mutate(self.pop[j])  # 突变种群中第j个体的基因
            self.best_gene = self.EO(self.best_gene)  # 极值优化，防止收敛局部最优
            self.best_dist = self.gen_distance(self.best_gene)  # 记录最优值
            writer.add_scalar(
                "best_dis", self.best_dist, i
            )  # 将当前最优适应度值添加到SummaryWriter对象中
        writer.close()  # 关闭SummaryWriter对象，结束训练过程

    def create_pop(self, size):
        """
        创建一个指定大小的种群，每个个体都是一个随机排列的城市顺序。

        :param size: 种群大小
        :return: 返回一个包含size个随机排列城市的NumPy数组
        """
        pop = [np.random.permutation(self.city_size) for _ in range(size)]
        return np.array(pop)

    def get_fitness(self, pop):
        """
        计算种群中每个个体的适应度值。

        :param pop: 种群，一个二维数组，每一行代表一个个体的基因编码。
        :return: 适应度值数组，与输入种群形状相同。
        """
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

    def EO(self, gene):
        """
        极值优化，传统遗传算法性能不好，这里混合EO
        其会在整个基因的领域内，寻找一个最佳变换以更新基因
        :param gene:
        :return:
        """
        local_fitness = np.zeros(self.city_size)  # 初始化局部适应度数组
        for g in range(self.city_size):  # 遍历城市数量
            local_fitness[g] = self.get_local_fitness(gene, g)  # 计算每个城市的局部适应度
        max_city_i = np.argmax(local_fitness)  # 找到局部适应度最高的城市的索引
        maxgen = gene[:]  # 复制当前基因
        if 1 < max_city_i < self.city_size - 1:  # 如果最高适应度的城市的索引不在边界上
            for j in range(max_city_i):  # 遍历最高适应度城市的左侧城市
                maxgen = gene[:]  # 复制当前基因
                jj = max_city_i  # 初始化右侧城市的索引为最高适应度城市的索引
                while jj < self.city_size:  # 遍历右侧城市
                    gen1 = self.exechange_gen(maxgen, j, jj)  # 交换两个城市的位置
                    d = self.gen_distance(maxgen)  # 计算原始基因的距离
                    d1 = self.gen_distance(gen1)  # 计算交换位置后的基因的距离
                    if d > d1:  # 如果交换位置后的基因距离更短
                        maxgen = gen1[:]  # 更新最大适应度基因
                    jj += 1  # 右侧城市的索引加1
        gene = maxgen  # 更新基因
        return gene  # 返回更新后的基因

    # def select_pop(self, pop):
    #     # 选择种群，优胜劣汰，策略1：低于平均的要替换改变
    #     best_f_index = np.argmax(self.fitness)
    #     av = np.median(self.fitness, axis=0)
    #     for i in range(self.pop_size):
    #         if i != best_f_index and self.fitness[i] < av:
    #             pi = self.cross(pop[best_f_index], pop[i])
    #             pi = self.mutate(pi)
    #             pop[i, :] = pi[:]
    #     return pop
    #
    # def select_pop2(self, pop):
    #     # 选择种群，优胜劣汰，策略2：轮盘赌，适应度低的替换的概率大
    #     probility = self.fitness / self.fitness.sum()
    #     idx = np.random.choice(
    #         np.arange(self.pop_size), size=self.pop_size, replace=True, p=probility
    #     )
    #     n_pop = pop[idx, :]
    #     return n_pop
    #
    # def select_pop3(self, pop):
    #     # 选择种群，优胜劣汰，锦标赛选择
    #     tournament_size = 3  # 锦标赛的大小，即每次选择的个体数量
    #     selected_pop = []
    #
    #     for _ in range(self.pop_size):
    #         tournament = np.random.choice(
    #             range(self.pop_size), size=tournament_size, replace=False
    #         )  # 随机选择tournament_size个个体作为锦标赛参与者
    #         best_f_index = max(
    #             tournament, key=lambda x: self.fitness[x]
    #         )  # 选择适应度最高的个体作为胜者
    #         selected_pop.append(pop[best_f_index].copy())  # 将胜者加入选中的个体中
    #     return np.array(selected_pop)

    def select_pop4(self, pop):
        """
        选择种群，优胜劣汰，锦标赛选择与精英保留策略的结合
        :param pop:
        :return:
        """
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
        """
        交叉p1,p2的部分基因片段
        :param part1: 第一个部分基因片段
        :param part2: 第二个部分基因片段
        :return: 新的基因片段
        """
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
        """
        对给定的基因进行变异操作。

        :param gene: 待变异的基因
        :return: 变异后的基因
        """
        if np.random.rand() > self.m_rate:  # 以一定概率进行变异
            return gene  # 不进行变异，直接返回原基因
        index1 = np.random.randint(0, self.city_size - 1)  # 随机选择第一个索引
        index2 = np.random.randint(
            index1, self.city_size - 1
        )  # 随机选择第二个索引，确保第一个索引小于等于第二个索引
        newGene = self.reverse_gen(gene, index1, index2)  # 调用reverse_gen方法进行基因变异
        if newGene.shape[0] != self.city_size:  # 检查变异后的基因长度是否与预期相符
            print("m error")  # 输出错误信息
            return self.creat_pop(1)  # 创建一个新的个体并返回
        return newGene  # 返回变异后的基因

    def reverse_gen(self, gene, i, j):
        """
        将给定的基因片段（从索引i到索引j）反转，并将其插入到原始基因序列中。

        :param gene: 原始基因序列
        :param i: 要反转的基因片段的起始索引
        :param j: 要反转的基因片段的结束索引
        :return: 包含反转基因片段的新基因序列
        """
        if i >= j:
            return gene
        if j > self.city_size - 1:
            return gene
        part1 = gene[:]
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

    def exechange_gen(self, gene, i, j):
        """
        交换列表中下标为i和j的元素。

        :param gene: 待交换元素的列表
        :param i: 第一个元素的下标
        :param j: 第二个元素的下标
        :return: 交换元素后的列表
        """
        gene[i], gene[j] = gene[j], gene[i]
        return gene

    def gen_distance(self, gene):
        """
        计算基因序列对应的路径长度

        :param gene: 基因序列，表示城市之间的顺序
        :return: 路径长度
        """
        distance = 0.0
        for i in range(-1, len(self.cities) - 1):
            index1, index2 = gene[i], gene[i + 1]
            city1, city2 = self.cities[index1], self.cities[index2]
            distance += self.ct_distance(city1, city2)
        return distance

    def ct_distance(self, city1, city2):
        """
        计算两个城市之间的欧氏距离。

        :param city1: 第一个城市的坐标，类型为numpy数组
        :param city2: 第二个城市的坐标，类型为numpy数组
        :return: 两个城市之间的欧氏距离，类型为浮点数
        """
        diff = city1 - city2
        squared_diff = np.power(diff, 2)
        sum_squared_diff = np.sum(squared_diff)
        distance = np.sqrt(sum_squared_diff)
        return distance

    def draw(self):
        """
        绘制城市地图
        :return:
        """
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
        plt.savefig(f"img/City.png")

        # 显示图形
        plt.show()


if __name__ == "__main__":
    c_rate = 0.3986  # 交叉概率     0.4075 0.3986
    m_rate = 0.253  # 突变概率    0.4345 0.253
    pop_size = 83.73  # 种群大小    84.64 83.73
    iteration = 1200  # 迭代次数
    seed = 2023  # 随机种子
    tsp = TSP(c_rate, m_rate, pop_size, iteration, seed)

with open(f"results/result.txt", "w") as f:
    count = 0
    for item in tsp.best_gene:
        f.write("%s " % item)
        count += 1
        if count % 10 == 0:
            f.write("\n")
