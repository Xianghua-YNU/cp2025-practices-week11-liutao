
import numpy as np
import matplotlib.pyplot as plt

# 物理常数
kB = 1.380649e-23  # 玻尔兹曼常数，单位：J/K

# 样本参数
V = 1000e-6  # 体积，1000立方厘米转换为立方米
rho = 6.022e28  # 原子数密度，单位：m^-3
theta_D = 428  # 德拜温度，单位：K


def integrand(x):
    """被积函数：x^4 * e^x / (e^x - 1)^2

    这个函数表示德拜模型中的能量分布。分子中的x^4表示声子态密度，
    分母中的(e^x - 1)^2表示玻色-爱因斯坦分布的导数。

    参数：
    x : float 或 numpy.ndarray
        积分变量，代表约化能量 (E/kT)

    返回：
    float 或 numpy.ndarray：被积函数的值
    """
    # 对于数组输入，逐个计算每个元素
    if isinstance(x, np.ndarray):
        return np.array([integrand(xi) for xi in x])

    # 对于单个值的计算
    exp_x = np.exp(x)
    return x**4 * exp_x / (exp_x - 1)**2


def gauss_quadrature(f, a, b, n):
    """实现高斯-勒让德积分

    使用高斯-勒让德求积公式计算定积分。这个方法对于光滑函数特别有效，
    使用n个点就可以精确积分2n-1次多项式。

    参数：
    f : callable
        被积函数
    a, b : float
        积分区间的端点
    n : int
        高斯点的数量

    返回：
    float：积分结果
    """
    # 获取高斯-勒让德求积的节点和权重
    x, w = np.polynomial.legendre.leggauss(n)

    # 将[-1,1]区间映射到[a,b]区间
    t = 0.5 * (x + 1) * (b - a) + a

    # 计算积分
    return 0.5 * (b - a) * np.sum(w * f(t))


def cv(T):
    """计算给定温度T下的热容

    根据德拜模型计算固体的热容。该模型在低温下给出T^3定律，
    在高温下趋近于杜隆-珀替定律的预测值。

    参数：
    T : float
        温度，单位：K

    返回：
    float：热容值，单位：J/K
    """
    # 计算积分上限
    upper_limit = theta_D / T

    # 使用高斯积分计算
    integral = gauss_quadrature(integrand, 0, upper_limit, 50)

    # 计算热容
    return 9 * V * rho * kB * (T / theta_D)**3 * integral


def plot_cv():
    """绘制热容随温度的变化曲线

    生成一个专业的图表，展示热容随温度的变化。包括：
    - 合适的坐标轴标签
    - 网格线
    - 图例
    - 标题
    """
    # 生成温度点（使用线性间距）
    T = np.linspace(5, 500, 200)

    # 计算对应的热容值
    C_V = np.array([cv(t) for t in T])

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 绘制热容曲线
    plt.plot(T, C_V, 'b-', label='Debye Model')

    # 添加参考线
    # 低温T^3行为
    T_low = np.linspace(5, 50, 50)
    C_low = cv(50) * (T_low/50)**3
    plt.plot(T_low, C_low, 'r--', label='T³ Law')

    # Add labels and title
    plt.xlabel('Temperature (K)')
    plt.ylabel('Heat Capacity (J/K)')
    plt.title('Solid Heat Capacity vs Temperature (Debye Model)')

    # Add grid
    plt.grid(True, which='both', ls='-', alpha=0.2)

    # Add legend
    plt.legend()

    # 显示图表
    plt.show()


def main():
    # 测试一些特征温度点的热容值
    test_temperatures = [5, 100, 300, 500]
    print("\n测试不同温度下的热容值：")
    print("-" * 40)
    print("温度 (K)\t热容 (J/K)")
    print("-" * 40)
    for T in test_temperatures:
        result = cv(T)
        print(f"{T:8.1f}\t{result:10.3e}")

    # 绘制热容曲线
    plot_cv()


if __name__ == '__main__':
    main()
