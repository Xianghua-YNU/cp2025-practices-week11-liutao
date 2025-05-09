import numpy as np
import matplotlib.pyplot as plt

# --- 物理和线圈参数 ---
MU0 = 4 * np.pi * 1e-7  # 真空磁导率 (T*m/A)
I = 1.0  # 电流 (A) - 假设为1A，实际计算中常数因子可以合并

def Helmholtz_coils(r_low, r_up, d):
   '''
    计算亥姆霍兹线圈（或两个不同半径线圈）的磁场。
    输入:
        r_low: 下方线圈的半径 (m)
        r_up: 上方线圈的半径 (m)
        d: 两线圈中心之间的距离 (m)
    返回:
        Y, Z: 空间坐标网格
        By, Bz: y和z方向的磁场分量 (T)
    '''
    # 生成积分角度phi
    phi = np.linspace(0, 2*np.pi, 20)
    # 定义空间网格范围
    r = max(r_low, r_up)
    y = np.linspace(-2*r, 2*r, 25)
    z = np.linspace(-2*d, 2*d, 25)
    
    # 创建三维网格（顺序为 y, z, phi）
    Y, Z, phi = np.meshgrid(y, z, phi)
    
    # 计算到下方线圈（z=+d/2）和上方线圈（z=-d/2）的距离
    r1 = np.sqrt((r_low * np.cos(phi))**2 + (Y - r_low * np.sin(phi))**2 + (Z - d/2)**2)
    r2 = np.sqrt((r_up * np.cos(phi))**2 + (Y - r_up * np.sin(phi))**2 + (Z + d/2)**2)
    
    # 避免除以零
    r1[r1 < 1e-9] = 1e-9
    r2[r2 < 1e-9] = 1e-9
    
    # 计算被积函数
    dBy = (r_low * (Z - d/2) * np.sin(phi)) / r1**3 + (r_up * (Z + d/2) * np.sin(phi)) / r2**3
    dBz = (r_low * (r_low - Y * np.sin(phi))) / r1**3 + (r_up * (r_up - Y * np.sin(phi))) / r2**3
    
    # 对phi轴进行积分（显式指定积分轴和间隔）
    By_unscaled = np.trapz(dBy, x=phi[0,0,:], axis=2)
    Bz_unscaled = np.trapz(dBz, x=phi[0,0,:], axis=2)
    
    # 应用比例因子
    scaling_factor = (MU0 * I) / (4 * np.pi)
    By = scaling_factor * By_unscaled
    Bz = scaling_factor * Bz_unscaled
    
    return Y[:, :, 0], Z[:, :, 0], By, Bz

def plot_magnetic_field_streamplot(r_low, r_up, d):
    Y, Z, By, Bz = Helmholtz_coils(r_low, r_up, d)
    
    # 设置流线起始点（z=0平面）
    bSY = np.linspace(-0.8*r_low, 0.8*r_low, 10)
    bSY, bSZ = np.meshgrid(bSY, 0)
    start_points = np.vstack([bSY.ravel(), bSZ.ravel()]).T
    
    # 绘制流线图
    plt.figure(figsize=(8, 6))
    plt.streamplot(Y, Z, By, Bz, density=1.5, color='k', 
                   arrowstyle='->', start_points=start_points)
    
    # 绘制线圈截面
    plt.plot([-r_low, r_low], [-d/2, -d/2], 'b-', lw=2, label=f'Lower Coil (R={r_low}m)')
    plt.plot([-r_up, r_up], [d/2, d/2], 'r-', lw=2, label=f'Upper Coil (R={r_up}m)')
    
    # 设置图形属性
    plt.xlabel('y (m)')
    plt.ylabel('z (m)')
    plt.title(f'Helmholtz Coils: R1={r_low}m, R2={r_up}m, d={d}m')
    plt.gca().set_aspect('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    # 标准亥姆霍兹配置（d=R时磁场最均匀）
    coil_radius = 0.5  # 线圈半径 (m)
    coil_distance = 0.5 # 线圈间距 (m)
    plot_magnetic_field_streamplot(coil_radius, coil_radius, coil_distance)
