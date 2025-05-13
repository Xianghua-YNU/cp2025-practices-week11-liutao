import numpy as np
import matplotlib.pyplot as plt

# --- 常量定义 ---
a = 1.0  # 圆环半径 (单位: m)
C = 1.0 / (2 * np.pi)  # 对应 q=1 的常数项

# --- 计算函数 ---
def calculate_potential_on_grid(y_coords, z_coords):
    """计算yz平面电势分布"""
    print("开始计算电势...")
    num_phi = 1000  # 积分点数
    phi = np.linspace(0, 2*np.pi, num_phi)
    y_grid, z_grid = np.meshgrid(y_coords, z_coords, indexing='ij')
    
    # 计算场点到圆环各点的距离
    x_ring = a * np.cos(phi)[np.newaxis, np.newaxis, :]
    y_ring = a * np.sin(phi)[np.newaxis, np.newaxis, :]
    R = np.sqrt((0 - x_ring)**2 + (y_grid[:, :, np.newaxis] - y_ring)**2 + z_grid[:, :, np.newaxis]**2)
    
    # 处理极小值
    R[R < 1e-10] = 1e-10
    
    # 积分计算电势
    dV = C / R
    V = np.trapz(dV, phi, axis=2)
    print("电势计算完成.")
    return V, y_grid, z_grid

def calculate_electric_field_on_grid(V, y_coords, z_coords):
    """通过电势梯度计算电场"""
    print("开始计算电场...")
    dy = y_coords[1] - y_coords[0]
    dz = z_coords[1] - z_coords[0]
    
    # 计算梯度并取负值
    grad_y, grad_z = np.gradient(-V, dy, dz)
    print("电场计算完成.")
    return grad_y, grad_z

# --- 可视化函数 ---
def plot_potential_and_field(y_coords, z_coords, V, Ey, Ez, y_grid, z_grid):
    """绘制等势线与电场线"""
    print("开始绘图...")
    fig = plt.figure(figsize=(12, 6))
    
    # 等势线图
    plt.subplot(1, 2, 1)
    levels = 20
    cf = plt.contourf(y_grid/a, z_grid/a, V, levels=levels, cmap='viridis')
    plt.colorbar(cf, label='Electric Potential')
    plt.contour(y_grid/a, z_grid/a, V, levels=levels, colors='k', linewidths=0.5)
    plt.xlabel('y/a'), plt.ylabel('z/a')
    plt.title('Equipotential Contours')
    plt.gca().set_aspect('equal')
    plt.grid(True)
    
    # 电场流线图
    plt.subplot(1, 2, 2)
    E_magnitude = np.sqrt(Ey**2 + Ez**2)
    stream = plt.streamplot(y_grid/a, z_grid/a, Ey, Ez, 
                          color=E_magnitude, cmap='plasma', linewidth=1, 
                          density=1.5, arrowsize=1)
    plt.colorbar(stream.lines, label='Electric Field Magnitude')
    plt.plot([-a, a], [0, 0], 'ro', markersize=5, label='Ring')
    plt.legend()
    plt.xlabel('y/a'), plt.ylabel('z/a')
    plt.title('Electric Field Streamlines')
    plt.gca().set_aspect('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    print("绘图完成.")

# --- 主程序 ---
if __name__ == "__main__":
    num_points = 40
    range_factor = 2
    y_range = np.linspace(-range_factor*a, range_factor*a, num_points)
    z_range = np.linspace(-range_factor*a, range_factor*a, num_points)
    
    V, y_grid, z_grid = calculate_potential_on_grid(y_range, z_range)
    Ey, Ez = calculate_electric_field_on_grid(V, y_range, z_range)
    
    plot_potential_and_field(y_range, z_range, V, Ey, Ez, y_grid, z_grid)
