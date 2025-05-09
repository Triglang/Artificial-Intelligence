import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# 手动定义数据（确保所有数据完整）
cases = ['case1', 'case2', 'case3', 'case4', 'case5', 'case6']

# Manhattan启发式数据
manhattan_time = [0.002, 0.0005, 12.6959, 82.1854, 791.9662, 3397.9407]
manhattan_mem = [98.62, 9.86, 379883.74, 1998566.01, 13360956.74, 27561181.09]

# Manhattan+反转启发式数据（假设所有数据均完整）
manhattan_rev_time = [0.0027, 0.0005, 13.4649, 74.7587, 764.9172, 1864.7039]
manhattan_rev_mem = [98.68, 9.86, 272561.03, 1606712.52, 12083791.30, 24093703.96]

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), dpi=150)
plt.rcParams['font.family'] = 'DejaVu Sans'  # 设置字体防止中文乱码

# ================== 时间对比图 ==================
# 绘制曲线
ax1.plot(cases, manhattan_time, marker='o', markersize=8, linestyle='--', linewidth=2,
        color='#1f77b4', label='Manhattan')
ax1.plot(cases, manhattan_rev_time, marker='s', markersize=8, linestyle='-.', linewidth=2,
        color='#ff7f0e', label='Manhattan+Reversed')

# 添加数值标签
for i, (mt, mrt) in enumerate(zip(manhattan_time, manhattan_rev_time)):
    ax1.text(i, mt, f'{mt:.4f}', ha='center', va='bottom', color='#1f77b4', fontsize=9)
    ax1.text(i, mrt, f'{mrt:.4f}', ha='center', va='top', color='#ff7f0e', fontsize=9)

# 配置坐标轴
ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(ScalarFormatter())
ax1.set_title('Time Comparison (Log Scale)', fontsize=14, pad=20)
ax1.set_ylabel('Time (seconds)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper left', fontsize=12)

# ================== 内存对比图 ==================
ax2.plot(cases, manhattan_mem, marker='o', markersize=8, linestyle='--', linewidth=2,
        color='#2ca02c', label='Manhattan')
ax2.plot(cases, manhattan_rev_mem, marker='s', markersize=8, linestyle='-.', linewidth=2,
        color='#d62728', label='Manhattan+Reversed')

# 添加数值标签
for i, (mm, mrm) in enumerate(zip(manhattan_mem, manhattan_rev_mem)):
    ax2.text(i, mm, f'{mm/1e3:.1f}K' if mm > 1e3 else f'{mm:.2f}', 
            ha='center', va='bottom', color='#2ca02c', fontsize=9)
    ax2.text(i, mrm, f'{mrm/1e3:.1f}K' if mrm > 1e3 else f'{mrm:.2f}', 
            ha='center', va='top', color='#d62728', fontsize=9)

# 配置坐标轴
ax2.set_yscale('log')
ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.set_title('Memory Comparison (Log Scale)', fontsize=14, pad=20)
ax2.set_ylabel('Peak Memory (KB)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper left', fontsize=12)

# 全局设置
plt.suptitle('Heuristic Function Performance Comparison', y=1.02, fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig("heuristic_comparison.png", bbox_inches='tight', dpi=300)
# plt.show()