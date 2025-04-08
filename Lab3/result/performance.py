import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# 手动定义数据
cases = ['case1', 'case2', 'case3', 'case4', 'case5', 'case6']

# A*算法数据
a_star_time = [0.002, 0.0005, 12.6959, 82.1854, 791.9662, 3397.9407]
a_star_mem = [98.62, 9.86, 379883.74, 1998566.01, 13360956.74, 27561181.09]

# IDA*算法数据（第六个值为NaN）
ida_star_time = [0.002, 0.0003, 95.376, 213.1784, 6924.2008, np.nan]
ida_star_mem = [14.93, 2.98, 21.95, 7.98, 12.98, np.nan]

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 绘制时间对比（自动跳过NaN）
ax1.plot(cases, a_star_time, marker='o', linestyle='--', label='A* Time', color='#1f77b4')
ax1.plot(cases[:5], ida_star_time[:5], marker='s', linestyle='-.', label='IDA* Time', color='#ff7f0e')

# 在点上添加数值标签
for i, (case, t) in enumerate(zip(cases, a_star_time)):
    ax1.text(i, t, f'{t:.4f}', ha='center', va='bottom', color='#1f77b4', fontsize=8)
for i, (case, t) in enumerate(zip(cases[:5], ida_star_time[:5])):
    ax1.text(i, t, f'{t:.4f}', ha='center', va='top', color='#ff7f0e', fontsize=8)

# 设置对数刻度并禁用科学计数法
ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(ScalarFormatter())
ax1.set_title('Time Comparison (Log Scale)', fontsize=12, pad=20)
ax1.set_ylabel('Time (seconds)')
ax1.grid(True, alpha=0.4)
ax1.legend()

# 绘制内存对比
ax2.plot(cases, a_star_mem, marker='o', linestyle='--', label='A* Memory', color='#2ca02c')
ax2.plot(cases[:5], ida_star_mem[:5], marker='s', linestyle='-.', label='IDA* Memory', color='#d62728')

# 在点上添加数值标签
for i, (case, m) in enumerate(zip(cases, a_star_mem)):
    ax2.text(i, m, f'{m:.2f}', ha='center', va='bottom', color='#2ca02c', fontsize=8)
for i, (case, m) in enumerate(zip(cases[:5], ida_star_mem[:5])):
    ax2.text(i, m, f'{m:.2f}', ha='center', va='top', color='#d62728', fontsize=8)

# 设置对数刻度并禁用科学计数法
ax2.set_yscale('log')
ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.set_title('Memory Comparison (Log Scale)', fontsize=12, pad=20)
ax2.set_ylabel('Peak Memory (KB)')
ax2.grid(True, alpha=0.4)
ax2.legend()

plt.suptitle('A* vs IDA* Algorithm Performance Comparison (Partial Data)', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig("performance.png")
