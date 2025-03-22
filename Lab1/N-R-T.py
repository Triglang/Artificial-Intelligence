import matplotlib.pyplot as plt

# 提供的数据
configurations = [
    "1N-1R-1T", "1N-1R-28T", "1N-1R-56T", "1N-2R-14T",
    "1N-2R-28T", "2N-2R-28T", "2N-2R-56T", "1N-4R-7T",
    "1N-4R-14T", "2N-4R-14T", "2N-4R-28T", "2N-8R-7T",
    "2N-8R-14T", "1N-16R-2T", "2N-16R-2T", "2N-16R-3T",
    "2N-16R-4T", "2N-16R-5T", "2N-16R-6T", "2N-16R-7T",
    "2N-32R-3T", "2N-32R-4T"
]
speeds = [
    0.427, 6.19, 5.95, 3.697, 4.107, 6.858, 5.997, 2.913,
    3.85, 5.965, 6.28, 5.61, 7.513, 3.068, 3.919, 5.202,
    6.246, 6.852, 7.364, 7.879, 7.059, 6.649
]

# 创建折线图
plt.figure(figsize=(12, 6))
plt.plot(configurations, speeds, marker='o', color='skyblue')  # 使用折线图而不是柱状图
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.ylabel("Speed (wSteps/Day)")
plt.title("Performance Across Hardware Configurations")
plt.tight_layout()
plt.show()