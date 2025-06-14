points = [(1, 2), (2, 4), (1, 9), (6, 5), (4, 2), (7, 2), (8, 2), (4, 3)]

initpoint = [(1, 2), (8, 2)]

def manhattan_distance(point1, point2):
    """计算曼哈顿距离"""
    return sum(abs(a - b) for a, b in zip(point1, point2))

if __name__ == "__main__":
    k = 2
    clusters = 