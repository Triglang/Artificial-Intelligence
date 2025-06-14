import random
import time
import tracemalloc
from queue import PriorityQueue

DEBUG = 0

def is_solvable(puzzle):
    """Check if a 15-puzzle is solvable by inversion counting."""
    inversions = 0
    for i in range(16):
        for j in range(i + 1, 16):
            if puzzle[j] and puzzle[i] > puzzle[j]:
                inversions += 1
    empty_row = 4 - (puzzle.index(0) // 4)
    return (inversions % 2 == 0) == (empty_row % 2 == 1)

def generate_random_puzzle():
    """generate a random 15-puzzle"""
    puzzle = list(range(16))
    random.shuffle(puzzle)
    while not is_solvable(puzzle):
        random.shuffle(puzzle)
    return puzzle

def print_state(state):
    """Print the 4x4 puzzle state in human-readable format."""
    assert len(state) == 16
    output = []
    for i in range(0, 16, 4):
        output.append(' '.join(f"{num:<5}" for num in state[i:i + 4]))
    return '\n'.join(output)

def misplaced(state):
    sum = 0
    for i in range(16):
        if state[i] == 0:
            continue
        if state[i] - 1 != i:
            sum += 1
    return sum

def manhattan_distance(state):
    """Calculate Manhattan distance heuristic for A* algorithm."""
    distance = 0
    for i in range(16):
        if state[i] == 0:
            continue
        target_row = (state[i] - 1) // 4
        target_col = (state[i] - 1) % 4
        current_row = i // 4
        current_col = i % 4
        distance += abs(target_row - current_row) + abs(target_col - current_col)
    return distance       


target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]  # 目标状态
# 构建目标位置字典：{数字: 目标索引}
target_pos = {num: idx for idx, num in enumerate(target)}
def inversion_heuristic(state):
    """
    计算相邻颠倒对的启发式值（固定步数=2/对）
    :param state: 当前状态，长度为16的元组（0表示空白块）
    :return: 启发式值
    """
    
    inversion_count = 0
    
    # 遍历所有相邻对（横向和纵向）
    for i in range(16):
        current_num = state[i]
        if current_num == 0:
            continue  # 忽略空白块
        
        # 检查右侧横向邻居
        if (i % 4) < 3:  # 非最右列
            right_neighbor = state[i + 1]
            if right_neighbor != 0:
                # 判断是否颠倒
                if target_pos[current_num] == i + 1 and target_pos[right_neighbor] == i:
                    inversion_count += 1
        
        # 检查下方纵向邻居
        if i < 12:  # 非最下行
            down_neighbor = state[i + 4]
            if down_neighbor != 0:
                # 判断是否颠倒：current_num在目标中的位置应在下方邻居上方
                if target_pos[current_num] == i + 4 and target_pos[down_neighbor] == i:
                    inversion_count += 1
    
    return inversion_count * 2  # 每对颠倒增加2步

def manhattan_reversed_position(state):
    return manhattan_distance(state) + inversion_heuristic(state)       

def generate_children(state):
    """Generate valid child states through possible blank tile moves."""
    blank_pos = state.index(0)
    row, col = blank_pos // 4, blank_pos % 4
    children = []
    
    # Define possible moves: (row, col, direction)
    moves = [(-1, 0, "U"), (1, 0, "D"), (0, -1, "L"), (0, 1, "R")]
    
    for dr, dc, direction in moves:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 4 and 0 <= new_col < 4:
            new_pos = new_row * 4 + new_col
            # 交换空白块位置
            new_state = list(state)
            new_state[blank_pos], new_state[new_pos] = new_state[new_pos], new_state[blank_pos]
            children.append((tuple(new_state), direction))
        
    return children

def reconstruct_path(came_from, end_state):
    """Reconstruct the solution path with both states and moves."""
    path = []
    current_state = end_state
    while came_from[current_state] is not None:
        parent_state, move = came_from[current_state]
        path.append((current_state, move))
        current_state = parent_state
    path.append((current_state, None))  # Initial state
    return path[::-1]  # Reverse to show start-to-end

def out_file(path, start_state, solution, duration, peak_mem):
    with open(path, 'w', encoding='utf-8') as f:
        # 生成结果字符串
        output = []
        for step, (state, move) in enumerate(solution):
            if move is None:
                output.append(f"--------Step {step}--------")
                output.append(print_state(start_state))
            else:
                output.append(f"--------Step {step}--------: {move}")
                output.append(print_state(state))
        
        # 添加性能数据
        output.append(f"\nTime: {duration:.4f} s")
        output.append(f"Peak Memory: {peak_mem / 1024:.2f} KB")
        
        # 同时输出到控制台和文件
        for line in output:
            print(line)
            f.write(line + '\n')
            
heuristic = manhattan_reversed_position

FOUND = 0
def IDA_star(start_state, goal_state):
    def search(path, direction, g, f_limit):
        state = tuple(path[-1])
        current_f = g + heuristic(state)
        if current_f > f_limit:
            return current_f
        if state == goal_state:
            return FOUND
        min_f = float('inf')
        
        new_path = [
            (g + 1 + heuristic(neighbor), neighbor, move) 
            for neighbor, move in generate_children(state) 
            if neighbor not in path
        ]        
                
        new_path.sort()
                
        for _, neighbor, move in new_path:
                path.append(neighbor)
                direction.append(move)
                t = search(path, direction, g+1, f_limit)
                if t == FOUND:
                    return FOUND
                if t < min_f:
                    min_f = t
                path.pop()
                direction.pop()
        return min_f

    f_limit = heuristic(start_state)
    path = [start_state]
    direction = [None]
    while True:
        t = search(path, direction, 0, f_limit)
        if t == FOUND:
            return zip(path, direction)
        if t == float('inf'):
            return None
        f_limit = t

if __name__ == "__main__":
    test_cases = [
        (1, 2, 4, 8, 5, 7, 11, 10, 13, 15, 0, 3, 14, 6, 9, 12),
        (5, 1, 3, 4, 2, 7, 8, 12, 9, 6, 11, 15, 0, 13, 10, 14),
        (14, 10, 6, 0, 4, 9, 1, 8, 2, 3, 5, 11, 12, 13, 7, 15),
        (6, 10, 3, 15, 14, 8, 7, 11, 5, 1, 0, 2, 13, 12, 9, 4),
    ]
    
    goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)
    
    for test_case, start_state in enumerate(test_cases):
        # ----------计时区开始----------
        tracemalloc.start()
        start_time = time.perf_counter()
        solution = IDA_star(start_state, goal_state)
        duration = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # ----------计时区结束----------
        
        file = "IDA_star test case " + str(test_case) + ".txt"
        out_file(file, start_state, solution, duration, peak_mem)
