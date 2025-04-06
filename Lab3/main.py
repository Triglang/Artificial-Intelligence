from A_star import A_star
from A_star import out_file
from A_star import generate_random_puzzle
from IDA_star import IDA_star

import time
import tracemalloc

def easy_test():
    easy_cases = [
        (2, 3, 4, 8, 1, 6, 7, 0, 5, 10, 15, 11, 13, 14, 9, 12),
        (0, 1, 4, 8, 6, 3, 7, 12, 5, 2, 9, 11, 13, 10, 14, 15)
    ]
    
    goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)
    
    # ----------A*搜索正确性测试----------
    for idx, start_state in enumerate(easy_cases):
        # ----------计时区开始----------
        tracemalloc.start()
        start_time = time.perf_counter()
        solution = A_star(start_state, goal_state)
        duration = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # ----------计时区结束----------
        file = "A_star easy case " + str(idx + 1) + ".txt"
        out_file(file, start_state, solution, duration, peak_mem)
        
    # ----------IDA*搜索正确性测试----------
    for idx, start_state in enumerate(easy_cases):
        # ----------计时区开始----------
        tracemalloc.start()
        start_time = time.perf_counter()
        solution = IDA_star(start_state, goal_state)
        duration = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # ----------计时区结束----------
        file = "IDA_star easy case " + str(idx + 1) + ".txt"
        out_file(file, start_state, solution, duration, peak_mem)
        
def difficult_test():
    # ----------困难测例测试----------
    difficult_cases = [
        (1, 2, 4, 8, 5, 7, 11, 10, 13, 15, 0, 3, 14, 6, 9, 12),
        (5, 1, 3, 4, 2, 7, 8, 12, 9, 6, 11, 15, 0, 13, 10, 14),
        (14, 10, 6, 0, 4, 9, 1, 8, 2, 3, 5, 11, 12, 13, 7, 15),
        (6, 10, 3, 15, 14, 8, 7, 11, 5, 1, 0, 2, 13, 12, 9, 4),
        (11, 3, 1, 7, 4, 6, 8, 2, 15, 9, 10, 13, 14, 12, 0, 5),
        (0, 5, 15, 14, 7, 9, 6, 13, 1, 2, 12, 10, 8, 11, 4, 3)
    ]
    
    goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)
    
    # ----------A*搜索正确性测试----------
    for idx, start_state in enumerate(difficult_cases):
        # ----------计时区开始----------
        tracemalloc.start()
        start_time = time.perf_counter()
        solution = A_star(start_state, goal_state)
        duration = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # ----------计时区结束----------
        file = "A_star difficult case " + str(idx + 1) + ".txt"
        out_file(file, start_state, solution, duration, peak_mem)
        
    # ----------IDA*搜索正确性测试----------
    for idx, start_state in enumerate(difficult_cases):
        # ----------计时区开始----------
        tracemalloc.start()
        start_time = time.perf_counter()
        solution = IDA_star(start_state, goal_state)
        duration = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # ----------计时区结束----------
        file = "IDA_star difficult case " + str(idx + 1) + ".txt"
        out_file(file, start_state, solution, duration, peak_mem)
        
def random_test():
    goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)
    
    # ----------随机测例测试----------
    for idx in range(10):
        start_state = tuple(generate_random_puzzle())
        # ----------A*计时区开始----------
        tracemalloc.start()
        start_time = time.perf_counter()
        solution = A_star(start_state, goal_state)
        duration = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # ----------A*计时区结束----------
        file = "A_star random case " + str(idx + 1) + ".txt"
        out_file(file, start_state, solution, duration, peak_mem)
        
        # ----------IDA*计时区开始----------
        tracemalloc.start()
        start_time = time.perf_counter()
        solution = IDA_star(start_state, goal_state)
        duration = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # ----------IDA*计时区结束----------
        
        file = "A_star random case " + str(idx + 1) + ".txt"
        out_file(file, start_state, solution, duration, peak_mem)

if __name__ == "__main__":
    easy_test()
    difficult_test()
    random_test()
