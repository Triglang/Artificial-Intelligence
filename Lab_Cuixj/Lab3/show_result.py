import pygame
import sys
import imageio
import numpy as np

# 配置参数
TILE_SIZE = 80
GRID_MARGIN = 5
WINDOW_SIZE = (TILE_SIZE + GRID_MARGIN) * 4 + GRID_MARGIN
COLORS = {
    'background': (0, 0, 0),
    'tile': (255, 215, 0),
    'text': (0, 0, 0),
    'blank': (30, 30, 30),
    'highlight': (200, 200, 200)
}

class PuzzleVisualizer:
    def __init__(self, solution_path, output_path="solution.gif"):
        self.solution_path = solution_path
        self.output_path = output_path
        self.frames = []
        
        if not pygame.get_init():
            pygame.init()
            
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption(f"15-Puzzle - {output_path}")
        self.font = pygame.font.Font(None, 36)

    def _tuple_to_grid(self, state_tuple):
        return [state_tuple[i*4:(i+1)*4] for i in range(4)]
    
    def _capture_frame(self):
        frame = pygame.surfarray.array3d(self.screen)
        self.frames.append(np.transpose(frame, (1, 0, 2)))

    def _update_display(self, step):
        state_tuple, move = self.solution_path[step]
        grid = self._tuple_to_grid(state_tuple)
        
        self.screen.fill(COLORS['background'])
        for y in range(4):
            for x in range(4):
                num = grid[y][x]
                rect = [
                    (GRID_MARGIN + TILE_SIZE) * x + GRID_MARGIN,
                    (GRID_MARGIN + TILE_SIZE) * y + GRID_MARGIN,
                    TILE_SIZE,
                    TILE_SIZE
                ]
                
                color = COLORS['blank'] if num == 0 else COLORS['tile']
                pygame.draw.rect(self.screen, color, rect, border_radius=8)
                
                if num != 0:
                    text = self.font.render(str(num), True, COLORS['text'])
                    text_rect = text.get_rect(center=(
                        rect[0] + TILE_SIZE/2,
                        rect[1] + TILE_SIZE/2
                    ))
                    self.screen.blit(text, text_rect)
        
        info_text = f"Step: {step}/{len(self.solution_path)-1}  Move: {move if move else 'Start'}"
        text_surf = self.font.render(info_text, True, (255, 255, 255))  # 修复此处
        self.screen.blit(text_surf, (10, WINDOW_SIZE-40))
        pygame.display.flip()

    def _generate_gif(self):
        if len(self.frames) == 0:
            return

        extended_frames = [self.frames[0]] * 2 
        extended_frames += self.frames
        extended_frames += [self.frames[-1]] * 2

        try:
            with imageio.get_writer(self.output_path, mode='I', fps=2) as writer:
                for frame in extended_frames:
                    writer.append_data(frame)
            print(f"成功生成：{self.output_path}")
        except Exception as e:
            print(f"生成失败：{str(e)}")

    def run(self):
        try:
            for step in range(len(self.solution_path)):
                self._update_display(step)
                self._capture_frame()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                
                pygame.time.delay(500)
            self._generate_gif()
            return True
            
        finally:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    test_cases = [
        [
            ((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0), None),
            ((1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,15), "R"),
            ((1,2,3,4,5,6,7,8,9,10,11,12,13,0,14,15), "U"),
            ((1,2,3,4,5,6,7,8,9,10,11,0,13,12,14,15), "L")
        ],
        [
            ((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0), None),
            ((1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,15), "R"),
            ((1,2,3,4,5,6,7,8,9,10,11,12,13,0,14,15), "U")
        ]
    ]

    for idx, solution in enumerate(test_cases):
        visualizer = PuzzleVisualizer(solution, f"case_{idx}.gif")
        success = visualizer.run()
        
        if not success:
            print(f"提前终止：case_{idx}.gif")
            break
        
        pygame.time.delay(100)  

    print("所有案例处理完成")