import pygame
import sys
import math

# ---------------------- CONFIGURATION ----------------------
GRID_SIZE = 5  # 5x5 board
CELL_SIZE = 100  # pixels per cell
BOARD_SIZE = GRID_SIZE * CELL_SIZE  # width/height of board portion
PANEL_WIDTH = 220  # control panel width
WIN_HEIGHT = BOARD_SIZE
WIN_WIDTH = BOARD_SIZE + PANEL_WIDTH

# Colors (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GRAY = (230, 230, 230)
DARK_GRAY = (140, 140, 140)
BLUE = (30, 144, 255)
RED = (220, 20, 60)
HIGHLIGHT = (0, 0, 0)  # Black outline for highlighted pieces

# UI button specifications
BUTTON_HEIGHT = 50
BUTTON_PADDING = 15
FONT_SIZE = 18

# Drawing modes
action_modes = [
    "BLUE_PIECE",
    "RED_PIECE",
    "HIGHLIGHT",
    "ARROW"
]


class Button:
    """Simple rectangular button class"""
    def __init__(self, rect, text, mode):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.mode = mode
        self.hover = False

    def draw(self, surface, font, active=False):
        color = DARK_GRAY if active else (LIGHT_GRAY if self.hover else WHITE)
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        label = font.render(self.text, True, BLACK)
        label_rect = label.get_rect(center=self.rect.center)
        surface.blit(label, label_rect)

    def is_over(self, pos):
        return self.rect.collidepoint(pos)


# ---------------------- DATA STRUCTURES ----------------------
# Store placed items so we can redraw every frame
dots = []        # list of dicts: {row, col, color}
highlights = []  # list of (row, col)
arrows = []      # list of dicts: {start: (row, col), end: (row, col)}


# ---------------------- DRAWING HELPERS ----------------------

def cell_center(row, col):
    """Return pixel coords of the center of a cell"""
    return col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2


def draw_board(surface):
    # Background
    surface.fill(WHITE)
    # Grid lines
    for i in range(GRID_SIZE + 1):
        pygame.draw.line(surface, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, BOARD_SIZE), 2)
        pygame.draw.line(surface, BLACK, (0, i * CELL_SIZE), (BOARD_SIZE, i * CELL_SIZE), 2)



def draw_pieces(surface):
    radius = CELL_SIZE // 3
    for piece in dots:
        center = cell_center(piece["row"], piece["col"])
        pygame.draw.circle(surface, piece["color"], center, radius)



def draw_highlights(surface):
    radius = CELL_SIZE // 3 + 6
    for (row, col) in highlights:
        center = cell_center(row, col)
        pygame.draw.circle(surface, HIGHLIGHT, center, radius, 3)



def draw_arrow(surface, start_rc, end_rc):
    start_px = cell_center(*start_rc)
    end_px = cell_center(*end_rc)

    # Draw line
    pygame.draw.line(surface, BLACK, start_px, end_px, 4)

    # Draw arrowhead
    angle = math.atan2(end_px[1] - start_px[1], end_px[0] - start_px[0])
    head_length = 18
    head_angle = math.pi / 7  # ~25 degrees on each side
    left = (end_px[0] - head_length * math.cos(angle - head_angle),
            end_px[1] - head_length * math.sin(angle - head_angle))
    right = (end_px[0] - head_length * math.cos(angle + head_angle),
             end_px[1] - head_length * math.sin(angle + head_angle))
    pygame.draw.polygon(surface, BLACK, [end_px, left, right])


# ---------------------- MAIN FUNCTION ----------------------

def main():
    pygame.init()
    pygame.display.set_caption("Choko Board Editor")
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, FONT_SIZE)

    # Create buttons
    buttons = []
    y = BUTTON_PADDING
    for mode in action_modes:
        btn_rect = (BOARD_SIZE + BUTTON_PADDING, y, PANEL_WIDTH - 2 * BUTTON_PADDING, BUTTON_HEIGHT)
        buttons.append(Button(btn_rect, mode.replace("_", " "), mode))
        y += BUTTON_HEIGHT + BUTTON_PADDING

    current_mode = "BLUE_PIECE"
    arrow_start = None  # For two-click arrow creation

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    # Save screenshot
                    pygame.image.save(screen.subsurface((0, 0, BOARD_SIZE, BOARD_SIZE)), "choko_board.png")
                    print("Board saved as choko_board.png")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
                if mouse_pos[0] < BOARD_SIZE:  # Within board
                    col = mouse_pos[0] // CELL_SIZE
                    row = mouse_pos[1] // CELL_SIZE

                    if current_mode == "BLUE_PIECE":
                        dots.append({"row": row, "col": col, "color": BLUE})
                    elif current_mode == "RED_PIECE":
                        dots.append({"row": row, "col": col, "color": RED})
                    elif current_mode == "HIGHLIGHT":
                        if (row, col) in highlights:
                            highlights.remove((row, col))  # toggle off
                        else:
                            highlights.append((row, col))
                    elif current_mode == "ARROW":
                        if arrow_start is None:
                            arrow_start = (row, col)
                        else:
                            if (row, col) != arrow_start:
                                arrows.append({"start": arrow_start, "end": (row, col)})
                            arrow_start = None  # reset for next arrow
                else:
                    # Side panel click: check buttons
                    for btn in buttons:
                        if btn.is_over(mouse_pos):
                            current_mode = btn.mode
                            arrow_start = None  # reset any arrow in progress
                            break

        # Update hover states
        for btn in buttons:
            btn.hover = btn.is_over(mouse_pos)

        # Drawing
        board_surface = pygame.Surface((BOARD_SIZE, BOARD_SIZE))
        draw_board(board_surface)
        draw_pieces(board_surface)
        draw_highlights(board_surface)
        for arrow in arrows:
            draw_arrow(board_surface, arrow["start"], arrow["end"])
        # If arrow in progress, draw preview
        if current_mode == "ARROW" and arrow_start is not None and mouse_pos[0] < BOARD_SIZE:
            preview_end = (mouse_pos[1] // CELL_SIZE, mouse_pos[0] // CELL_SIZE)
            draw_arrow(board_surface, arrow_start, preview_end)

        # Draw board to screen
        screen.blit(board_surface, (0, 0))

        # Draw control panel
        panel_rect = pygame.Rect(BOARD_SIZE, 0, PANEL_WIDTH, WIN_HEIGHT)
        pygame.draw.rect(screen, LIGHT_GRAY, panel_rect)
        for btn in buttons:
            btn.draw(screen, font, active=(btn.mode == current_mode))

        # Frame update
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
