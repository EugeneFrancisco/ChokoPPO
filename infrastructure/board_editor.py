import pygame
import sys
import math

"""
Choko Board Editor
==================
An interactive editor for creating clean, publication‑ready diagrams of Choko board
positions.

Features
--------
* Place blue or red pieces on a 5 × 5 grid.
* Highlight pieces with a black circle.
* Draw arrows between squares (shaft auto‑shortens so the arrowhead doesn’t clip).
* Thicker 4‑px grid lines **and** a 4‑px outer border.
* Gentle off‑white board background (#FAF8F0).
* Double‑click any square to erase everything on it, or use the *Erase* tool.
* Save the current board to a high‑resolution PNG with the **S** key.
* **NEW:** Row labels **A–E** (left) and column labels **1–5** (top) exactly like
  chess/Go co‑ordinates.

Run with: `python choko_board_editor.py`
"""

# ---------------------------- CONFIG ----------------------------
GRID_SIZE = 5           # 5×5 board
CELL_SIZE = 100         # pixels per cell
BOARD_SIZE = GRID_SIZE * CELL_SIZE
MARGIN = 40            # room for coordinate labels
PANEL_WIDTH = 220       # control panel width
BORDER = 4             # outer border / grid line thickness

WIN_HEIGHT = MARGIN + BOARD_SIZE + MARGIN  # bottom margin == top margin
WIN_WIDTH = MARGIN + BOARD_SIZE + PANEL_WIDTH

# Colours (R,G,B)
OFF_WHITE = (250, 248, 240)
BLACK      = (0, 0, 0)
LIGHT_GRAY = (230, 230, 230)
RED        = (220,  50,  50)
BLUE       = ( 40, 110, 250)
ARROW_HEAD = BLACK

ROW_LABELS = ["A", "B", "C", "D", "E"]
COL_LABELS = ["1", "2", "3", "4", "5"]

# --------------------------- STATE -----------------------------
pygame.init()
FONT   = pygame.font.SysFont("Arial", 24)
SMALLF = pygame.font.SysFont("Arial", 20)

actions = ["BLUE", "RED", "HIGHLIGHT", "ARROW", "ERASE"]
current_action = "BLUE"

# Each cell holds either None, "R", "B" or "H" (highlight flag alongside piece)
# Arrows are stored as list of (row_from, col_from, row_to, col_to)
board   = [[None]*GRID_SIZE for _ in range(GRID_SIZE)]
highlit = [[False]*GRID_SIZE for _ in range(GRID_SIZE)]
arrows  = []
arrow_in_progress = None  # (row, col) when dragging

double_click_time = 350   # ms within which two clicks count as double
last_click_time   = 0
last_click_cell   = None

# ------------------------ HELPER FUNCTIONS ----------------------

def board_to_pixel(row: int, col: int):
    """Return centre pixel of a square"""
    x = MARGIN + col * CELL_SIZE + CELL_SIZE // 2
    y = MARGIN + row * CELL_SIZE + CELL_SIZE // 2
    return x, y


def pixel_to_cell(x: int, y: int):
    """Convert mouse pixel to (row,col) or None if outside board"""
    if x < MARGIN or y < MARGIN or \
       x >= MARGIN + BOARD_SIZE or y >= MARGIN + BOARD_SIZE:
        return None
    col = (x - MARGIN) // CELL_SIZE
    row = (y - MARGIN) // CELL_SIZE
    return row, col


def draw_grid(screen):
    """Draw grid lines and outer border"""
    # Background
    pygame.draw.rect(screen, OFF_WHITE,
                     pygame.Rect(MARGIN, MARGIN, BOARD_SIZE, BOARD_SIZE))

    # Grid lines
    for i in range(GRID_SIZE + 1):
        # Vertical
        x = MARGIN + i * CELL_SIZE
        pygame.draw.line(screen, BLACK, (x, MARGIN), (x, MARGIN + BOARD_SIZE), BORDER)
        # Horizontal
        y = MARGIN + i * CELL_SIZE
        pygame.draw.line(screen, BLACK, (MARGIN, y), (MARGIN + BOARD_SIZE, y), BORDER)

    # Outer border (drawn again to cover corners)
    pygame.draw.rect(screen, BLACK,
                     pygame.Rect(MARGIN, MARGIN, BOARD_SIZE, BOARD_SIZE), BORDER)

    # Row labels (A‑E) – centred vertically per row, left of board
    for r, label in enumerate(ROW_LABELS):
        text = FONT.render(label, True, BLACK)
        text_rect = text.get_rect()
        text_rect.centery = MARGIN + r * CELL_SIZE + CELL_SIZE // 2
        text_rect.right = MARGIN - 8  # 8‑px gap from left border
        screen.blit(text, text_rect)

    # Column labels (1‑5) – centred horizontally per col, above board
    for c, label in enumerate(COL_LABELS):
        text = FONT.render(label, True, BLACK)
        text_rect = text.get_rect()
        text_rect.centerx = MARGIN + c * CELL_SIZE + CELL_SIZE // 2
        text_rect.bottom = MARGIN - 8  # 8‑px gap above top border
        screen.blit(text, text_rect)


def draw_pieces(screen):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            piece = board[r][c]
            if piece:
                color = BLUE if piece == "B" else RED
                x, y = board_to_pixel(r, c)
                pygame.draw.circle(screen, color, (x, y), CELL_SIZE // 3)
            if highlit[r][c]:
                x, y = board_to_pixel(r, c)
                pygame.draw.circle(screen, BLACK, (x, y), CELL_SIZE // 3 + 6, 3)


def draw_arrows(screen):
    for (r1, c1, r2, c2) in arrows:
        x1, y1 = board_to_pixel(r1, c1)
        x2, y2 = board_to_pixel(r2, c2)
        draw_arrow(screen, x1, y1, x2, y2)


# Arrow util constants
ARROW_SHAFT_WIDTH = 6
ARROW_HEAD_LEN    = 20
ARROW_HEAD_WIDTH  = 16


def draw_arrow(screen, x1, y1, x2, y2):
    """Draw an arrow whose shaft stops short of the tip so arrowhead tip is flush."""
    dx, dy = x2 - x1, y2 - y1
    distance = math.hypot(dx, dy)
    if distance == 0:
        return
    # unit vector along line
    ux, uy = dx / distance, dy / distance
    # shorten shaft so arrowhead touches but does not overlap
    shaft_len = distance - ARROW_HEAD_LEN
    sx2 = x1 + ux * shaft_len
    sy2 = y1 + uy * shaft_len
    # Draw shaft
    pygame.draw.line(screen, BLACK, (x1, y1), (sx2, sy2), ARROW_SHAFT_WIDTH)
    # Arrowhead – triangle
    left = (sx2 + uy * (ARROW_HEAD_WIDTH / 2), sy2 - ux * (ARROW_HEAD_WIDTH / 2))
    right = (sx2 - uy * (ARROW_HEAD_WIDTH / 2), sy2 + ux * (ARROW_HEAD_WIDTH / 2))
    tip = (x2, y2)
    pygame.draw.polygon(screen, ARROW_HEAD, [left, right, tip])


# ------------------------ RENDER PANEL --------------------------
BTN_H = 50
BTN_MARGIN = 10


def render_panel(screen):
    panel_x = MARGIN + BOARD_SIZE
    pygame.draw.rect(screen, LIGHT_GRAY,
                     pygame.Rect(panel_x, 0, PANEL_WIDTH, WIN_HEIGHT))

    for i, action in enumerate(actions):
        y = BTN_MARGIN + i * (BTN_H + BTN_MARGIN)
        btn_rect = pygame.Rect(panel_x + BTN_MARGIN, y, PANEL_WIDTH - 2*BTN_MARGIN, BTN_H)
        color = (180, 180, 180) if current_action == action else (210, 210, 210)
        pygame.draw.rect(screen, color, btn_rect, border_radius=6)
        label = SMALLF.render(action, True, BLACK)
        label_rect = label.get_rect(center=btn_rect.center)
        screen.blit(label, label_rect)


# ------------------------ SAVE FUNCTION -------------------------

def save_board(screen):
    filename = "choko_board.png"
    pygame.image.save(screen, filename)
    print(f"Saved board to {filename}")


# ---------------------- EVENT HANDLING -------------------------

def handle_click(row, col):
    global current_action
    if row is None:
        return

    global last_click_time, last_click_cell
    now = pygame.time.get_ticks()
    double_click = (last_click_cell == (row, col)) and (now - last_click_time <= double_click_time)
    last_click_time = now
    last_click_cell = (row, col)

    if double_click or current_action == "ERASE":
        # clear square & arrows connected to it
        board[row][col] = None
        highlit[row][col] = False
        arrows[:] = [a for a in arrows if not ((a[0], a[1]) == (row, col) or (a[2], a[3]) == (row, col))]
        return

    if current_action == "BLUE":
        board[row][col] = "B"
    elif current_action == "RED":
        board[row][col] = "R"
    elif current_action == "HIGHLIGHT":
        highlit[row][col] = not highlit[row][col]


def main():
    global current_action, arrow_in_progress
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Choko Board Editor")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                # Check if click is in panel
                if x >= MARGIN + BOARD_SIZE:
                    # Which button?
                    panel_relative_y = y - BTN_MARGIN
                    if panel_relative_y >= 0:
                        idx = panel_relative_y // (BTN_H + BTN_MARGIN)
                        if 0 <= idx < len(actions):
                            current_action = actions[idx]
                    continue

                cell = pixel_to_cell(x, y)
                if current_action == "ARROW":
                    if cell:
                        arrow_in_progress = cell
                else:
                    handle_click(*cell)

            elif event.type == pygame.MOUSEBUTTONUP and current_action == "ARROW":
                if arrow_in_progress:
                    cell_from = arrow_in_progress
                    cell_to   = pixel_to_cell(*event.pos)
                    if cell_to and cell_to != cell_from:
                        arrows.append((*cell_from, *cell_to))
                    arrow_in_progress = None

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_board(screen)

        # --------------- DRAW ---------------
        screen.fill(WHITE := (255, 255, 255))
        draw_grid(screen)
        draw_arrows(screen)
        draw_pieces(screen)
        render_panel(screen)

        # Temp arrow preview while dragging
        if current_action == "ARROW" and arrow_in_progress and pygame.mouse.get_pressed()[0]:
            x1, y1 = board_to_pixel(*arrow_in_progress)
            x2, y2 = pygame.mouse.get_pos()
            draw_arrow(screen, x1, y1, x2, y2)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
