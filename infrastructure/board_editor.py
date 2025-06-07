import pygame
import sys
import math

"""
Choko Board Editor
------------------
An interactive editor for creating clean, publication‑ready diagrams of Choko board
positions.  Features:
• Place blue or red pieces on a 5×5 grid.
• Highlight pieces with a black circle.
• Draw arrows between squares.
• Thicker grid lines, bold outer border, and a gentle off‑white board background.
• Double‑click any square to erase everything on it.
• "Erase" tool in the side panel for single‑click deletion.
• Press **S** to save a PNG snapshot of the board area.

**Update 2025‑06‑06**
• Arrow shaft shortened so the tip of the arrowhead no longer clips through the line.
• Added a 4‑pixel black border around the entire board for crisper export.
• Board background changed from pure white to off‑white for softer contrast.

Dependencies: Pygame ≥ 2.0 (for the ``event.clicks`` double‑click attribute).
"""

# ---------------------- CONFIGURATION ----------------------
GRID_SIZE   = 5      # 5x5 board
CELL_SIZE   = 100    # pixels per cell
BOARD_SIZE  = GRID_SIZE * CELL_SIZE  # width/height of board portion
PANEL_WIDTH = 230    # control‑panel width
WIN_HEIGHT  = BOARD_SIZE
WIN_WIDTH   = BOARD_SIZE + PANEL_WIDTH

# Colors (R, G, B)
OFF_WHITE  = (250, 240, 240)  # subtle off‑white for board background
WHITE      = (255, 255, 255)
BLACK      = (0, 0, 0)
LIGHT_GRAY = (230, 230, 230)
DARK_GRAY  = (140, 140, 140)
BLUE       = ( 30, 144, 255)
RED        = (220,  20,  60)
HIGHLIGHT  = BLACK   # outline colour for highlighted pieces

# UI button specifications
BUTTON_HEIGHT  = 50
BUTTON_PADDING = 15
FONT_SIZE      = 18

# Drawing modes
action_modes = [
    "BLUE_PIECE",
    "RED_PIECE",
    "HIGHLIGHT",
    "ARROW",
    "ERASE"
]


class Button:
    """Simple rectangular button widget"""
    def __init__(self, rect, text, mode):
        self.rect   = pygame.Rect(rect)
        self.text   = text
        self.mode   = mode
        self.hover  = False

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
# Each item is stored in board coordinates (row, col)

dots       = []  # list of {row, col, color}
highlights = []  # list of (row, col)
arrows     = []  # list of {start: (row, col), end: (row, col)}


# ---------------------- DRAWING HELPERS ----------------------

def cell_center(row: int, col: int):
    """Pixel coordinates of centre of a board cell."""
    return col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2


def draw_board(surface):
    """Draw the board grid with thicker lines, off‑white fill, and outer border."""
    surface.fill(OFF_WHITE)
    line_width = 4  # thicker grid lines
    for i in range(GRID_SIZE + 1):
        # Vertical
        pygame.draw.line(surface, BLACK,
                         (i * CELL_SIZE, 0),
                         (i * CELL_SIZE, BOARD_SIZE),
                         line_width)
        # Horizontal
        pygame.draw.line(surface, BLACK,
                         (0, i * CELL_SIZE),
                         (BOARD_SIZE, i * CELL_SIZE),
                         line_width)
    # Bold outer border
    pygame.draw.rect(surface, BLACK, surface.get_rect(), 4)


def draw_pieces(surface):
    radius = CELL_SIZE // 3
    for piece in dots:
        centre = cell_center(piece["row"], piece["col"])
        pygame.draw.circle(surface, piece["color"], centre, radius)


def draw_highlights(surface):
    radius = CELL_SIZE // 3 + 6
    for (row, col) in highlights:
        centre = cell_center(row, col)
        pygame.draw.circle(surface, HIGHLIGHT, centre, radius, 3)


def draw_arrow(surface, start_rc, end_rc):
    """Draw an arrow from one square centre to another with a non‑overlapping head."""
    start_px = cell_center(*start_rc)
    end_px   = cell_center(*end_rc)

    # Compute geometry
    angle      = math.atan2(end_px[1] - start_px[1], end_px[0] - start_px[0])
    head_len   = 18
    head_angle = math.pi / 7  # ≈25° half‑angle of arrowhead

    # Base of arrowhead (so shaft stops before the head)
    shaft_end = (end_px[0] - head_len * math.cos(angle),
                 end_px[1] - head_len * math.sin(angle))

    # Main shaft
    pygame.draw.line(surface, BLACK, start_px, shaft_end, 4)

    # Arrow head
    left  = (end_px[0] - head_len * math.cos(angle - head_angle),
             end_px[1] - head_len * math.sin(angle - head_angle))
    right = (end_px[0] - head_len * math.cos(angle + head_angle),
             end_px[1] - head_len * math.sin(angle + head_angle))
    pygame.draw.polygon(surface, BLACK, [end_px, left, right])


# ---------------------- ERASING UTIL ----------------------

def erase_at_cell(row, col):
    """Remove any pieces, highlights, or arrows that occupy ``(row, col)``."""
    global dots, highlights, arrows
    dots       = [d for d in dots if not (d["row"] == row and d["col"] == col)]
    highlights = [h for h in highlights if h != (row, col)]
    arrows     = [a for a in arrows  if a["start"] != (row, col) and a["end"] != (row, col)]


# ---------------------- MAIN FUNCTION ----------------------

def main():
    pygame.init()
    pygame.display.set_caption("Choko Board Editor")
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont(None, FONT_SIZE)

    # Build side‑panel buttons
    buttons = []
    y = BUTTON_PADDING
    for mode in action_modes:
        btn_rect = (BOARD_SIZE + BUTTON_PADDING,
                    y,
                    PANEL_WIDTH - 2 * BUTTON_PADDING,
                    BUTTON_HEIGHT)
        buttons.append(Button(btn_rect, mode.replace("_", " "), mode))
        y += BUTTON_HEIGHT + BUTTON_PADDING

    current_mode = "BLUE_PIECE"
    arrow_start  = None  # first square selected for arrow drawing

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                # Save board‑only screenshot
                pygame.image.save(screen.subsurface((0, 0, BOARD_SIZE, BOARD_SIZE)),
                                   "choko_board.png")
                print("Board saved as choko_board.png")

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicks = getattr(event, "clicks", 1)  # Pygame ≥2 provides clicks count

                if mouse_pos[0] < BOARD_SIZE:  # click on board area
                    col = mouse_pos[0] // CELL_SIZE
                    row = mouse_pos[1] // CELL_SIZE

                    # Double‑click always erases
                    if clicks >= 2:
                        erase_at_cell(row, col)
                        arrow_start = None  # cancel any arrow‑in‑progress
                        continue  # skip further processing for this event

                    # Single‑click behaviour depends on mode
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
                            arrow_start = None
                    elif current_mode == "ERASE":
                        erase_at_cell(row, col)
                        arrow_start = None

                else:  # Click on control panel
                    for btn in buttons:
                        if btn.is_over(mouse_pos):
                            current_mode = btn.mode
                            arrow_start  = None  # cancel any arrow‑in‑progress
                            break

        # Update hover states for buttons
        for btn in buttons:
            btn.hover = btn.is_over(mouse_pos)

        # --- Drawing ---
        board_surface = pygame.Surface((BOARD_SIZE, BOARD_SIZE))
        draw_board(board_surface)
        draw_pieces(board_surface)
        draw_highlights(board_surface)
        for arrow in arrows:
            draw_arrow(board_surface, arrow["start"], arrow["end"])
        # Arrow preview
        if current_mode == "ARROW" and arrow_start is not None and mouse_pos[0] < BOARD_SIZE:
            preview_end = (mouse_pos[1] // CELL_SIZE, mouse_pos[0] // CELL_SIZE)
            draw_arrow(board_surface, arrow_start, preview_end)

        # Blit board to main screen
        screen.blit(board_surface, (0, 0))

        # Draw side panel and buttons
        panel_rect = pygame.Rect(BOARD_SIZE, 0, PANEL_WIDTH, WIN_HEIGHT)
        pygame.draw.rect(screen, LIGHT_GRAY, panel_rect)
        for btn in buttons:
            btn.draw(screen, font, active=(btn.mode == current_mode))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()