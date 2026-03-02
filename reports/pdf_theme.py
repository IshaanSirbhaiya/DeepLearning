"""
SafeEdge PDF Theme — Branded report template using fpdf2.
"""

from fpdf import FPDF
from datetime import datetime


class SafeEdgePDF(FPDF):
    """Branded PDF with SafeEdge fire-safety styling."""

    # Brand colors (RGB)
    RED = (220, 53, 69)
    DARK = (33, 37, 41)
    BLUE = (13, 110, 253)
    ORANGE = (253, 126, 20)
    YELLOW = (255, 193, 7)
    GREEN = (25, 135, 84)
    LIGHT = (248, 249, 250)
    WHITE = (255, 255, 255)
    GRAY = (108, 117, 125)

    def __init__(self, title: str, subtitle: str, audience: str):
        super().__init__()
        self.report_title = title
        self.report_subtitle = subtitle
        self.audience = audience
        self.set_auto_page_break(auto=True, margin=20)
        self.alias_nb_pages()

    def header(self):
        if self.page_no() == 1:
            return  # cover page has no header
        self.set_fill_color(*self.DARK)
        self.rect(0, 0, 210, 14, "F")
        self.set_text_color(*self.WHITE)
        self.set_font("Helvetica", "B", 8)
        self.set_y(3)
        self.cell(0, 8, f"SafeEdge Intelligence  |  {self.report_title}", align="C")
        self.ln(14)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*self.GRAY)
        self.cell(
            0, 10,
            f"SafeEdge  |  {self.audience}  |  Page {self.page_no()}/{{nb}}",
            align="C",
        )

    def add_cover_page(self):
        self.add_page()
        # Dark background block
        self.set_fill_color(*self.DARK)
        self.rect(0, 0, 210, 297, "F")

        # Red accent bar
        self.set_fill_color(*self.RED)
        self.rect(0, 80, 210, 6, "F")

        # Title
        self.set_text_color(*self.WHITE)
        self.set_font("Helvetica", "B", 28)
        self.set_y(95)
        self.multi_cell(0, 14, self.report_title, align="C")

        # Subtitle
        self.set_font("Helvetica", "", 14)
        self.set_text_color(180, 180, 180)
        self.ln(5)
        self.multi_cell(0, 8, self.report_subtitle, align="C")

        # Audience tag
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*self.RED)
        self.ln(10)
        self.cell(0, 8, f"Prepared for: {self.audience}", align="C")

        # Date
        self.set_font("Helvetica", "", 10)
        self.set_text_color(150, 150, 150)
        self.ln(15)
        self.cell(0, 8, datetime.now().strftime("%B %d, %Y"), align="C")

        # Footer branding
        self.set_y(250)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*self.WHITE)
        self.cell(0, 8, "SafeEdge", align="C")
        self.ln(6)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, "Edge-Based Fire Safety Intelligence System", align="C")
        self.ln(6)
        self.cell(0, 8, "DLW 2026  |  Track 3: AI in Security  |  NTU Singapore", align="C")

    def add_section(self, title: str):
        self.add_page()
        # Red accent bar
        self.set_fill_color(*self.RED)
        self.rect(10, self.get_y(), 4, 10, "F")
        self.set_x(18)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*self.DARK)
        self.cell(0, 10, title)
        self.ln(10)

    def add_subsection(self, title: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*self.DARK)
        self.cell(0, 7, title)
        self.ln(8)

    def add_narrative(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def add_chart(self, image_path: str, caption: str = "", width: int = 180):
        x = (210 - width) / 2
        self.image(image_path, x=x, w=width)
        if caption:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*self.GRAY)
            self.cell(0, 6, caption, align="C")
            self.ln(8)
        else:
            self.ln(4)

    def add_key_stat(self, label: str, value: str, color=None):
        if color is None:
            color = self.RED
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(*color)
        self.cell(45, 12, value, align="C")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*self.GRAY)
        self.cell(45, 12, label, align="L")

    def add_stat_row(self, stats: list):
        """Add a row of key stats. stats = [(label, value, color), ...]"""
        y = self.get_y()
        col_width = 190 / len(stats)
        for label, value, color in stats:
            # Box
            self.set_fill_color(245, 245, 245)
            self.rect(self.get_x(), y, col_width - 2, 22, "F")
            # Value
            self.set_font("Helvetica", "B", 20)
            self.set_text_color(*color)
            self.set_xy(self.get_x(), y + 1)
            self.cell(col_width - 2, 10, str(value), align="C")
            # Label
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*self.GRAY)
            self.set_xy(self.get_x() - col_width + 2, y + 12)
            self.cell(col_width - 2, 8, label, align="C")
            self.set_xy(self.get_x() + 2, y)
        self.set_y(y + 26)

    def add_table(self, headers: list, rows: list):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*self.DARK)
        self.set_text_color(*self.WHITE)
        col_w = 190 / len(headers)
        for h in headers:
            self.cell(col_w, 7, str(h), border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*self.DARK)
        for i, row in enumerate(rows):
            if i % 2 == 0:
                self.set_fill_color(*self.LIGHT)
            else:
                self.set_fill_color(*self.WHITE)
            for val in row:
                self.cell(col_w, 6, str(val), border=1, fill=True, align="C")
            self.ln()
        self.ln(4)
