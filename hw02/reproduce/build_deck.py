"""Build presentation/slides.pptx from slides_orig.pptx.

Main part = explanation of the paper (original slides + new explainer slides).
Appendix  = our reproduction plots (details live in the report).
Run: python build_deck.py   (needs python-pptx, Pillow)
"""
import copy
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
SRC, DST = ROOT / "presentation" / "slides_orig.pptx", ROOT / "presentation" / "slides.pptx"

INK, ACC, MUTED, CARD, GRID = "1F2A33", "D2622A", "5C6570", "ECE7DD", "CFC8B8"
SOFT, LACC, WHITE = "F2CDB6", "E9A883", "FFFFFF"
X0, XR = 812800, 11696192  # content left / right edge (EMU)
TOP = 1615439


def I(v):  # inches -> EMU
    return int(v * 914400)


def rgb(h):
    return RGBColor.from_string(h)


# ---------- low-level helpers ----------
def _clear_tf(tf):
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    p0 = tf.paragraphs[0]
    for r in list(p0.runs):
        r._r.getparent().remove(r._r)


def _apply_run(run, text, st, size, color, bold):
    run.text = text
    f = run.font
    f.name = "Inter"
    f.size = Pt(st.get("s", size))
    f.bold = st.get("b", bold)
    f.color.rgb = rgb(st.get("c", color))
    # python-pptx here has no Font.subscript/superscript — set OOXML baseline
    if st.get("sub") or st.get("sup"):
        rPr = run._r.get_or_add_rPr()
        rPr.set("baseline", "-25000" if st.get("sub") else "30000")


def set_text(shape, text):
    """Replace text keeping first run formatting (used on cloned deck shapes)."""
    p = shape.text_frame.paragraphs[0]
    p.runs[0].text = text
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
    for extra in shape.text_frame.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)


def set_runs(shape, runs, size=None, color=INK, bold=False, align=None):
    """Replace shape text with styled runs: [(text, {b,c,s,sub,sup}), ...]."""
    tf = shape.text_frame
    _clear_tf(tf)
    p = tf.paragraphs[0]
    if align is not None:
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
    base = size or (p.runs[0].font.size.pt if False else 18)
    # size from existing title if not given
    if size is None:
        try:
            base = shape.text_frame.paragraphs[0].font.size.pt  # may be None
        except Exception:
            base = 18
        base = base or 18
    for text, st in runs:
        r = p.add_run()
        _apply_run(r, text, st, base, color, bold)


def clone(shape, slide):
    el = copy.deepcopy(shape._element)
    slide.shapes._spTree.append(el)
    return slide.shapes[-1]


def tb(slide, x, y, w, h, paras, size=16, color=INK, bold=False, align="l", anchor="t", gap=0.0, line=1.25):
    """Textbox. paras: str | list of (str | list of (text, {b,c,s,sub,sup}))."""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        p.line_spacing = line
        if i:
            p.space_before = Pt(gap)
        for text, st in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            _apply_run(r, text, st, size, color, bold)
    return box


# short formula fragments used across slides
def H(name, **kw):  # h with subscript name
    return [("h", kw), (name, {**kw, "sub": True})]


def X(name, **kw):
    return [("x", kw), (name, {**kw, "sub": True})]


def J(name, **kw):
    return [("J", kw), (name, {**kw, "sub": True})]


def D(name, **kw):  # delta
    return [("δ", kw), (name, {**kw, "sub": True})]


def A_(name, **kw):
    return [("A", kw), (name, {**kw, "sub": True})]


def R(name, **kw):
    return [("r", kw), (name, {**kw, "sub": True})]


def B_(name, **kw):
    return [("B", kw), (name, {**kw, "sub": True})]


def C_(name, **kw):
    return [("C", kw), (name, {**kw, "sub": True})]


def b_(name, **kw):
    return [("b", kw), (name, {**kw, "sub": True})]


def bare(shape):
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def box(slide, kind, x, y, w, h, fill=None, line=None, lw=1.0):
    s = slide.shapes.add_shape(kind, x, y, w, h)
    bare(s)
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(lw)
    else:
        s.line.fill.background()
    return s


def card(slide, x, y, w, h, fill=CARD):
    s = box(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill)
    s.adjustments[0] = 0.07
    return s


def label(shape, text_or_runs, size=14, color=INK, bold=True):
    tf = shape.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _clear_tf(tf)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    runs = [(text_or_runs, {})] if isinstance(text_or_runs, str) else text_or_runs
    for text, st in runs:
        r = p.add_run()
        _apply_run(r, text, st, size, color, bold)


def arrow(slide, x1, y1, x2, y2, color=ACC, lw=1.75, head=True):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    bare(c)
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(lw)
    if head:
        etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"), type="triangle", w="med", len="med")
    return c


def grid(slide, x, y, n, s, fill_of, text_of=None, size=12, text_color=None):
    """n x n grid. text_of(i,j) -> str | runs | (text_or_runs, color) | None."""
    for i in range(n):
        for j in range(n):
            f = fill_of(i, j)
            c = box(slide, MSO_SHAPE.RECTANGLE, x + j * s, y + i * s, s, s, f, GRID if not f else BG, 1.5 if f else 0.75)
            t = text_of(i, j) if text_of else None
            if not t:
                continue
            col = text_color or (WHITE if f == ACC else INK)
            if isinstance(t, tuple) and len(t) == 2 and isinstance(t[1], str) and len(t[1]) == 6:
                t, col = t
            label(c, t, size, col)


BG = "F7F5F0"


# ---------- slide scaffolding ----------
class Deck:
    def __init__(self):
        self.prs = Presentation(SRC)
        self.orig = list(self.prs.slides)
        self.title_src = self.orig[7].shapes[0]  # title placeholder (slide 8)
        self.cap_src = self.orig[3].shapes[2]  # caption box (slide 4)
        self.stat = self.orig[9]  # stat cards (slide 10)
        self.bg = self.orig[3]._element.cSld.find(qn("p:bg"))
        self.by_key = {}

    def new(self, key, title, caption=None, cap_y=None):
        s = self.prs.slides.add_slide(self.orig[3].slide_layout)
        for ph in list(s.shapes):
            ph._element.getparent().remove(ph._element)
        if self.bg is not None:
            s._element.cSld.insert(0, copy.deepcopy(self.bg))
        set_text(clone(self.title_src, s), title)
        if caption:
            cap = clone(self.cap_src, s)
            set_text(cap, caption)
            cap.top = cap_y or I(5.9)
            cap.height = I(0.9)
        self.by_key[key] = s
        return s


def head_card(s, x, y, w, h, head, body, size=17, head_color=ACC, hsize=19):
    card(s, x, y, w, h)
    tb(s, x + I(0.3), y + I(0.28), w - I(0.6), I(0.4), head, size=hsize, color=head_color, bold=True)
    tb(s, x + I(0.3), y + I(0.85), w - I(0.6), h - I(1.0), body, size=size, gap=6, line=1.25)


def big_card(s, x, y, w, h, big, name, small, big_size=48):
    card(s, x, y, w, h)
    tb(s, x + I(0.3), y + I(0.25), w - I(0.6), I(0.9), big, size=big_size, color=ACC, bold=True)
    tb(s, x + I(0.3), y + I(1.3), w - I(0.6), I(0.4), name, size=19, bold=True)
    tb(s, x + I(0.3), y + I(1.75), w - I(0.6), h - I(1.85), small, size=16, color=INK, line=1.25)


# ---------- explainer slides ----------
def slide_system(d):
    s = d.new("system", "Все шаги — одна система уравнений",
              "Каждое уравнение связано только с предыдущим, поэтому ненулевые блоки лежат на двух диагоналях. "
              "Такую систему можно решить быстро.", cap_y=I(6.0))
    w = I(5.3)
    card(s, X0, TOP, w, I(3.85))
    tb(s, X0 + I(0.3), TOP + I(0.28), w - I(0.6), I(0.4), "Что нужно найти", size=19, color=ACC, bold=True)
    eqs = [
        H("1", b=True, s=22) + [(" = f(0, ", {"b": True, "s": 22})] + X("1", b=True, s=22) + [(")", {"b": True, "s": 22})],
        H("2", b=True, s=22) + [(" = f(", {"b": True, "s": 22})] + H("1", b=True, s=22) + [(", ", {"b": True, "s": 22})]
        + X("2", b=True, s=22) + [(")", {"b": True, "s": 22})],
        [("…", {"b": True, "s": 22})],
        H("L", b=True, s=22) + [(" = f(", {"b": True, "s": 22})] + H("prev", b=True, s=22) + [(", ", {"b": True, "s": 22})]
        + X("L", b=True, s=22) + [(")", {"b": True, "s": 22})],
    ]
    tb(s, X0 + I(0.3), TOP + I(0.95), w - I(0.6), I(2.0), eqs, size=22, gap=2, line=1.1)
    tb(s, X0 + I(0.3), TOP + I(2.95), w - I(0.6), I(0.7),
       "Неизвестны все h сразу, это L уравнений. Метод Ньютона заменяет их линейными.", size=14, color=MUTED)
    x = I(7.15)
    tb(s, x, TOP, I(5.2), I(0.4), "Один шаг Ньютона: линейная система", size=16, color=MUTED, bold=True)
    n, sz, y = 5, I(0.52), TOP + I(0.6)

    def f(i, j):
        return CARD if i == j else (ACC if j == i - 1 else None)

    def t(i, j):
        if i == j:
            return "I"
        if j == i - 1:
            return [("J", {}), (str(i + 1), {"sub": True})]
        return None

    grid(s, x, y, n, sz, f, t, size=13)
    vx = x + n * sz + I(0.2)
    for k in range(n):
        c = box(s, MSO_SHAPE.RECTANGLE, vx, y + k * sz, sz, sz, LACC, BG, 1.5)
        label(c, D(str(k + 1)), 13)
    tb(s, vx + sz + I(0.08), y, I(0.4), n * sz, "=", size=26, bold=True, anchor="m", align="c")
    rx = vx + sz + I(0.55)
    for k in range(n):
        c = box(s, MSO_SHAPE.RECTANGLE, rx, y + k * sz, sz, sz, SOFT, BG, 1.5)
        label(c, R(str(k + 1)), 13)
    tb(s, x, y + n * sz + I(0.2), I(5.4), I(0.7),
       [[("I", {"b": True}), (" — единичная матрица. ", {})]
        + J("t", b=True, c=ACC) + [(" — якобиан шага t. ", {}), ("δ", {"b": True}),
         (" — поправки, ", {}), ("r", {"b": True}),
         (" — на сколько черновик промахнулся.", {})]], size=14, color=MUTED)


def slide_jacobian(d):
    s = d.new("jac", "Что такое якобиан",
              "Если якобиан диагональный, вся «таблица» — один вектор: каждое число состояния зависит только от самого себя.",
              cap_y=I(5.65))
    tb(s, X0, TOP, I(5.6), I(3.6),
       [[("Ячейка берёт старое состояние h и считает новое: ", {})]
        + H("next") + [(" = f(", {})] + H("prev") + [(", ", {})] + X("t") + [(").", {})],
        [("Якобиан J — таблица d × d. В строке i и столбце j стоит число: насколько сдвинется i-е число нового состояния, "
          "если чуть изменить j-е число старого.", {})],
        [("Ньютону J нужен на каждом шаге: по нему он заменяет ячейку касательной.", {})]],
       size=17, gap=14, line=1.3)
    n, sz, y = 6, I(0.38), TOP + I(0.1)
    xa, xb = I(7.0), I(10.0)
    grid(s, xa, y, n, sz, lambda i, j: LACC)
    grid(s, xb, y, n, sz, lambda i, j: ACC if i == j else None)
    tb(s, xa, y + n * sz + I(0.18), I(2.5), I(0.35), "плотный", size=16, bold=True)
    tb(s, xa, y + n * sz + I(0.55), I(2.5), I(0.35),
       [[("d", {}), ("2", {"sup": True}), (" чисел на шаг", {})]], size=14, color=MUTED)
    tb(s, xb, y + n * sz + I(0.18), I(2.5), I(0.35), "диагональный", size=16, bold=True)
    tb(s, xb, y + n * sz + I(0.55), I(2.5), I(0.35), "d чисел на шаг", size=14, color=MUTED)


def slide_cost(d):
    s = d.new("cost", "Почему якобиан должен быть диагональным",
              "Так же устроены Mamba и S4D: диагональная матрица состояния. ParaRNN переносит этот приём на нелинейные ячейки.",
              cap_y=I(5.2))
    w, h = I(5.64), I(3.2)
    card(s, X0, TOP, w, h)
    tb(s, X0 + I(0.3), TOP + I(0.25), w - I(0.6), I(0.9), "≈ 17 ГБ", size=48, color=ACC, bold=True)
    tb(s, X0 + I(0.3), TOP + I(1.3), w - I(0.6), I(0.4), "плотные якобианы", size=19, bold=True)
    tb(s, X0 + I(0.3), TOP + I(1.75), w - I(0.6), h - I(1.85),
       [[("Память L·d", {}), ("2", {"sup": True}),
         (", одно перемножение двух якобианов стоит d", {}), ("3", {"sup": True}),
         (" операций. Пример: L = 4096, d = 1024, fp32.", {})]], size=16)
    card(s, I(7.0), TOP, w, h)
    tb(s, I(7.3), TOP + I(0.25), w - I(0.6), I(0.9), "≈ 17 МБ", size=48, color=ACC, bold=True)
    tb(s, I(7.3), TOP + I(1.3), w - I(0.6), I(0.4), "диагональные якобианы", size=19, bold=True)
    tb(s, I(7.3), TOP + I(1.75), w - I(0.6), h - I(1.85),
       "Память L·d, перемножение — d операций, всё считается независимо по каналам. Тот же пример, в 1000 раз меньше.",
       size=16)


def slide_reduction(d):
    s = d.new("reduce", "Параллельная редукция: 3 раунда вместо 8 шагов", caption=None)
    # caption with real subscripts
    cap = clone(d.cap_src, s)
    set_runs(cap, [("Склейка двух кусков — снова шаг вида ", {})]
             + D("t") + [(" = ", {})] + A_("t") + [("·", {})] + D("prev") + [(" + ", {})] + R("t")
             + [(", поэтому формула одна и та же на всех раундах.", {})], size=18)
    cap.top = I(5.95)
    cap.height = I(0.9)
    x0, dx, y0, dy, dia = I(2.35), I(0.62), I(2.05), I(0.95), I(0.44)
    names = ["старт", "раунд 1", "раунд 2", "раунд 3"]

    def look(span):
        if span == 1:
            return CARD, INK
        if span < 4:
            return SOFT, INK
        if span < 8:
            return LACC, INK
        return ACC, WHITE

    for r in range(4):
        tb(s, X0, y0 + r * dy + I(0.1), I(1.4), I(0.3), names[r], size=15, color=MUTED, bold=True)
        for l in range(1, 9):
            cx = x0 + (l - 1) * dx
            if r:
                step = 2 ** (r - 1)
                px = x0 + (l - 1) * dx + dia // 2
                arrow(s, px, y0 + (r - 1) * dy + dia, px, y0 + r * dy, color=GRID, lw=1, head=False)
                if l - step >= 1:
                    qx = x0 + (l - step - 1) * dx + dia // 2
                    arrow(s, qx, y0 + (r - 1) * dy + dia, px, y0 + r * dy, color=ACC, lw=1.5)
    for r in range(4):  # nodes on top of lines
        for l in range(1, 9):
            span = min(l, 2 ** r)
            fill, txt = look(span)
            c = box(s, MSO_SHAPE.OVAL, x0 + (l - 1) * dx, y0 + r * dy, dia, dia, fill)
            label(c, str(span), 13, txt)
    card(s, I(7.7), TOP, I(4.74), I(4.1))
    tb(s, I(8.0), TOP + I(0.28), I(4.2), I(0.4), "Как читать схему", size=19, color=ACC, bold=True)
    tb(s, I(8.0), TOP + I(0.85), I(4.15), I(3.2),
       ["Число в кружке — сколько шагов уже склеено в этом месте.",
        "Раунд 1: каждый берёт соседа слева. Раунд 2: того, кто на 2 шага левее. Раунд 3: на 4 шага.",
        [("Охват удваивается, поэтому для длины L хватает log", {}), ("2", {"sub": True}),
         (" L раундов, а все кружки в раунде считаются одновременно.", {})]],
       size=15, gap=10)


def slide_backward(d):
    s = d.new("bwd", "Обратный проход: проще прямого")
    tb(s, X0, TOP, I(11.6), I(0.8),
       "Прямому проходу нужен Ньютон, потому что ячейка нелинейна по состоянию. Градиенты же зависят от состояния линейно.",
       size=18, line=1.3)
    f = box(s, MSO_SHAPE.ROUNDED_RECTANGLE, X0, TOP + I(1.1), I(11.56), I(1.1), CARD)
    f.adjustments[0] = 0.11
    label(f, [("∇", {})] + H("prev") + [(" L  =  ", {})]
          + J("t") + [("T", {"sup": True}), (" · ∇", {})] + H("next")
          + [(" L  +  ∂L/∂", {})] + H("prev"), 26, ACC)
    w = I(5.64)
    head_card(s, X0, TOP + I(2.55), w, I(2.2), "Один проход, без Ньютона",
              "Это та же линейная рекуррентность, поэтому годится тот же решатель редукции.", size=17)
    head_card(s, I(7.0), TOP + I(2.55), w, I(2.2), "Идёт справа налево",
              "Якобианы транспонируют, массив переворачивают и сдвигают. В коде это flip и roll.", size=17)


def slide_gru(d):
    s = d.new("gru", "GRU: формулы и его якобиан")
    w = I(6.2)
    card(s, X0, TOP, w, I(4.55))
    kw = {"s": 17, "b": True}
    paras = [
        [("z = σ(", kw)] + A_("z", **kw) + [("·", kw)] + H("prev", **kw) + [(" + ", kw)]
        + B_("z", **kw) + [("·x + ", kw)] + b_("z", **kw) + [(")", kw)],
        [("насколько обновлять состояние", {"s": 14, "c": MUTED})],
        [("r = σ(", kw)] + A_("r", **kw) + [("·", kw)] + H("prev", **kw) + [(" + ", kw)]
        + B_("r", **kw) + [("·x + ", kw)] + b_("r", **kw) + [(")", kw)],
        [("сколько прошлого учитывать", {"s": 14, "c": MUTED})],
        [("c = tanh(", kw)] + A_("c", **kw) + [("·(", kw)] + H("prev", **kw) + [(" ⊙ r) + ", kw)]
        + B_("c", **kw) + [("·x + ", kw)] + b_("c", **kw) + [(")", kw)],
        [("кандидат на новое состояние", {"s": 14, "c": MUTED})],
        H("next", **kw) + [(" = (1 − z) ⊙ ", kw)] + H("prev", **kw) + [(" + z ⊙ c", kw)],
        [("смесь старого и кандидата", {"s": 14, "c": MUTED})],
    ]
    tb(s, X0 + I(0.3), TOP + I(0.3), w - I(0.6), I(3.9), paras, gap=7)
    x = I(7.3)
    tb(s, x, TOP, I(5.1), I(0.4),
       [[("Три пути от ", {"b": True, "c": ACC, "s": 19})] + H("prev", b=True, c=ACC, s=19)
        + [(" к ", {"b": True, "c": ACC, "s": 19})] + H("next", b=True, c=ACC, s=19)], size=19)
    tb(s, x, TOP + I(0.6), I(5.1), I(3.2),
       [[("1. Напрямую: ", {"b": True}), ("(1 − z). Часть старого просто переходит дальше.", {})],
        [("2. Через гейт z: ", {"b": True}), ("(c − ", {})] + H("prev") + [(") · σ", {}), ("′", {"sup": True}), ("(z) · ", {})]
        + A_("z") + [(". Гейт сам зависит от ", {})] + H("prev") + [(".", {})],
        [("3. Через кандидата: ", {"b": True}), ("z · σ", {}), ("′", {"sup": True}), ("(c) · ", {})] + A_("c")
        + [(" · (r + ", {})] + H("prev") + [(" · σ", {}), ("′", {"sup": True}), ("(r) · ", {})]
        + A_("r") + [("). Здесь ещё и гейт r.", {})]],
       size=15, gap=8, line=1.25)
    box(s, MSO_SHAPE.RECTANGLE, x, TOP + I(3.35), I(5.1), I(0.02), GRID)
    tb(s, x, TOP + I(3.5), I(5.1), I(1.1),
       "Если A диагональные, все три слагаемых — векторы, и J тоже вектор. В коде берут −J: система записана как h − f(h).",
       size=14, color=MUTED, line=1.3)


def slide_lstm(d):
    s = d.new("lstm", "LSTM: состояние из двух частей")
    w = I(6.2)
    card(s, X0, TOP, w, I(4.55))
    kw = {"s": 16, "b": True}
    paras = [
        [("f = σ(", kw)] + A_("f", **kw) + [("·", kw)] + H("prev", **kw) + [(" + ", kw)]
        + B_("f", **kw) + [("·x + ", kw)] + C_("f", **kw) + [("·", kw)]
        + [("c", kw), ("prev", {**kw, "sub": True})] + [(" + ", kw)] + b_("f", **kw) + [(")", kw)],
        [("забывание; ворота «подглядывают» в ", {"s": 14, "c": MUTED})]
        + [("c", {"s": 14, "c": MUTED}), ("prev", {"s": 14, "c": MUTED, "sub": True})],
        [("c = f ⊙ ", kw)] + [("c", kw), ("prev", {**kw, "sub": True})] + [(" + (1 − f) ⊙ z", kw)],
        [("новое c; z — кандидат", {"s": 14, "c": MUTED})],
        [("o = σ(", kw)] + A_("o", **kw) + [("·", kw)] + H("prev", **kw) + [(" + ", kw)]
        + B_("o", **kw) + [("·x + ", kw)] + C_("o", **kw) + [("·c + ", kw)] + b_("o", **kw) + [(")", kw)],
        [("выходные ворота", {"s": 14, "c": MUTED})],
        H("next", **kw) + [(" = o ⊙ tanh(c)", kw)],
        [("выход ячейки", {"s": 14, "c": MUTED})],
    ]
    tb(s, X0 + I(0.3), TOP + I(0.3), w - I(0.6), I(3.9), paras, gap=7)
    x = I(7.3)
    tb(s, x, TOP, I(5.1), I(0.4), "Якобиан — блок 2 × 2", size=19, color=ACC, bold=True)
    sz = I(0.8)
    for i, j, name in ((0, 0, "cc"), (0, 1, "ch"), (1, 0, "hc"), (1, 1, "hh")):
        c = box(s, MSO_SHAPE.RECTANGLE, x + j * sz, TOP + I(0.6) + i * sz, sz, sz, ACC if i == j else LACC, BG, 1.5)
        label(c, J(name), 15, WHITE if i == j else INK)
    tb(s, x + I(1.85), TOP + I(0.6), I(3.3), I(1.6),
       "Состояние — пара (c, h). Каждый блок сам диагональный: на каждый канал приходится своя матрица 2 × 2.",
       size=14, color=MUTED, line=1.3)
    tb(s, x, TOP + I(2.5), I(5.1), I(2.2),
       [J("cc", b=True) + [(", ", {"b": True})] + J("ch", b=True)
        + [(": как прошлые c и h влияют на новое c.", {})],
        J("hc", b=True) + [(", ", {"b": True})] + J("hh", b=True)
        + [(": h берётся из нового c, поэтому эти блоки строятся через ", {})]
        + J("cc") + [(" и ", {})] + J("ch") + [(".", {})]],
       size=15, gap=8, line=1.3)


def slide_tradeoff(d):
    s = d.new("tradeoff", "Диагональ: что теряем и чем платим",
              "Идея статьи: не приближать якобиан, а подогнать сам модельный блок под дешёвый и точный якобиан.",
              cap_y=I(5.65))
    w = I(5.64)
    head_card(s, X0, TOP, w, I(2.0), "Что теряем",
              "Внутри ячейки каналы не смешиваются: каждое число состояния видит только своё прошлое. Как в Mamba.")
    head_card(s, I(7.0), TOP, w, I(2.0), "Как компенсируют",
              "Смешивание делают матрицы входа B (блочные, по головам), свёртки и MLP между слоями.")
    card(s, X0, TOP + I(2.3), I(11.84), I(1.55))
    tb(s, X0 + I(0.3), TOP + I(2.55), I(11.2), I(1.2),
       [[("Другой путь (Gonzalez и др., 2024): ", {"b": True, "c": ACC}),
         ("оставить полную ячейку и взять только диагональ её якобиана. Тогда Ньютон приближённый. "
          "ParaRNN меняет ячейку, и якобиан получается точным.", {})]], size=15, line=1.3)


def slide_impl(d):
    s = d.new("impl", "Как этим пользоваться: три реализации",
              "Пользователь пишет только шаг ячейки f(h, x). Якобианы и параллельный решатель фреймворк собирает сам.",
              cap_y=I(5.65))
    w, h, gap = I(3.7), I(3.5), I(0.37)
    items = [("parallel", "Только PyTorch. Для отладки и проб новых ячеек. Самый медленный из трёх."),
             ("parallel_CUDA", "Решатель редукции на CUDA, якобианы собирает autograd. Градиенты считаются автоматически."),
             ("parallel_FUSED", "Весь Ньютон в одном CUDA-ядре. Быстрее всех, но формулы якобианов нужно дать вручную.")]
    for k, (name, text) in enumerate(items):
        head_card(s, X0 + k * (w + gap), TOP, w, h, name, text, size=17)


def slide_limits(d):
    s = d.new("limits", "Что нужно для сходимости, и чего нет",
              "Для новой ячейки сходимость Ньютона нужно проверять отдельно: гарантии «за 3 шага» нет.",
              cap_y=I(5.65))
    w, h, gap = I(3.7), I(3.5), I(0.37)
    bodies = [
        [("h", {"s": 17}), ("0", {"sup": True, "s": 17}), ("t", {"sub": True, "s": 17}),
         (" = f(0, ", {"s": 17})] + X("t", s=17)
        + [("): ячейка с нулевым прошлым. Считается параллельно, одним проходом.", {"s": 17})],
        [("Теория: до L, и тогда выигрыша нет. На практике 3–4: на старте обучения и после него.", {"s": 17})],
        [("Каждая итерация собирает якобианы и делает редукцию. Это лишняя работа, но глубина растёт как log L.", {"s": 17})],
    ]
    for k, (name, body) in enumerate(zip(("Начальная догадка", "Сколько итераций", "Цена"), bodies)):
        card(s, X0 + k * (w + gap), TOP, w, h)
        tb(s, X0 + k * (w + gap) + I(0.3), TOP + I(0.28), w - I(0.6), I(0.4), name, size=19, color=ACC, bold=True)
        tb(s, X0 + k * (w + gap) + I(0.3), TOP + I(0.85), w - I(0.6), h - I(1.0), [body], size=17, gap=6)


def slide_arch(d):
    s = d.new("arch", "Как ячейку встраивают в языковую модель",
              "Берут Transformer DCLM, заменяют внимание на RNN-слой. Размеры: 400M, 1B, 2.9B и 7B параметров.",
              cap_y=I(5.15))
    y, h = TOP + I(0.5), I(1.1)
    steps = [("токены", CARD, INK), ("эмбеддинг", CARD, INK), ("свёртка", SOFT, INK), ("ячейка\nParaGRU / ParaLSTM", ACC, WHITE),
             ("гейт", SOFT, INK), ("MLP", CARD, INK), ("выход", CARD, INK)]
    widths = [I(1.2), I(1.4), I(1.3), I(2.5), I(1.1), I(1.1), I(1.1)]
    gap = (XR - X0 - sum(widths)) // (len(steps) - 1)
    x = X0
    for (t, f, c), w in zip(steps, widths):
        b = box(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, f)
        b.adjustments[0] = 0.12
        label(b, t, 14, c)
        if x > X0:
            arrow(s, x - gap + I(0.04), y + h // 2, x - I(0.04), y + h // 2, color=MUTED, lw=1.25)
        x += w + gap
    tb(s, X0, y + h + I(0.35), I(11.5), I(1.4),
       ["Блок «свёртка → ячейка → гейт» повторяется N раз, это упрощённая схема. Свёртку и гейт авторы взяли из Mamba.",
        "Обучали на одном корпусе, объём данных подбирали по правилу Chinchilla."],
       size=15, gap=8, line=1.3, color=INK)


def slide_paper_results(d):
    s = d.new("res", "Что показала статья",
              "Данные авторов. k-hop: нелинейные ячейки 100%, Mamba2 98%, Transformer 78%. В MQAR все 100%.",
              cap_y=I(5.2))
    w, h, gap = I(3.7), I(3.0), I(0.37)
    cards = [("до 665×", "быстрее цикла", "Полное параллельное применение ячейки, длина 512. Для двух ячеек: 447× и 665×."),
             ("35–37", "токенов в секунду", "Генерация у ParaGRU и ParaLSTM, у Mamba около 27. Не замедляется с длиной."),
             ("100%", "на чётности", "Нелинейные ячейки решают задачу. Mamba2 около 51%, Transformer 53%.")]
    for k, (big, name, small) in enumerate(cards):
        big_card(s, X0 + k * (w + gap), TOP, w, h, big, name, small, big_size=44)


def slide_divider(d):
    s = d.new("divider", "Приложение: наше воспроизведение",
              "Методика и выводы — в отчёте. Здесь только графики. Запуск: RTX 3060 Laptop, 6 ГБ.", cap_y=TOP)
    tb(s, X0, TOP + I(1.0), I(11.5), I(3.0),
       ["Ньютон сходится за 3 итерации до точности float32.",
        "Обучение в режимах цикла и параллельном совпадает.",
        "Ускорение до 760× на длине 4096 (ParaGRU, fused).",
        "Чётность: линейная RNN 0,50, ParaGRU и ParaLSTM 1,00."],
       size=20, gap=14, line=1.3)
    return s


# ---------- plot slides (appendix) ----------
PLOTS = [
    ("newton", "Ньютон сходится за три шага",
     "Ошибка падает в 10 раз за шаг и после третьей итерации упирается в точность float32. Длины 128, 512 и 2048 ведут себя одинаково."),
    ("modes", "Обучение не отличается от цикла", None),  # caption set below with real superscript
    ("speed", "Чем длиннее строка, тем больше выигрыш",
     "Время прямого и обратного прохода. Цикл растёт почти линейно, параллельные режимы почти плоские. Логарифмическая шкала."),
    ("reduction", "Где именно мы выигрываем", None),
    ("parity_train", "Обучение: линейная RNN стоит на месте",
     "Линейная RNN остаётся на 0,50, как случайное угадывание. ParaGRU и ParaLSTM срываются вверх за 225–250 шагов."),
    ("parity_trace", "Одна строка глазами моделей",
     "Серая линия — правильный ответ, точки — предсказания. Оранжевые точки — ошибки: линейная ошибается в 27 из 50 позиций."),
    ("parity_position", "Точность по позиции в строке",
     "Нелинейные ячейки верны на любой позиции, кроме первых двух. Линейная RNN держится около 0,5 везде."),
]


def plot_slide(d, name, title, caption):
    s = d.new(name, title, caption)
    if name == "modes":
        cap = clone(d.cap_src, s)
        set_runs(cap, [("Сорок шагов обучения ParaGRU в режимах sequential и parallel_CUDA. "
                        "Кривые совпадают, максимальная разница 1,2·10", {}),
                       ("-7", {"sup": True}), (".", {})], size=18)
        cap.top = I(5.9)
        cap.height = I(0.9)
    img = Image.open(FIG / f"{name}.png")
    ratio = img.width / img.height
    if name == "reduction":
        h = 4500000
        w = int(h * ratio)
        s.shapes.add_picture(str(FIG / f"{name}.png"), X0, TOP, w, h)
        left = X0 + w + 400000
        width = XR - left
        card_shape = clone(d.stat.shapes[5], s)
        card_shape.left, card_shape.width, card_shape.top, card_shape.height = left, width, TOP + 700000, 2308860
        for src, txt in ((6, "28×"), (7, "ядро редукции"),
                         (8, "CUDA против PyTorch, длина 4096. Только решение линейной рекуррентности.")):
            t = clone(d.stat.shapes[src], s)
            set_text(t, txt)
            t.left, t.width = left + 279400, width - 558800
            t.top = t.top - d.stat.shapes[5].top + card_shape.top
        return s
    w = min(XR - X0, int(3900000 * ratio))
    h = int(w / ratio)
    s.shapes.add_picture(str(FIG / f"{name}.png"), X0, TOP, w, h)
    cap = s.shapes[-2]
    cap.top = TOP + h + 300000
    return s


def main():
    d = Deck()
    for fn in (slide_system, slide_jacobian, slide_cost, slide_reduction, slide_backward, slide_gru, slide_lstm,
               slide_tradeoff, slide_impl, slide_limits, slide_arch, slide_paper_results, slide_divider):
        fn(d)
    for name, title, cap in PLOTS:
        plot_slide(d, name, title, cap)

    o, n = d.orig, d.by_key
    # original-slide formula fixes (sub/sup + prev/next)
    # slide 3 (index 2): Mamba formula in text shape 6
    set_runs(o[2].shapes[6],
             [("Вернула рекуррентность, но линейную:\n", {})]
             + H("t") + [(" = a·", {})] + H("prev") + [(" + b.\n", {}),
              ("Такие шаги можно склеивать парами.", {})], size=15)
    # slide 8 (index 7): δ formula
    set_text(o[6].shapes[1], "Вблизи черновика нелинейную ячейку заменяют касательной. Тогда поправка шага t зависит только от поправки шага prev:")
    set_runs(o[6].shapes[2], D("t") + [(" = ", {})] + A_("t") + [(" · ", {})] + D("prev") + [(" + ", {})] + R("t"),
             size=31, color=ACC, bold=True, align="c")
    set_runs(o[6].shapes[4], A_("t"), size=19, color=ACC, bold=True)
    set_runs(o[6].shapes[7], R("t"), size=19, color=ACC, bold=True)
    # 7B + summary
    set_text(o[11].shapes[0], "Результат на моделях 7B")
    set_text(o[11].shapes[1], "Числа из таблицы авторов: ячейку ставят вместо внимания и учат на SlimPajama. Это недели на нескольких H100.")
    set_runs(o[12].shapes[2],
             [("Ответ совпадает с циклом до 10", {})] + [("-7", {"sup": True}),
              (" (наш запуск). Выигрыш в сотни раз: у авторов до 665×, у нас ≈760× на длине 4096.", {})], size=18)
    # check cards with 10^{-7}
    set_runs(o[8].shapes[6], [("~10", {}), ("-7", {"sup": True})], size=48, color=ACC, bold=True)
    order = [o[0], o[1], o[2], o[3], o[4], o[5], n["system"], o[6], n["jac"], n["reduce"], n["cost"], n["bwd"],
             o[7], n["gru"], n["lstm"], n["tradeoff"], n["impl"], n["limits"], n["arch"], n["res"], o[11], o[12],
             n["divider"], o[8], n["newton"], n["modes"], o[9], n["speed"], n["reduction"], o[10],
             n["parity_train"], n["parity_trace"], n["parity_position"]]
    ids = d.prs.slides._sldIdLst
    by_id = {el.get("id"): el for el in ids}
    els = [by_id[str(s.slide_id)] for s in order]
    for el in list(ids):
        ids.remove(el)
    for el in els:
        ids.append(el)
    d.prs.save(DST)
    print("saved", DST, len(els), "slides")


if __name__ == "__main__":
    main()
