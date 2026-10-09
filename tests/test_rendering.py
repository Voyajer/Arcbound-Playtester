"""Headless rendering checks for the Tkinter GUI.

Validates that canvas-based widgets (EvalGraph, EvaluationBar) actually draw
items after layout, and that the app builds without errors.
"""

import sys
from pathlib import Path

import pytest

tk = pytest.importorskip("tkinter")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcbound.gui.components.eval_graph import EvalGraph  # noqa: E402
from arcbound.gui.components.eval_bar import EvaluationBar  # noqa: E402


def _pump(root, iterations=20):
    """Force geometry management + event processing so winfo_* is valid."""
    root.update_idletasks()
    root.update()
    for _ in range(iterations):
        root.update()


def test_eval_graph_draws_after_layout():
    root = tk.Tk()
    root.geometry("600x300")
    graph = EvalGraph(root, width=500, height=150)
    graph.pack(fill=tk.BOTH, expand=True)
    _pump(root)

    # After layout the canvas should have a real size.
    w = graph.canvas.winfo_width()
    h = graph.canvas.winfo_height()
    assert w > 1 and h > 1, f"canvas not laid out: {w}x{h}"

    # Add points and force a redraw.
    graph.add_point(1, 0.2)
    graph.add_point(2, 0.5)
    graph.add_point(3, -0.3)
    _pump(root)

    items = graph.canvas.find_all()
    # Grid lines (5) + zero line + data line + 3 dots + 3 x-labels = at least 12
    assert len(items) >= 10, f"expected drawn items, got {len(items)}"

    # The chart must fill the allocated canvas width (not just the requested size).
    max_x = max(
        max(graph.canvas.coords(it)[0::2])
        for it in items
        if graph.canvas.type(it) in ("line", "rectangle", "polygon")
    )
    assert max_x > w * 0.9, f"chart does not fill canvas: max_x={max_x:.0f}, w={w}"
    root.destroy()


def test_eval_graph_redraws_on_resize():
    """The graph must re-render when the window is resized (winfo_reqwidth bug)."""
    root = tk.Tk()
    root.geometry("400x200")
    graph = EvalGraph(root, width=300, height=120)
    graph.pack(fill=tk.BOTH, expand=True)
    graph.add_point(1, 0.1)
    graph.add_point(2, 0.4)
    _pump(root)

    # Resize the window larger; the canvas should grow and the graph re-draw.
    root.geometry("800x400")
    _pump(root)

    w = graph.canvas.winfo_width()
    assert w > 300, f"canvas did not grow on resize: {w}"
    # There should still be drawn content (grid + line).
    assert len(graph.canvas.find_all()) >= 10
    root.destroy()


def test_eval_bar_draws_after_layout():
    root = tk.Tk()
    root.geometry("200x300")
    bar = EvaluationBar(root, player_name="AI", opponent_name="Opp", width=30, height=200)
    bar.pack(fill=tk.BOTH, expand=True)
    _pump(root)

    w = bar.canvas.winfo_width()
    h = bar.canvas.winfo_height()
    assert w > 1 and h > 1, f"canvas not laid out: {w}x{h}"

    bar.set_eval(0.7)
    _pump(root)

    items = bar.canvas.find_all()
    # Background + fill + 5 scale lines + 5 scale labels + 2 names = ~14
    assert len(items) >= 10, f"expected drawn items, got {len(items)}"

    # Scale labels must be inside the canvas (not clipped on the right edge).
    for it in items:
        if bar.canvas.type(it) == "text":
            x, _y = bar.canvas.coords(it)
            assert x < w, f"text clipped outside canvas: x={x:.0f}, w={w}"
    root.destroy()


def test_app_builds_without_error():
    """The full app should construct and lay out without exceptions."""
    from arcbound.gui.app import ArcboundApp

    app = ArcboundApp()
    _pump(app)
    # Status bar labels should have non-empty text.
    assert app.lbl_server.cget("text")
    assert app.lbl_port.cget("text")
    # Notebook should have 5 tabs.
    assert len(app.notebook.tabs()) == 5
    app.destroy()


def _find_status_frame(app):
    """Return the top-level frame that contains the 'Server:' label."""
    for child in app.winfo_children():
        for desc in child.winfo_children():
            try:
                if desc.winfo_class() == "TLabel" and str(desc.cget("text")).startswith("Server:"):
                    return child
            except tk.TclError:
                pass
    return None


def test_status_bar_visible_at_multiple_sizes():
    """The bottom status bar must not collapse to 1px at any window size.

    Regression test: the status bar was packed with the default side=TOP
    *after* the expanding main frame, so the main frame consumed all
    vertical space and the status bar was squeezed to 1px (invisible).
    """
    from arcbound.gui.app import ArcboundApp

    for size in ["1250x800", "1250x700", "1000x600", "900x550"]:
        app = ArcboundApp()
        app.geometry(size)
        _pump(app)

        status = _find_status_frame(app)
        assert status is not None, f"status bar not found at {size}"

        h = status.winfo_height()
        wh = app.winfo_height()
        bottom = status.winfo_y() + h
        # The status bar must have real height (not collapsed to 1px) and
        # sit at the bottom of the window.
        assert h >= 30, f"status bar collapsed to {h}px at {size}"
        assert bottom >= wh - 2, f"status bar not at bottom: bottom={bottom}, window={wh} at {size}"
        app.destroy()
