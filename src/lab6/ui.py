"""Графический интерфейс ЛР №6 на PySide6."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from .geometry import SOLIDS, Polyhedron, save_obj
from .renderer import Viewport
from .transforms import (
    around_point, reflection, rotate_around_line, rotation_axis, scaling, translation,
)


STYLE = """
QMainWindow, QWidget { background: #0c1525; color: #e8f0ff; font-family: 'Segoe UI', Arial; font-size: 12px; }
QGroupBox { border: 1px solid #263b57; border-radius: 10px; margin-top: 15px; padding: 13px 10px 10px; font-weight: 600; background: #111e31; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #c7d9fa; }
QDoubleSpinBox, QComboBox { border: 1px solid #324a6b; border-radius: 6px; background: #172941; padding: 5px; min-height: 24px; selection-background-color: #317be2; }
QDoubleSpinBox:focus, QComboBox:focus { border-color: #6aa9ff; }
QComboBox QAbstractItemView { background: #16263c; selection-background-color: #355f94; }
QPushButton { background: #235baf; border: 1px solid #3e7acb; border-radius: 7px; color: white; min-height: 29px; padding: 4px 10px; font-weight: 600; }
QPushButton:hover { background: #3374ce; }
QPushButton:pressed { background: #1c4585; }
QPushButton[secondary="true"] { background: #203047; border: 1px solid #38516f; }
QPushButton[secondary="true"]:hover { background: #2a4161; }
QCheckBox { spacing: 7px; }
QScrollArea { border: none; background: transparent; }
QLabel[muted="true"] { color: #8da3c3; }
QMenuBar { background: #101c30; color: #e8f0ff; }
QMenuBar::item:selected { background: #264976; }
QMenu { background: #142237; color: #e8f0ff; }
QMenu::item:selected { background: #315985; }
QStatusBar { color: #97acc9; background: #101c30; }
"""


def spin(value: float = 0.0, low: float = -1000.0, high: float = 1000.0,
         step: float = 0.25) -> QDoubleSpinBox:
    field = QDoubleSpinBox()
    field.setRange(low, high)
    field.setDecimals(3)
    field.setSingleStep(step)
    field.setValue(value)
    field.setKeyboardTracking(False)
    return field


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Лабораторная №6 — Аффинные преобразования 3D")
        self.resize(1360, 840)
        self.setMinimumSize(1040, 660)
        self.setStyleSheet(STYLE)
        self.model: Polyhedron = SOLIDS["Гексаэдр (куб)"]()
        self.history: list[np.ndarray] = []

        container = QWidget()
        self.setCentralWidget(container)
        outer = QHBoxLayout(container)
        outer.setContentsMargins(12, 12, 12, 10)
        outer.setSpacing(12)

        self.viewport = Viewport()
        self.viewport.set_polyhedron(self.model)
        outer.addWidget(self.viewport, stretch=3)

        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setMinimumWidth(350)
        scroller.setMaximumWidth(440)
        controls = QWidget()
        self.panel = QVBoxLayout(controls)
        self.panel.setContentsMargins(8, 4, 8, 8)
        self.panel.setSpacing(10)
        scroller.setWidget(controls)
        outer.addWidget(scroller, stretch=1)

        self._build_panel()
        self._build_menu()
        self._refresh()
        self.statusBar().showMessage("Все преобразования выполняются матрицами 4×4 · Ctrl+Z — отмена")

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("Файл")
        for name, callback, shortcut in [
            ("Экспорт модели в OBJ...", self._export_obj, "Ctrl+E"),
            ("Сохранить изображение PNG...", self._save_picture, "Ctrl+S"),
            ("Выход", self.close, "Ctrl+Q"),
        ]:
            action = QAction(name, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            menu.addAction(action)
        edit = self.menuBar().addMenu("Правка")
        undo = QAction("Отменить преобразование", self)
        undo.setShortcut(QKeySequence("Ctrl+Z"))
        undo.triggered.connect(self._undo)
        edit.addAction(undo)
        help_menu = self.menuBar().addMenu("Справка")
        help_action = QAction("Как пользоваться", self)
        help_action.triggered.connect(self._help)
        help_menu.addAction(help_action)

    def _section(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setSpacing(7)
        self.panel.addWidget(box)
        return box, layout

    @staticmethod
    def _row(label: str, values: list[QWidget]) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        if label:
            text = QLabel(label)
            text.setMinimumWidth(36)
            layout.addWidget(text)
        for element in values:
            layout.addWidget(element, 1)
        return widget

    @staticmethod
    def _button(text: str, callback, secondary=False) -> QPushButton:
        button = QPushButton(text)
        if secondary:
            button.setProperty("secondary", True)
        button.clicked.connect(callback)
        return button

    def _build_panel(self) -> None:
        title = QLabel("3D · Лабораторная №6")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.panel.addWidget(title)
        subtitle = QLabel("Многогранники · Матрицы · Проекции")
        subtitle.setProperty("muted", True)
        self.panel.addWidget(subtitle)

        _, layout = self._section("01 · Многогранник и проекция")
        self.solid_combo = QComboBox()
        self.solid_combo.addItems(SOLIDS.keys())
        self.solid_combo.setCurrentText("Гексаэдр (куб)")
        self.solid_combo.currentTextChanged.connect(self._switch_solid)
        layout.addWidget(self.solid_combo)
        self.projection_combo = QComboBox()
        self.projection_combo.addItem("Перспективная", "perspective")
        self.projection_combo.addItem("Аксонометрическая (параллельная)", "orthographic")
        self.projection_combo.currentIndexChanged.connect(self._switch_projection)
        layout.addWidget(self.projection_combo)
        checks = self._row("", [QCheckBox("Грани"), QCheckBox("Рёбра"), QCheckBox("Оси")])
        layout.addWidget(checks)
        self.faces_check, self.edges_check, self.axes_check = checks.findChildren(QCheckBox)
        for checkbox, name in [(self.faces_check, "show_faces"),
                               (self.edges_check, "show_edges"),
                               (self.axes_check, "show_axes")]:
            checkbox.setChecked(True)
            checkbox.toggled.connect(partial(self._toggle_view, name))

        _, layout = self._section("02 · Перенос в пространстве")
        self.move_fields = [spin(0) for _ in range(3)]
        layout.addWidget(self._row("Δ XYZ", self.move_fields))
        layout.addWidget(self._button("Переместить", self._move))

        _, layout = self._section("03 · Вращение")
        self.rotation_axis_combo = QComboBox()
        self.rotation_axis_combo.addItems(["X", "Y", "Z"])
        self.rotation_degrees = spin(30, -3600, 3600, 5)
        layout.addWidget(self._row("Ось / °", [self.rotation_axis_combo, self.rotation_degrees]))
        layout.addWidget(self._button("Вокруг оси координат", self._rotate_origin))
        layout.addWidget(self._button("Вокруг оси через центр фигуры", self._rotate_center, secondary=True))

        _, layout = self._section("04 · Масштабирование")
        self.scale_fields = [spin(1, -10, 10, 0.1) for _ in range(3)]
        layout.addWidget(self._row("XYZ", self.scale_fields))
        layout.addWidget(self._button("Относительно начала координат", self._scale_origin))
        layout.addWidget(self._button("Относительно центра фигуры", self._scale_center, secondary=True))
        layout.addWidget(QLabel("0 по оси сводит модель к плоскости — для отмены Ctrl+Z."))

        _, layout = self._section("05 · Отражение")
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(["XY", "XZ", "YZ"])
        layout.addWidget(self._row("Плоск.", [self.plane_combo]))
        layout.addWidget(self._button("Отразить относительно плоскости", self._reflect))

        _, layout = self._section("06 · Поворот вокруг произвольной прямой")
        layout.addWidget(QLabel("Прямая задаётся точками A и B"))
        self.line_a = [spin(x) for x in (0., 0., 0.)]
        self.line_b = [spin(x) for x in (1., 1., 1.)]
        self.line_angle = spin(45, -3600, 3600, 5)
        layout.addWidget(self._row("A", self.line_a))
        layout.addWidget(self._row("B", self.line_b))
        layout.addWidget(self._row("Угол °", [self.line_angle]))
        layout.addWidget(self._button("Повернуть вокруг AB", self._rotate_line))

        _, layout = self._section("07 · Состояние и управление")
        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        self.matrix_label = QLabel()
        self.matrix_label.setFont(QFont("Consolas", 9))
        self.matrix_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.matrix_label)
        actions = self._row("", [
            self._button("↶ Отменить", self._undo, secondary=True),
            self._button("Сбросить объект", self._reset_model, secondary=True),
        ])
        layout.addWidget(actions)
        layout.addWidget(self._button("Сбросить камеру", self.viewport.reset_camera, secondary=True))
        self.panel.addStretch()

    def _refresh(self) -> None:
        self.viewport.update()
        center = self.model.center
        self.info.setText(
            f"Вершин: {len(self.model.vertices)} · Рёбер: {len(self.model.edges)} · "
            f"Граней: {len(self.model.faces)}\n"
            f"Центр: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})\n"
            "Модельная матрица 4×4:")
        self.matrix_label.setText("\n".join(
            "  ".join(f"{number:7.3f}" for number in row) for row in self.model.model_matrix))

    def _apply(self, matrix: np.ndarray, description: str) -> None:
        try:
            self.history.append(self.model.model_matrix.copy())
            self.model.apply(matrix)
            self._refresh()
            self.statusBar().showMessage(description, 6500)
        except (ValueError, FloatingPointError) as exc:
            if self.history:
                self.history.pop()
            QMessageBox.warning(self, "Невозможно выполнить преобразование", str(exc))

    def _values(self, fields) -> list[float]:
        return [field.value() for field in fields]

    def _move(self) -> None:
        values = self._values(self.move_fields)
        self._apply(translation(*values), f"Перенос: {values}")

    def _rotate_origin(self) -> None:
        axis, angle = self.rotation_axis_combo.currentText(), self.rotation_degrees.value()
        self._apply(rotation_axis(axis, angle), f"Поворот {axis} вокруг начала координат на {angle}°")

    def _rotate_center(self) -> None:
        axis, angle = self.rotation_axis_combo.currentText(), self.rotation_degrees.value()
        self._apply(around_point(rotation_axis(axis, angle), self.model.center),
                    f"Поворот вокруг оси {axis} через центр на {angle}°")

    def _scale_origin(self) -> None:
        values = self._values(self.scale_fields)
        self._apply(scaling(*values), f"Масштабирование относительно O: {values}")

    def _scale_center(self) -> None:
        values = self._values(self.scale_fields)
        self._apply(around_point(scaling(*values), self.model.center),
                    f"Масштабирование относительно центра: {values}")

    def _reflect(self) -> None:
        plane = self.plane_combo.currentText()
        self._apply(reflection(plane), f"Отражение относительно {plane}")

    def _rotate_line(self) -> None:
        try:
            matrix = rotate_around_line(self._values(self.line_a), self._values(self.line_b), self.line_angle.value())
        except ValueError as exc:
            QMessageBox.warning(self, "Ошибка", str(exc))
            return
        self._apply(matrix, f"Поворот вокруг AB на {self.line_angle.value()}°")

    def _undo(self) -> None:
        if self.history:
            self.model.model_matrix = self.history.pop()
            self._refresh()
            self.statusBar().showMessage("Последнее преобразование отменено", 4000)
        else:
            self.statusBar().showMessage("История пуста", 3000)

    def _reset_model(self) -> None:
        self.history.append(self.model.model_matrix.copy())
        self.model.reset()
        self._refresh()

    def _switch_solid(self, name: str) -> None:
        self.model = SOLIDS[name]()
        self.history.clear()
        self.viewport.set_polyhedron(self.model)
        self._refresh()

    def _switch_projection(self) -> None:
        self.viewport.mode = self.projection_combo.currentData()
        self.viewport.update()

    def _toggle_view(self, name: str, checked: bool) -> None:
        setattr(self.viewport, name, checked)
        self.viewport.update()

    def _export_obj(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт модели в OBJ", "model.obj", "Wavefront OBJ (*.obj)")
        if path:
            try:
                save_obj(self.model, Path(path))
                self.statusBar().showMessage(f"Сохранено: {path}", 6000)
            except OSError as exc:
                QMessageBox.critical(self, "Ошибка сохранения", str(exc))

    def _save_picture(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить снимок", "polyhedron.png", "PNG (*.png)")
        if path and not self.viewport.grab().save(path, "PNG"):
            QMessageBox.critical(self, "Ошибка", "Не удалось сохранить изображение")

    def _help(self) -> None:
        QMessageBox.information(self, "Управление", """
        1. Выберите многогранник и проекцию.
        2. Задайте значения и нажмите кнопку соответствующего преобразования.
        3. Левой кнопкой мыши вращайте камеру, колесом меняйте масштаб вида.
        4. Все изменения объекта выполняются матрицами 4×4.
        5. Ctrl+Z — отмена, Ctrl+E — экспорт OBJ, Ctrl+S — PNG.
        6. ★ обозначает обязательные для нашей команды бонусные фигуры.
        """)
