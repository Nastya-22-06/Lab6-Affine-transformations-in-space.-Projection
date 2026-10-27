"""Qt-виджет: собственное матричное проецирование и отрисовка полигонов."""
from __future__ import annotations

import math
import numpy as np

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QBrush, QPolygonF
from PySide6.QtWidgets import QWidget

from .geometry import Polyhedron
from .projection import look_at, orthographic, perspective, project


class Viewport(QWidget):
    camera_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.polyhedron: Polyhedron | None = None
        self.mode = "perspective"
        self.show_faces = True
        self.show_edges = True
        self.show_axes = True
        self.yaw = 45.0
        self.pitch = 35.2643897
        self.distance = 7.5
        self.last_mouse_pos = None
        self.setMinimumSize(490, 440)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def set_polyhedron(self, model: Polyhedron) -> None:
        self.polyhedron = model
        self.update()

    def reset_camera(self) -> None:
        self.yaw, self.pitch, self.distance = 45.0, 35.2643897, 7.5
        self.update()
        self.camera_changed.emit()

    def _camera(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, b = math.radians(self.yaw), math.radians(self.pitch)
        eye = self.distance * np.array([math.sin(a) * math.cos(b), math.sin(b), math.cos(a) * math.cos(b)])
        view = look_at(eye, np.zeros(3))
        aspect = max(1., self.width()) / max(1., self.height())
        projection = (perspective(46., aspect) if self.mode == "perspective"
                      else orthographic(self.distance * 0.43, aspect))
        return view, projection, eye

    def _line3d(self, painter: QPainter, a: np.ndarray, b: np.ndarray,
                view: np.ndarray, projection: np.ndarray, color: str,
                width: float = 1.4) -> tuple[QPointF, QPointF] | None:
        pts, valid = project(np.array([a, b], float), view, projection, self.width(), self.height())
        if not np.all(valid):
            return None
        first, last = QPointF(*pts[0]), QPointF(*pts[1])
        pen = QPen(QColor(color))
        pen.setWidthF(width)
        painter.setPen(pen)
        painter.drawLine(first, last)
        return first, last

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        background = QLinearGradient(0, 0, 0, self.height())
        background.setColorAt(0, QColor("#101b30"))
        background.setColorAt(1, QColor("#080f1d"))
        painter.fillRect(self.rect(), QBrush(background))

        view, projection, eye = self._camera()
        if self.show_axes:
            self._draw_grid(painter, view, projection)
            self._draw_axes(painter, view, projection)

        if self.polyhedron is not None:
            points = self.polyhedron.world_vertices
            pixels, valid = project(points, view, projection, self.width(), self.height())
            camera_points = (view @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
            face_records = []
            for index, face in enumerate(self.polyhedron.faces):
                ids = list(face.indices)
                if not np.all(valid[ids]):
                    continue
                polygon = QPolygonF([QPointF(float(pixels[i, 0]), float(pixels[i, 1])) for i in ids])
                normal = face.normal(points)
                # В ЛР №6 это только декоративное различие цвета граней, не алгоритм ЛР №9.
                toward_light = np.array([-0.32, 0.86, 0.41])
                brightness = 0.48 + 0.52 * abs(float(np.dot(normal, toward_light) / np.linalg.norm(toward_light)))
                depth = float(np.mean(camera_points[ids, 2]))
                face_records.append((depth, index, polygon, brightness))

            # Для выпуклых Платоновых тел painter's algorithm достаточно для ЛР №6.
            for _, index, polygon, brightness in sorted(face_records, key=lambda f: f[0]):
                hue = (197 + index * 17) % 360
                fill = QColor.fromHsv(hue, 115, int(205 * brightness + 36), 234)
                painter.setBrush(QBrush(fill) if self.show_faces else Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor("#c5d9fb")) if self.show_edges
                               else Qt.PenStyle.NoPen)
                painter.drawPolygon(polygon)

            if self.show_edges and not self.show_faces:
                # Каркас отображает также невидимые рёбра — удобно для защиты.
                for a, b in self.polyhedron.edges:
                    if valid[a] and valid[b]:
                        painter.setPen(QPen(QColor("#a5ddff")))
                        painter.drawLine(QPointF(*pixels[a]), QPointF(*pixels[b]))

        painter.setPen(QColor("#aabbd3"))
        font = QFont("Arial", 10)
        painter.setFont(font)
        title = "ПЕРСПЕКТИВНАЯ" if self.mode == "perspective" else "АКСОНОМЕТРИЧЕСКАЯ"
        painter.drawText(20, 30, f"{title} ПРОЕКЦИЯ")
        painter.setPen(QColor("#778ca9"))
        painter.drawText(20, self.height() - 18, "Мышь: вращение камеры  ·  Колесо: приближение")
        painter.end()

    def _draw_grid(self, painter, view, projection) -> None:
        # Сетка лежит на плоскости XZ.
        for value in np.arange(-4.0, 4.01, 0.5):
            base_color = "#233551" if abs(value) > 0.01 else "#34506e"
            self._line3d(painter, np.array([value, -1.98, -4]), np.array([value, -1.98, 4]), view, projection, base_color, 0.7)
            self._line3d(painter, np.array([-4, -1.98, value]), np.array([4, -1.98, value]), view, projection, base_color, 0.7)

    def _draw_axes(self, painter, view, projection) -> None:
        for tip, color, name in [([2.7, 0, 0], "#f18c9d", "X"),
                                 ([0, 2.7, 0], "#5ee3ae", "Y"),
                                 ([0, 0, 2.7], "#78b7ff", "Z")]:
            result = self._line3d(painter, np.zeros(3), np.array(tip), view, projection, color, 2.2)
            if result:
                painter.setPen(QColor(color))
                painter.setFont(QFont("Arial", 11, QFont.Weight.Bold))
                painter.drawText(result[1] + QPointF(5, -5), name)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.last_mouse_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self.last_mouse_pos
            self.yaw += delta.x() * 0.45
            self.pitch = max(-86., min(86., self.pitch + delta.y() * 0.45))
            self.last_mouse_pos = event.position()
            self.update()
            self.camera_changed.emit()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self.last_mouse_pos = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 0.86 if event.angleDelta().y() > 0 else 1.16
        self.distance = max(3.5, min(25., self.distance * factor))
        self.update()
        self.camera_changed.emit()
