import sys
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QButtonGroup, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
from scipy.interpolate import CubicSpline

class AdvancedShapeDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Shape Editor (-1 to 1)")
        self.setGeometry(100, 100, 950, 750)

        # 데이터 구조
        self.points = []       # [(x, y), ...] 생성된 점들
        self.lines = []        # [((x1, y1), (x2, y2)), ...]
        self.curves = []       # [ [(x1, y1), (x2, y2), ...], ... ]

        # 마우스 상호작용 관련 변수
        self.is_dragging = False
        self.start_pt = None
        self.end_pt = None
        
        # 포인트 편집(이동) 관련 변수
        self.selected_point_idx = None # 이동할 점의 인덱스

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 1. 왼쪽: Matplotlib 캔버스
        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.setup_axes()
        main_layout.addWidget(self.canvas, stretch=4)

        # 마우스 이벤트 연결
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        # 2. 오른쪽: 컨트롤 패널
        control_layout = QVBoxLayout()
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.btn_group = QButtonGroup(self)
        
        self.btn_point = QPushButton("포인트 추가 모드 (클릭)")
        self.btn_point.setCheckable(True)
        self.btn_point.setChecked(True)
        self.btn_group.addButton(self.btn_point, 0)
        control_layout.addWidget(self.btn_point)

        self.btn_edit = QPushButton("포인트 편집 모드 (드래그 이동)")
        self.btn_edit.setCheckable(True)
        self.btn_group.addButton(self.btn_edit, 3)
        control_layout.addWidget(self.btn_edit)

        self.btn_line = QPushButton("직선 모드 (드래그 박스)")
        self.btn_line.setCheckable(True)
        self.btn_group.addButton(self.btn_line, 1)
        control_layout.addWidget(self.btn_line)

        self.btn_curve = QPushButton("곡선 모드 (드래그 박스)")
        self.btn_curve.setCheckable(True)
        self.btn_group.addButton(self.btn_curve, 2)
        control_layout.addWidget(self.btn_curve)

        control_layout.addSpacing(15)
        
        # 기능성 버튼들
        self.btn_mirror = QPushButton("Y축 기준 대칭 복사 (Mirror)")
        self.btn_mirror.setStyleSheet("background-color: #e1f5fe; font-weight: bold;")
        self.btn_mirror.clicked.connect(self.mirror_along_y)
        control_layout.addWidget(self.btn_mirror)

        # 🔄 리셋 버튼 추가
        self.btn_reset = QPushButton("전체 초기화 (Reset)")
        self.btn_reset.setStyleSheet("background-color: #ffebee; color: #c62828; font-weight: bold;")
        self.btn_reset.clicked.connect(self.reset_data)
        control_layout.addWidget(self.btn_reset)

        control_layout.addSpacing(25)

        self.btn_save = QPushButton("저장 (Save)")
        self.btn_save.clicked.connect(self.save_data)
        control_layout.addWidget(self.btn_save)

        self.btn_load = QPushButton("불러오기 (Load)")
        self.btn_load.clicked.connect(self.load_data)
        control_layout.addWidget(self.btn_load)

        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        main_layout.addWidget(control_widget, stretch=1)

    def setup_axes(self):
        self.ax.clear()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, which='both', linestyle='--', color='lightgray', alpha=0.7)
        self.ax.axhline(0, color='gray', linewidth=1.2)
        self.ax.axvline(0, color='gray', linewidth=1.2)

    def redraw(self):
        self.setup_axes()

        # 1. 포인트 그리기
        if self.points:
            px, py = zip(*self.points)
            self.ax.scatter(px, py, color='blue', s=50, zorder=5)

        # 2. 직선 그리기
        for line in self.lines:
            (x1, y1), (x2, y2) = line
            self.ax.plot([x1, x2], [y1, y2], color='green', linewidth=2, zorder=3)

        # 3. 곡선 그리기
        for curve in self.curves:
            self.draw_cubic_spline(curve)

        # 4. 드래그 박스 시각화
        mode = self.btn_group.checkedId()
        if self.is_dragging and self.start_pt and self.end_pt and mode in (1, 2):
            x1, y1 = self.start_pt
            x2, y2 = self.end_pt
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, 
                             edgecolor='crimson', facecolor='crimson', alpha=0.15, linestyle='--')
            self.ax.add_patch(rect)

        self.canvas.draw()

    def draw_cubic_spline(self, points):
        if len(points) < 2: return
        pts = np.array(points)
        
        distances = np.sqrt(np.diff(pts[:, 0])**2 + np.diff(pts[:, 1])**2)
        keep = np.where(distances > 1e-4)[0]
        keep = np.append(keep, len(pts) - 1)
        pts = pts[keep]

        if len(pts) == 2:
            self.ax.plot(pts[:, 0], pts[:, 1], color='orange', linewidth=2, zorder=4)
        elif len(pts) >= 3:
            try:
                t = np.zeros(len(pts))
                t[1:] = np.cumsum(np.sqrt(np.diff(pts[:, 0])**2 + np.diff(pts[:, 1])**2))
                
                cs_x = CubicSpline(t, pts[:, 0], bc_type='natural')
                cs_y = CubicSpline(t, pts[:, 1], bc_type='natural')
                
                t_new = np.linspace(t[0], t[-1], 150)
                x_new = cs_x(t_new)
                y_new = cs_y(t_new)
                
                self.ax.plot(x_new, y_new, color='orange', linewidth=2, zorder=4)
            except Exception:
                self.ax.plot(pts[:, 0], pts[:, 1], color='orange', linestyle='--', linewidth=1.5, zorder=4)

    def find_closest_point(self, target_pt, threshold=0.05):
        if not self.points: return None
        pts = np.array(self.points)
        dists = np.hypot(pts[:, 0] - target_pt[0], pts[:, 1] - target_pt[1])
        min_idx = np.argmin(dists)
        if dists[min_idx] < threshold:
            return min_idx
        return None

    # --- 마우스 이벤트 핸들러 ---
    def on_press(self, event):
        if event.inaxes != self.ax: return
        
        mode = self.btn_group.checkedId()
        click_pt = (event.xdata, event.ydata)

        if mode == 0:
            self.points.append(click_pt)
            self.redraw()
        elif mode == 3:
            idx = self.find_closest_point(click_pt)
            if idx is not None:
                self.is_dragging = True
                self.selected_point_idx = idx
        elif mode in (1, 2):
            self.is_dragging = True
            self.start_pt = click_pt
            self.end_pt = click_pt

    def on_drag(self, event):
        if not self.is_dragging or event.inaxes != self.ax: return
        
        mode = self.btn_group.checkedId()
        current_pt = (event.xdata, event.ydata)

        if mode == 3 and self.selected_point_idx is not None:
            old_pt = self.points[self.selected_point_idx]
            self.points[self.selected_point_idx] = current_pt
            
            for i, line in enumerate(self.lines):
                p1, p2 = line
                if np.allclose(p1, old_pt, atol=1e-5): p1 = current_pt
                if np.allclose(p2, old_pt, atol=1e-5): p2 = current_pt
                self.lines[i] = (p1, p2)
                
            for i, curve in enumerate(self.curves):
                for j, pt in enumerate(curve):
                    if np.allclose(pt, old_pt, atol=1e-5):
                        self.curves[i][j] = current_pt
                        
            self.redraw()
        elif mode in (1, 2):
            self.end_pt = current_pt
            self.redraw()

    def on_release(self, event):
        if not self.is_dragging: return
        self.is_dragging = False

        mode = self.btn_group.checkedId()
        x_end = event.xdata if event.xdata is not None else (self.end_pt[0] if self.end_pt else 0)
        y_end = event.ydata if event.ydata is not None else (self.end_pt[1] if self.end_pt else 0)

        if mode == 3:
            self.selected_point_idx = None
            self.redraw()
        elif mode in (1, 2) and self.start_pt:
            x_start, y_start = self.start_pt
            x_min, x_max = sorted([x_start, x_end])
            y_min, y_max = sorted([y_start, y_end])

            box_points = [
                pt for pt in self.points 
                if x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max
            ]

            if len(box_points) >= 2:
                dx = x_end - x_start
                dy = y_end - y_start
                drag_len = np.hypot(dx, dy)

                if drag_len > 1e-5:
                    ux = dx / drag_len
                    uy = dy / drag_len
                    def get_projection_score(pt):
                        return (pt[0] - x_start) * ux + (pt[1] - y_start) * uy
                    selected_points = sorted(box_points, key=get_projection_score)
                else:
                    selected_points = box_points

                if mode == 1:
                    self.lines.append((selected_points[0], selected_points[-1]))
                elif mode == 2:
                    self.curves.append(selected_points)

            self.start_pt = None
            self.end_pt = None
            self.redraw()

    # --- 🪞 Y축 대칭 기능 ---
    def mirror_along_y(self):
        if not self.points and not self.lines and not self.curves:
            QMessageBox.warning(self, "경고", "대칭 복사할 데이터가 없습니다.")
            return

        def mirror_pt(pt):
            return (-pt[0], pt[1])

        self.points.extend([mirror_pt(pt) for pt in self.points])
        self.lines.extend([(mirror_pt(l[0]), mirror_pt(l[1])) for l in self.lines])
        self.curves.extend([[mirror_pt(pt) for pt in c] for c in self.curves])
        self.redraw()
        QMessageBox.information(self, "대칭 완료", "Y축 기준 대칭 도형이 생성되었습니다.")

    # --- 🔄 초기화(Reset) 기능 구현 ---
    def reset_data(self):
        # 하나도 바뀐 게 없다면 경고창 없이 통과
        if not self.points and not self.lines and not self.curves:
            return
            
        reply = QMessageBox.question(
            self, '초기화 확인', 
            '작성 중인 모든 포인트와 선 데이터가 완전히 삭제됩니다.\n정말로 초기화하시겠습니까?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.points = []
            self.lines = []
            self.curves = []
            self.redraw()

    # --- 저장 및 불러오기 (JSON) ---
    def save_data(self):
        data = {"points": self.points, "lines": self.lines, "curves": self.curves}
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Shape Data", "", "JSON Files (*.json)")
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            QMessageBox.information(self, "저장 완료", "데이터가 성공적으로 저장되었습니다.")

    def load_data(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Shape Data", "", "JSON Files (*.json)")
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.points = [tuple(p) for p in data.get("points", [])]
                self.lines = [(tuple(l[0]), tuple(l[1])) for l in data.get("lines", [])]
                self.curves = [[tuple(p) for p in c] for c in data.get("curves", [])]
                self.redraw()
                QMessageBox.information(self, "불러오기 완료", "데이터를 불러왔습니다.")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"파일 로드 실패: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = AdvancedShapeDemo()
    demo.show()
    sys.exit(app.exec())