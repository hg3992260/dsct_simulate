import sys
import warnings
# Suppress annoying PyQt5/SIP deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message="sipPyTypeDict")

import numpy as np
from numpy.fft import fft, ifft, fftfreq, fftshift
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QSlider, QGroupBox, QFrame, QScrollArea, QSizePolicy, QCheckBox, QSplitter, QTabWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QVector3D
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from skimage.transform import radon, iradon, rescale
from skimage.draw import disk

# --- 1. 核心几何计算函数 (保持不变) ---

def calculate_geometry(alpha, RA, RB, FDD, SFOV_A, SFOV_B, Z_coverage, rotation_time, is_asymmetric=False, required_min_dist=150, required_arc_diff=120):
    """根据输入参数计算所有相关几何值和约束。"""
    
    # 转换为弧度
    alpha_rad = np.deg2rad(alpha)
    
    # 扇形半角 (Beta) - XY平面
    try:
        beta_A = np.arcsin((SFOV_A / 2) / RA)
        beta_B_orig = np.arcsin((SFOV_B / 2) / RB)
    except ValueError:
        return {"error": "SFOV/2 必须小于 F-ISO 距离 R"}, None, None

    # Handle Asymmetric Logic
    arc_B_orig = FDD * 2 * beta_B_orig
    arc_extension = 120.0 # mm
    
    if is_asymmetric:
        # Effective Arc (Virtual Left + Real Right)
        arc_B_effective = arc_B_orig + 2 * arc_extension
        # New Beta based on effective arc
        beta_B = (arc_B_effective / FDD) / 2
        # New SFOV B based on effective beta
        SFOV_B_effective = 2 * RB * np.sin(beta_B)
        
        # Inner SFOV (Double Sampled / Symmetric part)
        SFOV_B_inner = 2 * RB * np.sin(beta_B_orig)
        
        # Physical Mass Increase (Only Right side is real)
        # Assuming original mass M_det corresponds to arc_B_orig
        # Mass factor = (arc_B_orig + arc_extension) / arc_B_orig
        mass_factor_B = (arc_B_orig + arc_extension) / arc_B_orig
        
        # Physical SFOV B (For Scatter calculation)
        # Physical Arc = Original + Right Extension
        arc_B_physical = arc_B_orig + arc_extension
        beta_B_physical = (arc_B_physical / FDD) / 2
        SFOV_B_physical = 2 * RB * np.sin(beta_B_physical)
        
        # Asymmetric Sector Angle (in degrees) for temporal resolution
        angle_ext_rad = arc_extension / FDD
        angle_ext_deg = np.rad2deg(angle_ext_rad)
    else:
        beta_B = beta_B_orig
        SFOV_B_effective = SFOV_B
        SFOV_B_inner = SFOV_B
        mass_factor_B = 1.0
        SFOV_B_physical = SFOV_B
        angle_ext_deg = 0.0

    # 扇形全角
    fan_angle_A = np.rad2deg(2 * beta_A)
    fan_angle_B = np.rad2deg(2 * beta_B)
    
    # 锥角半角 (Kappa) - Z轴方向
    kappa_A = np.arctan((Z_coverage / 2) / RA)
    kappa_B = np.arctan((Z_coverage / 2) / RB)
    
    # 探测器高度 (Z方向)
    H_det_A = Z_coverage * FDD / RA
    H_det_B = Z_coverage * FDD / RB

    # 弧长 (Arc Length)
    arc_A = FDD * 2 * beta_A
    arc_B = FDD * 2 * beta_B # This is effective arc if asymmetric
    arc_diff = abs(arc_A - arc_B)
    
    # 坐标系定义: ISO 为原点 (0,0,0)
    FA_x, FA_y = RA * np.cos(-alpha_rad / 2), RA * np.sin(-alpha_rad / 2)
    FB_x, FB_y = RB * np.cos(alpha_rad / 2), RB * np.sin(alpha_rad / 2)
    
    # DetA XY 边缘点
    angle_FA_to_ISO = np.pi - alpha_rad / 2
    
    # --- 物理引擎: 碰撞检测 (Physics Engine) ---
    num_sample_points = 50
    
    # DetA 采样点
    theta_A = np.linspace(angle_FA_to_ISO - beta_A, angle_FA_to_ISO + beta_A, num_sample_points)
    DetA_x = FA_x + FDD * np.cos(theta_A)
    DetA_y = FA_y + FDD * np.sin(theta_A)
    pts_A = np.column_stack((DetA_x, DetA_y))
    
    # DetB 采样点
    angle_FB_to_ISO = np.pi + alpha_rad / 2
    
    if is_asymmetric:
        # Physical Det B: [Center - Beta_Orig, Center + Beta_Orig + Extension]
        # Left side is virtual (Center - Beta_New to Center - Beta_Orig) -> No Collision
        # Right side is real (Center + Beta_Orig to Center + Beta_Orig + Extension) -> Collision
        
        # Original Part
        theta_B_orig = np.linspace(angle_FB_to_ISO - beta_B_orig, angle_FB_to_ISO + beta_B_orig, num_sample_points)
        
        # Right Extension Part
        angle_ext = arc_extension / FDD
        theta_B_ext = np.linspace(angle_FB_to_ISO + beta_B_orig, angle_FB_to_ISO + beta_B_orig + angle_ext, 10)
        
        theta_B = np.concatenate([theta_B_orig, theta_B_ext])
    else:
        theta_B = np.linspace(angle_FB_to_ISO - beta_B, angle_FB_to_ISO + beta_B, num_sample_points)
        
    DetB_x = FB_x + FDD * np.cos(theta_B)
    DetB_y = FB_y + FDD * np.sin(theta_B)
    pts_B = np.column_stack((DetB_x, DetB_y))
    
    # 1. DetA <-> TubeB (FB) 距离
    dists_A_TubeB = np.sqrt(np.sum((pts_A - np.array([FB_x, FB_y]))**2, axis=1))
    min_dist_A_TubeB = np.min(dists_A_TubeB)
    idx_A_TubeB = np.argmin(dists_A_TubeB)
    closest_pt_A_TubeB = (pts_A[idx_A_TubeB][0], pts_A[idx_A_TubeB][1], 0)
    
    # 2. DetB <-> TubeA (FA) 距离
    dists_B_TubeA = np.sqrt(np.sum((pts_B - np.array([FA_x, FA_y]))**2, axis=1))
    min_dist_B_TubeA = np.min(dists_B_TubeA)
    idx_B_TubeA = np.argmin(dists_B_TubeA)
    closest_pt_B_TubeA = (pts_B[idx_B_TubeA][0], pts_B[idx_B_TubeA][1], 0)
    
    # 3. DetA <-> DetB 距离
    dists_Det_Det = np.sqrt(((pts_A[:, np.newaxis, :] - pts_B[np.newaxis, :, :]) ** 2).sum(axis=2))
    min_dist_Det_Det = np.min(dists_Det_Det)
    
    min_idx = np.unravel_index(np.argmin(dists_Det_Det), dists_Det_Det.shape)
    closest_pt_DetA_DetB = (pts_A[min_idx[0]][0], pts_A[min_idx[0]][1], 0)
    closest_pt_DetB_DetA = (pts_B[min_idx[1]][0], pts_B[min_idx[1]][1], 0)
    
    # 综合最小距离
    min_dist_exact = float(min(min_dist_A_TubeB, min_dist_B_TubeA, min_dist_Det_Det))
    
    # --- 离心力计算 (Centrifugal Force) ---
    M_tube = 30.0 # kg
    M_det = 15.0  # kg
    M_det_B_real = M_det * mass_factor_B
    
    omega = 2 * np.pi / rotation_time # rad/s
    
    # 1. 各个组件的离心力 (标量)
    F_cent_TubeA = M_tube * (omega ** 2) * (RA / 1000) # r in meters
    F_cent_TubeB = M_tube * (omega ** 2) * (RB / 1000)
    
    R_DetA = (FDD - RA)
    R_DetB = (FDD - RB)
    
    F_cent_DetA = M_det * (omega ** 2) * (R_DetA / 1000)
    F_cent_DetB = M_det_B_real * (omega ** 2) * (R_DetB / 1000)
    
    # 2. 矢量合力 (Vector Sum)
    angle_TubeA = -alpha_rad / 2
    angle_TubeB = alpha_rad / 2
    
    Fx_TubeA = F_cent_TubeA * np.cos(angle_TubeA)
    Fy_TubeA = F_cent_TubeA * np.sin(angle_TubeA)
    
    Fx_TubeB = F_cent_TubeB * np.cos(angle_TubeB)
    Fy_TubeB = F_cent_TubeB * np.sin(angle_TubeB)
    
    Fx_total = Fx_TubeA + Fx_TubeB
    Fy_total = Fy_TubeA + Fy_TubeB
    
    # 按照截图公式计算合力 (System Pressure = 2 * Vector Sum)
    F_total_mag = np.sqrt(F_cent_TubeA**2 + F_cent_TubeB**2 + 2 * F_cent_TubeA * F_cent_TubeB * np.cos(alpha_rad)) * 2
    
    # 计算系统合力的 G 值
    total_mass_tubes = 2 * M_tube
    g_force_total = F_total_mag / (total_mass_tubes * 9.8)
    
    # --- 物理时间分辨 (Physical Temporal Resolution) ---
    # Formula updated: (rotation_time / 4) * ((alpha + asymmetric_angle) / 90)
    effective_alpha = alpha + angle_ext_deg
    temporal_resolution = (rotation_time / 4) * (effective_alpha / 90)
    
    # --- 散射干扰系数 (Scatter Interference Coefficient) ---
    # MUST be based on REAL Physical Volume B
    ref_sfov = 500.0
    ref_z = 40.0
    vol_ref = (ref_sfov ** 2) * ref_z
    
    vol_A = (SFOV_A ** 2) * Z_coverage
    vol_B = (SFOV_B_physical ** 2) * Z_coverage # Using Physical SFOV
    
    scatter_coeff = (vol_A + vol_B) / vol_ref
    
    # --- 受影响的FOV Sphere体积 (Volume of Affected FOV Sphere) ---
    # Volume = (4/3) * pi * (R_outer^3 - R_inner^3)
    # R = SFOV / 2
    r_outer = SFOV_B_effective / 2
    r_inner = SFOV_B_inner / 2
    affected_volume_cm3 = ((4/3) * np.pi * (r_outer**3 - r_inner**3)) / 1000.0 # Convert mm^3 to cm^3 (mL)
    
    # --- 约束检查 ---
    is_safe = min_dist_exact >= required_min_dist
    is_arc_ok = arc_diff >= required_arc_diff
    
    results = {
        "fan_angle_A": float(fan_angle_A),
        "fan_angle_B": float(fan_angle_B),
        "arc_diff": float(arc_diff),
        "min_dist": float(min_dist_exact),
        "is_safe": bool(is_safe),
        "is_arc_ok": bool(is_arc_ok),
        "RA": float(RA),
        "RB": float(RB),
        "beta_A": float(beta_A), "beta_B": float(beta_B),
        "beta_B_orig": float(beta_B_orig), # Keep track of original beta
        "kappa_A": float(kappa_A), "kappa_B": float(kappa_B),
        "H_det_A": float(H_det_A), "H_det_B": float(H_det_B),
        "R_A_coord": (float(FA_x), float(FA_y), 0),
        "R_B_coord": (float(FB_x), float(FB_y), 0),
        
        "min_dist_A_TubeB": float(min_dist_A_TubeB),
        "min_dist_B_TubeA": float(min_dist_B_TubeA),
        "min_dist_Det_Det": float(min_dist_Det_Det),
        
        "closest_pt_A_TubeB": (float(closest_pt_A_TubeB[0]), float(closest_pt_A_TubeB[1]), 0),
        "closest_pt_B_TubeA": (float(closest_pt_B_TubeA[0]), float(closest_pt_B_TubeA[1]), 0),
        "closest_pt_DetA_DetB": (float(closest_pt_DetA_DetB[0]), float(closest_pt_DetA_DetB[1]), 0),
        "closest_pt_DetB_DetA": (float(closest_pt_DetB_DetA[0]), float(closest_pt_DetB_DetA[1]), 0),
        
        "g_force_TubeA": float(F_cent_TubeA / 9.8 / M_tube),
        "g_force_TubeB": float(F_cent_TubeB / 9.8 / M_tube),
        "g_force_DetA": float(F_cent_DetA / 9.8 / M_det),
        "g_force_DetB": float(F_cent_DetB / 9.8 / M_det_B_real), # Use real mass
        "F_total_mag": float(F_total_mag),
        "g_force_total": float(g_force_total),
        "F_total_vector": (float(Fx_total), float(Fy_total)),
        
        "temporal_resolution": float(temporal_resolution),
        "scatter_coeff": float(scatter_coeff),
        "affected_volume_cm3": float(affected_volume_cm3), # 新增体积
        
        "alpha_rad": float(alpha_rad),
        "is_asymmetric": is_asymmetric,
        "SFOV_B_effective": float(SFOV_B_effective),
        "SFOV_B_inner": float(SFOV_B_inner)
    }
    return results, fan_angle_A, fan_angle_B

# --- 2. GUI 组件 ---

class ParameterSlider(QWidget):
    def __init__(self, name, min_val, max_val, step, default_val, unit="", parent=None, is_float=False):
        super().__init__(parent)
        self.is_float = is_float
        self.scale_factor = 100 if is_float else 1
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 5, 0, 5)
        
        # Label Row
        label_layout = QHBoxLayout()
        self.name_label = QLabel(name)
        self.value_label = QLabel(f"{default_val} {unit}")
        self.value_label.setAlignment(Qt.AlignRight)
        label_layout.addWidget(self.name_label)
        label_layout.addWidget(self.value_label)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(min_val * self.scale_factor))
        self.slider.setMaximum(int(max_val * self.scale_factor))
        self.slider.setSingleStep(int(step * self.scale_factor))
        self.slider.setValue(int(default_val * self.scale_factor))
        self.slider.valueChanged.connect(self.update_label)
        
        layout.addLayout(label_layout)
        layout.addWidget(self.slider)
        self.setLayout(layout)
        
        self.unit = unit
        self.callback = None

    def update_label(self, val):
        real_val = val / self.scale_factor
        fmt = "{:.2f}" if self.is_float else "{:.0f}"
        self.value_label.setText((fmt + " {}").format(real_val, self.unit))
        if self.callback:
            self.callback()

    def value(self):
        return self.slider.value() / self.scale_factor

class ReconstructionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Tabs for different visualizations
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        # Tab 1: Sinogram & Truncation Profile
        self.tab_sino = QWidget()
        self.layout_sino = QVBoxLayout(self.tab_sino)
        
        # Splitter for Original vs Sinogram
        self.sino_top_splitter = QSplitter(Qt.Horizontal)
        
        # 1. Original Phantom View (Left)
        self.phantom_plot = pg.ImageView()
        self.phantom_plot.ui.histogram.hide()
        self.phantom_plot.ui.roiBtn.hide()
        self.phantom_plot.ui.menuBtn.hide()
        self.phantom_plot.getView().setAspectLocked(True)
        # self.phantom_plot.getView().setTitle("Original Phantom")
        
        # Legend for Phantom Plot
        self.legend = pg.LegendItem(offset=(50, 10))
        self.legend.setParentItem(self.phantom_plot.getView())
        
        # Add FOV Overlays to Phantom Plot
        self.fov_curve_inner = pg.PlotCurveItem(pen=pg.mkPen('g', width=2, style=Qt.DashLine), name="Safe FOV (Symmetric)")
        self.fov_curve_outer = pg.PlotCurveItem(pen=pg.mkPen('r', width=2), name="Max FOV (Effective)")
        self.phantom_plot.getView().addItem(self.fov_curve_inner)
        self.phantom_plot.getView().addItem(self.fov_curve_outer)
        
        # Add legend items
        self.legend.addItem(self.fov_curve_inner, "Safe FOV")
        self.legend.addItem(self.fov_curve_outer, "Max FOV")
        
        # Add Text Description for Artifacts
        text = pg.TextItem(
            html='<div style="text-align: center"><span style="color: #FFF;">Gray: Accurate Data</span><br><span style="color: #FF4444;">Red Glow: Truncation Artifacts (Data Loss)</span></div>',
            anchor=(0.5, 0)
        )
        self.phantom_plot.getView().addItem(text)
        text.setPos(0, -60) # Position near top
        
        self.sino_top_splitter.addWidget(self.phantom_plot)
        
        # 2. Sinogram View (Right)
        self.sino_plot = pg.ImageView()
        self.sino_plot.ui.histogram.hide()
        self.sino_plot.ui.roiBtn.hide()
        self.sino_plot.ui.menuBtn.hide()
        self.sino_plot.getView().setAspectLocked(False)
        # self.sino_plot.getView().setTitle("Sinogram (Truncated)")
        self.sino_top_splitter.addWidget(self.sino_plot)
        
        self.layout_sino.addWidget(self.sino_top_splitter)
        
        self.profile_plot = pg.PlotWidget(title="Truncation Profile (Center Row)")
        self.profile_plot.setLabel('bottom', "Detector Channel")
        self.profile_plot.setLabel('left', "Intensity")
        self.layout_sino.addWidget(self.profile_plot)
        self.layout_sino.setStretch(0, 2)
        self.layout_sino.setStretch(1, 1)
        
        self.tabs.addTab(self.tab_sino, "Sinogram")

        # Tab 2: Ideal Reconstruction (Reference)
        self.tab_ideal = QWidget()
        self.layout_ideal = QVBoxLayout(self.tab_ideal)
        
        self.ideal_plot = pg.ImageView()
        self.ideal_plot.ui.histogram.hide()
        self.ideal_plot.ui.roiBtn.hide()
        self.ideal_plot.ui.menuBtn.hide()
        self.ideal_plot.getView().setAspectLocked(True)
        self.layout_ideal.addWidget(self.ideal_plot)
        
        self.tabs.addTab(self.tab_ideal, "Ideal Recon")
        
        # Tab 3: Actual Reconstruction (Truncated)
        self.tab_recon = QWidget()
        self.layout_recon = QVBoxLayout(self.tab_recon)
        
        self.recon_plot = pg.ImageView()
        self.recon_plot.ui.histogram.hide()
        self.recon_plot.ui.roiBtn.hide()
        self.recon_plot.ui.menuBtn.hide()
        self.recon_plot.getView().setAspectLocked(True)
        # self.recon_plot.getView().setTitle("FBP Reconstruction Impact")
        self.layout_recon.addWidget(self.recon_plot)
        
        self.tabs.addTab(self.tab_recon, "Actual Recon")

        # Tab 4: Impact Analysis (Difference)
        self.tab_impact = QWidget()
        self.layout_impact = QVBoxLayout(self.tab_impact)
        
        self.impact_plot = pg.ImageView()
        self.impact_plot.ui.histogram.hide()
        self.impact_plot.ui.roiBtn.hide()
        self.impact_plot.ui.menuBtn.hide()
        self.impact_plot.getView().setAspectLocked(True)
        # self.impact_plot.getView().setTitle("Missing Data Artifacts")
        self.layout_impact.addWidget(self.impact_plot)
        
        self.tabs.addTab(self.tab_impact, "Artifact Overlay")

        # Tab 5: FBP Process (Sinogram -> Filtered -> Image)
        self.tab_fbp = QWidget()
        self.layout_fbp = QVBoxLayout(self.tab_fbp)
        
        # Splitter for FBP steps
        self.fbp_splitter = QSplitter(Qt.Vertical)
        
        # 1. Filtered Sinogram View
        self.filt_sino_plot = pg.ImageView()
        self.filt_sino_plot.ui.histogram.hide()
        self.filt_sino_plot.ui.roiBtn.hide()
        self.filt_sino_plot.ui.menuBtn.hide()
        self.filt_sino_plot.getView().setAspectLocked(False)
        # self.filt_sino_plot.getView().setTitle("Step 1: Filtered Sinogram (Ram-Lak)")
        
        # 2. Backprojection View (Same as Recon but emphasized)
        self.fbp_recon_plot = pg.ImageView()
        self.fbp_recon_plot.ui.histogram.hide()
        self.fbp_recon_plot.ui.roiBtn.hide()
        self.fbp_recon_plot.ui.menuBtn.hide()
        self.fbp_recon_plot.getView().setAspectLocked(True)
        # self.fbp_recon_plot.getView().setTitle("Step 2: Backprojection (Image Domain)")
        
        self.fbp_splitter.addWidget(self.filt_sino_plot)
        self.fbp_splitter.addWidget(self.fbp_recon_plot)
        
        self.layout_fbp.addWidget(self.fbp_splitter)
        self.tabs.addTab(self.tab_fbp, "FBP Process")
        
        # Internal State
        self.phantom_size = 128 # Keep small for real-time
        self.phantom = self.create_phantom(self.phantom_size)
        self.last_params = None

    def create_phantom(self, size):
        image = np.zeros((size, size))
        # Main body
        rr, cc = disk((size//2, size//2), size//2 - 5)
        image[rr, cc] = 1.0
        # Internal features
        rr, cc = disk((size//2 - size//4, size//2), size//8)
        image[rr, cc] = 0.5
        rr, cc = disk((size//2 + size//4, size//2), size//8)
        image[rr, cc] = 0.8
        rr, cc = disk((size//2, size//2 - size//4), size//10)
        image[rr, cc] = 0.2
        return image

    def update_data(self, results, params):
        # Determine sampling parameters
        sampling_rate = params.get('sampling_rate', 2000) # Hz
        rotation_time = params['rotation_time']
        
        # 1. Calculate Angles
        # For the comparison to be accurate regarding "removing the 120mm arc",
        # we need to simulate the STATIC truncation effect (0-180), 
        # because if we rotate 360, the system compensates and the difference becomes zero (no red pixels).
        # To show "Red pixels when clicking asymmetric", we must show the IMPACT of that missing data 
        # assuming the system hasn't spun around to fix it yet (or showing the raw data loss).
        
        # However, if the user wants to see "The effect of removing data", 
        # usually implies "Where is this data in the image?".
        # Using 0-180 (Short Scan) is the standard way to map sinogram loss to image space loss.
        
        n_projections = min(int(sampling_rate * rotation_time), 360) 
        theta = np.linspace(0., 180., n_projections, endpoint=False)
        
        # 2. Simulate Sinogram (Full / Ground Truth / Symmetric Reference)
        # This corresponds to the "Grey" image (Complete Symmetric FOV B)
        sinogram_full = radon(self.phantom, theta=theta, circle=True)
        
        # 3. Apply Truncation (Asymmetric FOV)
        # Map physical FOV to detector pixels (sinogram rows)
        # Phantom covers "Ref SFOV" (say 500mm or the max SFOV A)
        # Let's assume the Phantom Radius corresponds to Max(SFOV_A)/2 = 300mm
        max_fov_radius = 300.0 
        
        # Detector Array (Y-axis of sinogram)
        # Size = self.phantom_size (roughly, due to radon transform implementation)
        # Scikit-image radon returns sinogram of shape (M, N) where N is len(theta)
        # M is approx sqrt(2) * size? No, it's the projection size.
        sino_rows = sinogram_full.shape[0]
        
        # Coordinate mapping: Center row is 0 mm.
        # Row 0 is -max_fov_radius, Row M is +max_fov_radius
        pixel_size_mm = (2 * max_fov_radius) / sino_rows
        
        # Calculate active range for System B
        # Center is ISO.
        # Range: [-SFOV_B_left/2, +SFOV_B_right/2] ?
        # Wait, SFOV is diameter. 
        # In asymmetric mode:
        # Left limit (Virtual): -SFOV_B_inner/2 (Symmetric part limit? No.)
        # The code says:
        # Physical Det B: [Center - Beta_Orig, Center + Beta_Orig + Extension]
        # This maps to FOV coverage.
        # Beta_Orig corresponds to SFOV_B_inner/2
        # Beta_Orig + Extension corresponds to SFOV_B_effective/2 (approximately, on the right)
        
        limit_left_mm = -results['SFOV_B_inner'] / 2
        limit_right_mm = results['SFOV_B_effective'] / 2 # This is the outer radius on the right
        
        # But in asymmetric mode, the "Effective SFOV" usually refers to the diameter 
        # if we could rotate 360 and fill the gap.
        # Here we are talking about INSTANTANEOUS FOV coverage in the projection.
        # The detector B physically covers from Angle_ISO - Beta_Orig to Angle_ISO + Beta_Orig + Ext.
        # Projecting to ISO plane:
        # Left edge: - (SFOV_B_inner / 2)
        # Right edge: + (SFOV_B_effective_radius) ?
        # Let's use the Beta angles to be precise.
        # Radius at ISO = R * sin(Beta).
        # Left Limit = - RB * sin(beta_B_orig)
        # Right Limit = + RB * sin(beta_B_orig + angle_ext_rad)
        
        # Recalculate precisely
        RB = params['RB']
        beta_B_orig = results['beta_B_orig'] # rad
        FDD = params['FDD']
        angle_ext_rad = 120.0 / FDD
        
        # 3.1 Calculate Symmetric Limits (Baseline / Gray Image)
        # In symmetric mode, the detector covers [ -R_sym, +R_sym ]
        # The physical symmetric width corresponds to SFOV_B
        # But wait, we want to compare "Symmetric Version of B" vs "Asymmetric Version of B".
        # Symmetric B has width = SFOV_B (which is smaller than Effective Asymmetric Width).
        # OR, does the user mean "Symmetric Equivalent" (Virtual Extended)?
        
        # User said: "Gray is complete symmetric sector".
        # This implies the Baseline is the "Perfect" case where we have the data.
        # Let's assume Baseline = Full Coverage of the Asymmetric EXTENDED range (The effective FOV).
        
        r_effective = RB * np.sin(beta_B_orig + angle_ext_rad) 
        
        min_idx_sym = int(sino_rows/2 - (r_effective / pixel_size_mm))
        max_idx_sym = int(sino_rows/2 + (r_effective / pixel_size_mm))
        
        # Clamp
        min_idx_sym = max(0, min_idx_sym)
        max_idx_sym = min(sino_rows, max_idx_sym)
        
        # Create Symmetric Sinogram (Baseline)
        sino_symmetric = np.zeros_like(sinogram_full)
        sino_symmetric[min_idx_sym:max_idx_sym, :] = sinogram_full[min_idx_sym:max_idx_sym, :]
        
        # 3.2 Calculate Asymmetric Limits (Actual / Red Difference)
        # If checked, we cut off the left side.
        
        if params['is_asymmetric']:
            # Left side is truncated to 'r_left_actual' (Short side)
            r_left_actual = RB * np.sin(beta_B_orig)
            r_right_actual = RB * np.sin(beta_B_orig + angle_ext_rad)
            
            min_idx_asym = int(sino_rows/2 - (r_left_actual / pixel_size_mm))
            max_idx_asym = int(sino_rows/2 + (r_right_actual / pixel_size_mm))
            
            # Clamp
            min_idx_asym = max(0, min_idx_asym)
            max_idx_asym = min(sino_rows, max_idx_asym)
            
            # Create Asymmetric Sinogram
            sino_asym = np.zeros_like(sinogram_full)
            sino_asym[min_idx_asym:max_idx_asym, :] = sinogram_full[min_idx_asym:max_idx_asym, :]
            
            # The "Masked Sino" for reconstruction pipeline is the Asymmetric one
            masked_sino = sino_asym
            
            # For visualization of circles
            r_left = r_left_actual
            r_right = r_effective
            
            # Difference for Red Pixels
            # Diff = Symmetric - Asymmetric
            # This contains exactly the 120mm arc data
            diff_sino = sino_symmetric - sino_asym
            
        else:
            # Symmetric Mode
            masked_sino = sino_symmetric
            diff_sino = np.zeros_like(sinogram_full) # No difference
            
            r_left = r_effective
            r_right = r_effective
            
        # Update FOV Circles on Phantom
        center_px = self.phantom_size / 2
        mm_to_px = self.phantom_size / (2 * max_fov_radius)
        
        # Inner Circle
        r_in_px = r_left * mm_to_px
        theta_circle = np.linspace(0, 2*np.pi, 100)
        x_in = center_px + r_in_px * np.cos(theta_circle)
        y_in = center_px + r_in_px * np.sin(theta_circle)
        self.fov_curve_inner.setData(x_in, y_in)
        
        # Outer Circle
        r_out_px = r_right * mm_to_px
        x_out = center_px + r_out_px * np.cos(theta_circle)
        y_out = center_px + r_out_px * np.sin(theta_circle)
        self.fov_curve_outer.setData(x_out, y_out)
            
        # Clamp indices
        # min_idx = max(0, min_idx)
        # max_idx = min(sino_rows, max_idx)
        
        # masked_sino[min_idx:max_idx, :] = sinogram[min_idx:max_idx, :]
        
        # 4. Reconstruct
        # A. Ideal Reconstruction (Full Data / Reference)
        recon_ideal = iradon(sinogram_full, theta=theta, circle=True)
        
        # B. Truncated Reconstruction (Actual)
        reconstruction = iradon(masked_sino, theta=theta, circle=True)
        
        # C. Missing Data Reconstruction (Difference)
        # Difference Sinogram = Full - Masked
        # For our visualization request:
        # We want to show the difference between "Symmetric B" and "Asymmetric B".
        # This is exactly what we calculated in `diff_sino`.
        impact_image = iradon(diff_sino, theta=theta, circle=True)
        
        # Create High Contrast Overlay (Red Artifacts on Grayscale Phantom)
        # 1. Prepare Base (Phantom) - Normalize to 0-1
        base = self.phantom.copy()
        if base.max() > base.min():
            base = (base - base.min()) / (base.max() - base.min())
        
        # 2. Prepare Artifact Map (Magnitude of missing info)
        # Scale up the artifacts to make them visible (High Contrast)
        artifact = np.abs(impact_image)
        artifact_vis = np.clip(artifact * 5.0, 0, 1) # Gain of 5
        
        # 3. Create RGB Composite
        # We want the artifacts to appear as glowing RED/ORANGE on top of the grayscale phantom
        # R = Base + Artifact (Boost Red)
        # G = Base - Artifact (Suppress Green -> Red shift)
        # B = Base - Artifact (Suppress Blue -> Red shift)
        overlay_rgb = np.zeros((base.shape[0], base.shape[1], 3))
        
        # Logic: If artifact is near zero (gray part), we just show base.
        # If artifact is high (red part), we show red.
        
        # NOTE: 'base' is the pure phantom.
        # However, for correct visualization, the 'base' image should ideally be the 
        # SYMMETRIC reconstruction result (the gray image), not just the phantom.
        # Because we want to show: (Symmetric Recon) + (Red Artifacts).
        # But 'self.phantom' is the ground truth, which is even better.
        # Let's stick to using the Phantom as the base layer for clarity.
        
        overlay_rgb[..., 0] = np.clip(base + artifact_vis, 0, 1) # Red channel
        overlay_rgb[..., 1] = np.clip(base - artifact_vis * 0.5, 0, 1) # Green channel
        overlay_rgb[..., 2] = np.clip(base - artifact_vis * 0.5, 0, 1) # Blue channel
        
        # D. Filtered Sinogram Visualization (Manual Calculation for Display)
        # Filter along rows (detector channels)
        rows = masked_sino.shape[0]
        # Frequency domain
        freq = fftfreq(rows).reshape(-1, 1)
        omega = np.abs(freq) # Ram-Lak filter
        
        # FFT
        sino_fft = fft(masked_sino, axis=0)
        # Apply Filter
        filtered_fft = sino_fft * omega
        # IFFT
        filtered_sino = np.real(ifft(filtered_fft, axis=0))
        
        # 5. Update Plots
        # Transpose for pyqtgraph (Col, Row)
        # self.phantom_plot.setImage(self.phantom.T)
        # Display Overlay in Phantom Plot as well to show impact in context
        self.phantom_plot.setImage(np.transpose(overlay_rgb, (1, 0, 2)))
        
        self.sino_plot.setImage(masked_sino.T)
        self.ideal_plot.setImage(recon_ideal.T)
        self.recon_plot.setImage(reconstruction.T)
        
        # Display Overlay in Impact Tab (Transpose spatial dims: (W, H, 3) -> (H, W, 3) effectively)
        # pyqtgraph expects (width, height, channels)
        # Our overlay_rgb is (row, col, 3).
        # We need to transpose the first two dimensions to match the other plots
        self.impact_plot.setImage(np.transpose(overlay_rgb, (1, 0, 2)))
        
        # FBP Process Plots
        self.filt_sino_plot.setImage(filtered_sino.T)
        self.fbp_recon_plot.setImage(reconstruction.T)
        
        # Profile
        center_angle_idx = n_projections // 2
        profile = masked_sino[:, center_angle_idx]
        self.profile_plot.plot(profile, clear=True, pen='y')
        # Add limit lines
        
        # Use r_left/right derived values mapped to indices for display
        min_idx_disp = int(sino_rows/2 - (r_left / pixel_size_mm))
        max_idx_disp = int(sino_rows/2 + (r_right / pixel_size_mm))
        
        self.profile_plot.addItem(pg.InfiniteLine(pos=min_idx_disp, angle=90, pen='r'))
        self.profile_plot.addItem(pg.InfiniteLine(pos=max_idx_disp, angle=90, pen='r'))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("双源 CT 真实几何模拟器 (Cone Beam) - Native Edition")
        self.resize(1280, 800)

        # Main Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel: Controls ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(400)
        
        # Title
        title = QLabel("系统几何参数")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        # Signature Box
        sig_box = QFrame()
        sig_box.setStyleSheet("""
            QFrame {
                background-color: #e8f4f8;
                border: 1px solid #bce8f1;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        sig_layout = QVBoxLayout(sig_box)
        l1 = QLabel("Developed by Christ.paul90@gmail.com")
        l1.setStyleSheet("font-style: italic; color: #31708f; font-size: 11px;")
        l2 = QLabel("All the results have been rigorously supported by geometric parameters and the physics engine.")
        l2.setStyleSheet("font-weight: bold; color: #31708f; font-size: 11px;")
        l2.setWordWrap(True)
        sig_layout.addWidget(l1)
        sig_layout.addWidget(l2)
        left_layout.addWidget(sig_box)

        # Sliders
        self.sliders = {}
        self.add_slider("中心线夹角 α", 45, 150, 0.5, 95, "°", left_layout, "alpha", True)
        self.add_slider("F-ISO A (RA)", 400, 700, 1, 600, "mm", left_layout, "RA")
        self.add_slider("F-ISO B (RB)", 400, 700, 1, 600, "mm", left_layout, "RB")
        self.add_slider("FDD", 800, 1200, 1, 1100, "mm", left_layout, "FDD")
        self.add_slider("SFOV A", 400, 600, 1, 500, "mm", left_layout, "SFOV_A")
        self.add_slider("SFOV B", 300, 600, 1, 350, "mm", left_layout, "SFOV_B")
        self.add_slider("Z轴覆盖", 10, 160, 1, 80, "mm", left_layout, "Z_coverage")
        self.add_slider("旋转时间", 0.2, 0.5, 0.01, 0.28, "s", left_layout, "rotation_time", True)
        self.add_slider("探测器采样率", 1000, 5000, 100, 2000, "Hz", left_layout, "sampling_rate")

        # Asymmetric Mode Checkbox
        self.asymmetric_cb = QCheckBox("非对称扇区 (Asymmetric Mode)")
        self.asymmetric_cb.setStyleSheet("font-size: 14px; font-weight: bold; color: #333; margin-top: 10px;")
        self.asymmetric_cb.stateChanged.connect(self.update_simulation)
        left_layout.addWidget(self.asymmetric_cb)

        left_layout.addStretch()

        # Results Panel (Bottom Left)
        self.result_label = QLabel()
        self.result_label.setTextFormat(Qt.RichText)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("background-color: #f9f9f9; padding: 15px; border: 1px solid #ccc; font-size: 16px; font-family: Arial;")
        left_layout.addWidget(self.result_label)

        main_layout.addWidget(left_panel)

        # --- Right Panel: Split 3D Views & Reconstruction ---
        right_v_splitter = QSplitter(Qt.Vertical)
        
        # Top: 3D Views (Horizontal Split)
        top_h_splitter = QSplitter(Qt.Horizontal)
        
        # View 1: System Geometry (Global View)
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((230, 247, 255, 255)) 
        self.view.opts['distance'] = 2000
        self.view.opts['elevation'] = 90
        self.view.opts['azimuth'] = -90
        self.view.setWindowTitle("System Geometry")
        
        # View 2: FOV Impact Analysis (Local/FOV View)
        self.view_fov = gl.GLViewWidget()
        self.view_fov.setBackgroundColor((30, 30, 30, 255)) # Dark background for contrast
        self.view_fov.opts['distance'] = 800
        self.view_fov.opts['elevation'] = 0 # Side view / Axial view
        self.view_fov.opts['azimuth'] = 0
        self.view_fov.setWindowTitle("FOV Impact Analysis")
        
        top_h_splitter.addWidget(self.view)
        top_h_splitter.addWidget(self.view_fov)
        
        right_v_splitter.addWidget(top_h_splitter)
        
        # Bottom: Reconstruction Analysis (Third Independent Graph)
        self.recon_widget = ReconstructionWidget()
        right_v_splitter.addWidget(self.recon_widget)
        
        # Set layout stretch
        right_v_splitter.setStretchFactor(0, 6) # Top 60%
        right_v_splitter.setStretchFactor(1, 4) # Bottom 40%
        
        main_layout.addWidget(right_v_splitter, 1)

        # Add Grid to View 1
        # Darker grid lines for contrast against light blue
        grid_color = (50, 50, 50, 80) # Dark gray with transparency
        
        # Floor Grid (XY Plane)
        gz = gl.GLGridItem(color=grid_color)
        gz.translate(0, 0, -500)
        gz.setSize(x=2000, y=2000, z=0)
        gz.setSpacing(x=100, y=100, z=0)
        self.view.addItem(gz)
        
        # Back Grid (XZ Plane)
        gy = gl.GLGridItem(color=grid_color)
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -1000, 0)
        gy.setSize(x=2000, y=2000, z=0)
        gy.setSpacing(x=100, y=100, z=0)
        # self.view.addItem(gy) # Optional, can be cluttered

        # Side Grid (YZ Plane)
        gx = gl.GLGridItem(color=grid_color)
        gx.rotate(90, 0, 1, 0)
        gx.translate(-1000, 0, 0)
        gx.setSize(x=2000, y=2000, z=0)
        gx.setSpacing(x=100, y=100, z=0)
        # self.view.addItem(gx) # Optional

        # Add Axes - Centered at ISO
        axis = gl.GLAxisItem()
        axis.setSize(600, 600, 600) # Increase axis size further
        self.view.addItem(axis)

        # Add Axis Labels (X, Y, Z) and Ticks
        self.axis_labels = []
        
        # Main Labels
        for ax, pos, color in zip(['X', 'Y', 'Z'], [(650,0,0), (0,650,0), (0,0,650)], [(1,0,0,1), (0,1,0,1), (0,0,1,1)]):
            t = gl.GLTextItem(pos=pos, text=ax, font=QFont('Arial', 20, QFont.Bold), color=color)
            self.view.addItem(t)
            self.axis_labels.append(t)
            
        # Ticks along axes
        for i in range(-1000, 1001, 200):
            if i == 0: continue
            # X Axis Ticks
            t_x = gl.GLTextItem(pos=(i, 0, 0), text=str(i), font=QFont('Arial', 8), color=(0.3, 0.3, 0.3, 1))
            self.view.addItem(t_x)
            
            # Y Axis Ticks
            t_y = gl.GLTextItem(pos=(0, i, 0), text=str(i), font=QFont('Arial', 8), color=(0.3, 0.3, 0.3, 1))
            self.view.addItem(t_y)
            
            # Z Axis Ticks (shorter range)
            if abs(i) <= 500:
                t_z = gl.GLTextItem(pos=(0, 0, i), text=str(i), font=QFont('Arial', 8), color=(0.3, 0.3, 0.3, 1))
                self.view.addItem(t_z)

        # --- HUD (Heads-Up Display) for Parameters ---
        self.hud_label = QLabel(self.view)
        self.hud_label.setStyleSheet("""
            QLabel { 
                color: #000000; 
                background-color: rgba(255, 255, 255, 180); 
                padding: 10px; 
                border: 1px solid #999999; 
                border-radius: 5px;
                font-family: Consolas, monospace;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        self.hud_label.setAttribute(Qt.WA_TransparentForMouseEvents) # Allow mouse interaction with 3D view behind
        self.hud_label.move(10, 10)
        self.hud_label.show()

        main_layout.addWidget(self.view, 1) # Expand 3D view

        # Init Geometry Objects
        self.init_3d_objects()
        
        # Initial Calculation
        self.update_simulation()

    def add_slider(self, name, min_v, max_v, step, default, unit, layout, key, is_float=False):
        slider = ParameterSlider(name, min_v, max_v, step, default, unit, is_float=is_float)
        slider.callback = self.update_simulation
        layout.addWidget(slider)
        self.sliders[key] = slider

    def init_3d_objects(self):
        # --- Init Main View Objects ---
        self.items = {}
        # Placeholders for 3D items
        # Tubes (Points)
        self.items['tubes'] = gl.GLScatterPlotItem(size=25, pxMode=True)
        self.view.addItem(self.items['tubes'])
        
        # ISO (Point)
        self.items['iso'] = gl.GLScatterPlotItem(pos=np.array([[0,0,0]]), color=(0,0,0,1), size=12, pxMode=True)
        self.view.addItem(self.items['iso'])
        
        # Cone Lines
        self.items['lines_A'] = gl.GLLinePlotItem(width=3, color=(0.6, 0, 0, 1), mode='lines') # Dark Red
        self.view.addItem(self.items['lines_A'])
        
        self.items['lines_B'] = gl.GLLinePlotItem(width=5, color=(0, 0, 0.7, 1), mode='lines') # Dark Blue, Thick
        self.view.addItem(self.items['lines_B'])
        
        # Fan Sector (Solid)
        self.items['fan_A'] = gl.GLMeshItem(shader='shaded', color=(0.8, 0, 0, 0.3), glOptions='translucent')
        self.view.addItem(self.items['fan_A'])
        
        self.items['fan_B'] = gl.GLMeshItem(shader='shaded', color=(0, 0, 0.8, 0.4), glOptions='translucent') # Slightly more opaque for B
        self.view.addItem(self.items['fan_B'])

        # Fan Sector B Virtual (Missing/Ghost)
        self.items['fan_B_virtual'] = gl.GLMeshItem(shader='shaded', color=(1, 0, 0, 0.2), glOptions='translucent') # Red tint for missing
        self.view.addItem(self.items['fan_B_virtual'])
        
        # Detectors (Mesh)
        self.items['det_A'] = gl.GLMeshItem(shader='shaded', color=(0.8, 0, 0, 0.8), glOptions='translucent') # Darker Red
        self.view.addItem(self.items['det_A'])
        self.items['det_B'] = gl.GLMeshItem(shader='shaded', color=(0, 0, 0.8, 0.8), glOptions='translucent') # Darker Blue
        self.view.addItem(self.items['det_B'])
        
        # Detector Edges
        self.items['det_A_edge'] = gl.GLLinePlotItem(width=3, color=(0.4, 0, 0, 1), mode='lines') # Very dark red
        self.view.addItem(self.items['det_A_edge'])
        self.items['det_B_edge'] = gl.GLLinePlotItem(width=3, color=(0, 0, 0.4, 1), mode='lines') # Very dark blue
        self.view.addItem(self.items['det_B_edge'])

        # SFOV Sphere A
        self.items['sfov'] = gl.GLMeshItem(shader='shaded', color=(0, 0.8, 0, 0.15), glOptions='translucent') # Darker Green
        self.view.addItem(self.items['sfov'])
        self.items['sfov_equator'] = gl.GLLinePlotItem(width=2.5, color=(0, 0.5, 0, 1), mode='lines') # Dark green
        self.view.addItem(self.items['sfov_equator'])
        
        # SFOV Sphere B (New)
        self.items['sfov_B'] = gl.GLMeshItem(shader='shaded', color=(0, 0.8, 0.8, 0.15), glOptions='translucent') # Cyan/Teal
        self.view.addItem(self.items['sfov_B'])
        self.items['sfov_B_equator'] = gl.GLLinePlotItem(width=2.5, color=(0, 0.5, 0.5, 1), mode='lines') # Dark Cyan
        self.view.addItem(self.items['sfov_B_equator'])
        
        # Force Vector
        self.items['force_vec'] = gl.GLLinePlotItem(width=6, color=(0.4, 0, 0.4, 1), mode='lines')
        self.view.addItem(self.items['force_vec'])
        self.items['force_head'] = gl.GLScatterPlotItem(size=18, pxMode=True, color=(0.4, 0, 0.4, 1))
        self.view.addItem(self.items['force_head'])

        # Angle Arc
        self.items['angle_arc'] = gl.GLLinePlotItem(width=5, color=(0, 0, 0, 1), mode='lines')
        self.view.addItem(self.items['angle_arc'])

        # Text Labels
        self.text_items = {}
        for key in ['A_TubeB', 'B_TubeA', 'Det_Det', 'Angle']:
            t = gl.GLTextItem(pos=(0,0,0), text="", font=QFont('Arial', 14, QFont.Bold), color=(0, 0, 0, 1))
            self.view.addItem(t)
            self.text_items[key] = t
            
        # --- Init FOV View Objects (Secondary View) ---
        self.items_fov = {}
        
        # Add Axes to FOV view
        axis_fov = gl.GLAxisItem()
        axis_fov.setSize(300, 300, 300)
        self.view_fov.addItem(axis_fov)
        
        # SFOV Sphere B (Ghostly Reference)
        self.items_fov['sfov_B'] = gl.GLMeshItem(shader='shaded', color=(0, 1, 1, 0.1), glOptions='translucent')
        self.view_fov.addItem(self.items_fov['sfov_B'])
        
        # Affected FOV Shell (High Contrast Orange)
        self.items_fov['sfov_B_affected'] = gl.GLMeshItem(shader='shaded', color=(1, 0.5, 0, 0.4), glOptions='translucent')
        self.view_fov.addItem(self.items_fov['sfov_B_affected'])
        
        # Real Fan B (Solid Blue)
        self.items_fov['fan_B_real'] = gl.GLMeshItem(shader='shaded', color=(0, 0.5, 1, 0.6), glOptions='translucent')
        self.view_fov.addItem(self.items_fov['fan_B_real'])
        
        # Missing Fan B (Solid Red - Highlight)
        self.items_fov['fan_B_missing'] = gl.GLMeshItem(shader='shaded', color=(1, 0, 0, 0.6), glOptions='translucent')
        self.view_fov.addItem(self.items_fov['fan_B_missing'])
        
        # Missing Fan Edge (Wireframe for high contrast)
        self.items_fov['fan_B_missing_edge'] = gl.GLLinePlotItem(width=4, color=(1, 1, 0, 1), mode='lines') # Yellow edges
        self.view_fov.addItem(self.items_fov['fan_B_missing_edge'])
        
        # ISO Marker
        self.items_fov['iso'] = gl.GLScatterPlotItem(pos=np.array([[0,0,0]]), color=(1,1,1,1), size=10, pxMode=True)
        self.view_fov.addItem(self.items_fov['iso'])
        
        # Text Label for FOV View
        self.text_fov = gl.GLTextItem(pos=(0, 300, 300), text="FOV Analysis", font=QFont('Arial', 14, QFont.Bold), color=(1,1,1,1))
        self.view_fov.addItem(self.text_fov)

    def create_cylinder_mesh_data(self, radius, height, start_angle, end_angle, rows=10, cols=20):
        # Create mesh data for a partial cylinder surface
        theta = np.linspace(start_angle, end_angle, cols)
        z = np.linspace(-height/2, height/2, rows)
        theta_grid, z_grid = np.meshgrid(theta, z)
        
        x = radius * np.cos(theta_grid)
        y = radius * np.sin(theta_grid)
        
        # Vertices: (rows * cols, 3)
        verts = np.column_stack([x.flatten(), y.flatten(), z_grid.flatten()])
        
        # Faces
        faces = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                # Indices in the flattened array
                i0 = r * cols + c
                i1 = r * cols + (c + 1)
                i2 = (r + 1) * cols + c
                i3 = (r + 1) * cols + (c + 1)
                
                faces.append([i0, i1, i2]) # Triangle 1
                faces.append([i1, i3, i2]) # Triangle 2
                
        return verts, np.array(faces)

    def create_fan_sector_mesh(self, origin, corners):
        # Create a mesh for the fan sector (triangle fan from origin to detector corners)
        # origin: (x,y,z)
        # corners: list of 4 points [(x,y,z), ...]
        
        # We need to triangulate the pyramid/wedge shape
        # Base is the detector rectangle (corners 0-1-2-3)
        # Apex is the origin
        
        # Vertices: Origin + 4 Corners
        verts = np.vstack([np.array(origin), np.array(corners)])
        
        # Faces:
        # Side 1: Origin, C0, C1
        # Side 2: Origin, C1, C2
        # Side 3: Origin, C2, C3
        # Side 4: Origin, C3, C0
        # (Optional) Base: C0, C1, C2 and C0, C2, C3 (Usually not needed if we want to see inside)
        
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1]
        ])
        
        return verts, faces

    def update_simulation(self):
        # Get values
        params = {k: s.value() for k, s in self.sliders.items()}
        is_asymmetric = self.asymmetric_cb.isChecked()
        params['is_asymmetric'] = is_asymmetric
        
        # Calculate
        results, fan_A, fan_B = calculate_geometry(
            params['alpha'], params['RA'], params['RB'], params['FDD'],
            params['SFOV_A'], params['SFOV_B'], params['Z_coverage'], params['rotation_time'],
            is_asymmetric=is_asymmetric
        )
        
        if "error" in results:
            self.result_label.setText(f"<font color='red'>Error: {results['error']}</font>")
            return

        # --- Update 3D Objects ---
        
        # 1. Tubes
        FA = results["R_A_coord"]
        FB = results["R_B_coord"]
        self.items['tubes'].setData(pos=np.array([FA, FB]), color=np.array([[1,0,0,1], [0,0,1,1]]))
        
        # 2. Detectors
        # Det A: Center angle = pi - alpha/2. Span = fan_angle_A
        # System A
        angle_iso_A = np.pi - results['alpha_rad'] / 2
        beta_A_rad = np.deg2rad(fan_A) / 2
        verts_A, faces_A = self.create_cylinder_mesh_data(
            params['FDD'], results['H_det_A'], 
            angle_iso_A - beta_A_rad, angle_iso_A + beta_A_rad
        )
        # Shift mesh
        verts_A[:, 0] += FA[0]
        verts_A[:, 1] += FA[1]
        verts_A[:, 2] += FA[2]
        self.items['det_A'].setMeshData(vertexes=verts_A, faces=faces_A, color=(0.8, 0, 0, 0.8))
        
        # System B
        angle_iso_B = np.pi + results['alpha_rad'] / 2
        
        if is_asymmetric:
            # Handle Asymmetric Visualization
            beta_B_orig = results['beta_B_orig']
            ext_angle = (120.0 / params['FDD']) # Extension angle in radians
            
            # 1. Real Part (Original + Right Extension)
            # Span: [Center - Beta_Orig, Center + Beta_Orig + Extension]
            verts_B_real, faces_B_real = self.create_cylinder_mesh_data(
                params['FDD'], results['H_det_B'],
                angle_iso_B - beta_B_orig, angle_iso_B + beta_B_orig + ext_angle
            )
            verts_B_real[:, 0] += FB[0]
            verts_B_real[:, 1] += FB[1]
            verts_B_real[:, 2] += FB[2]
            self.items['det_B'].setMeshData(vertexes=verts_B_real, faces=faces_B_real, color=(0, 0, 0.8, 0.8)) # Solid Blue
            
            # 2. Virtual Part (Left Extension)
            if 'det_B_virtual' not in self.items:
                self.items['det_B_virtual'] = gl.GLMeshItem(shader='shaded', color=(0.5, 0.5, 1, 0.3), glOptions='translucent')
                self.view.addItem(self.items['det_B_virtual'])
                
            verts_B_virt, faces_B_virt = self.create_cylinder_mesh_data(
                params['FDD'], results['H_det_B'],
                angle_iso_B - beta_B_orig - ext_angle, angle_iso_B - beta_B_orig
            )
            verts_B_virt[:, 0] += FB[0]
            verts_B_virt[:, 1] += FB[1]
            verts_B_virt[:, 2] += FB[2]
            self.items['det_B_virtual'].setMeshData(vertexes=verts_B_virt, faces=faces_B_virt)
            self.items['det_B_virtual'].setVisible(True)
            
            # Update Corners for Edges/Lines
            # Real Corners (for lines and edges)
            start_angle = angle_iso_B - beta_B_orig
            end_angle = angle_iso_B + beta_B_orig + ext_angle
            
            corners_B = [
                (params['FDD'] * np.cos(start_angle), params['FDD'] * np.sin(start_angle), -results['H_det_B']/2),
                (params['FDD'] * np.cos(end_angle), params['FDD'] * np.sin(end_angle), -results['H_det_B']/2),
                (params['FDD'] * np.cos(end_angle), params['FDD'] * np.sin(end_angle), results['H_det_B']/2),
                (params['FDD'] * np.cos(start_angle), params['FDD'] * np.sin(start_angle), results['H_det_B']/2)
            ]
            
            # Fan covers Full Effective Range
            # fan_start = angle_iso_B - beta_B_orig - ext_angle
            # fan_end = angle_iso_B + beta_B_orig + ext_angle
            
            # Split Fan into Real and Virtual
            # Real: [Center - Beta_Orig, Center + Beta_Orig + Extension] (Original + Right Ext)
            fan_real_start = angle_iso_B - beta_B_orig
            fan_real_end = angle_iso_B + beta_B_orig + ext_angle
            
            fan_corners_B = [
                (params['FDD'] * np.cos(fan_real_start), params['FDD'] * np.sin(fan_real_start), -results['H_det_B']/2),
                (params['FDD'] * np.cos(fan_real_end), params['FDD'] * np.sin(fan_real_end), -results['H_det_B']/2),
                (params['FDD'] * np.cos(fan_real_end), params['FDD'] * np.sin(fan_real_end), results['H_det_B']/2),
                (params['FDD'] * np.cos(fan_real_start), params['FDD'] * np.sin(fan_real_start), results['H_det_B']/2)
            ]
            
            # Virtual: [Center - Beta_Orig - Extension, Center - Beta_Orig] (Left Ext)
            fan_virt_start = angle_iso_B - beta_B_orig - ext_angle
            fan_virt_end = angle_iso_B - beta_B_orig
            
            fan_corners_B_virtual = [
                (params['FDD'] * np.cos(fan_virt_start), params['FDD'] * np.sin(fan_virt_start), -results['H_det_B']/2),
                (params['FDD'] * np.cos(fan_virt_end), params['FDD'] * np.sin(fan_virt_end), -results['H_det_B']/2),
                (params['FDD'] * np.cos(fan_virt_end), params['FDD'] * np.sin(fan_virt_end), results['H_det_B']/2),
                (params['FDD'] * np.cos(fan_virt_start), params['FDD'] * np.sin(fan_virt_start), results['H_det_B']/2)
            ]
            
        else:
            if 'det_B_virtual' in self.items:
                self.items['det_B_virtual'].setVisible(False)
                
            beta_B_rad = np.deg2rad(fan_B) / 2
            verts_B, faces_B = self.create_cylinder_mesh_data(
                params['FDD'], results['H_det_B'], 
                angle_iso_B - beta_B_rad, angle_iso_B + beta_B_rad
            )
            verts_B[:, 0] += FB[0]
            verts_B[:, 1] += FB[1]
            verts_B[:, 2] += FB[2]
            self.items['det_B'].setMeshData(vertexes=verts_B, faces=faces_B, color=(0, 0, 0.8, 0.8))
            
            corners_B = [
                (params['FDD'] * np.cos(angle_iso_B - beta_B_rad), params['FDD'] * np.sin(angle_iso_B - beta_B_rad), -results['H_det_B']/2),
                (params['FDD'] * np.cos(angle_iso_B + beta_B_rad), params['FDD'] * np.sin(angle_iso_B + beta_B_rad), -results['H_det_B']/2),
                (params['FDD'] * np.cos(angle_iso_B + beta_B_rad), params['FDD'] * np.sin(angle_iso_B + beta_B_rad), results['H_det_B']/2),
                (params['FDD'] * np.cos(angle_iso_B - beta_B_rad), params['FDD'] * np.sin(angle_iso_B - beta_B_rad), results['H_det_B']/2)
            ]
            fan_corners_B = corners_B
            fan_corners_B_virtual = None

        # Detector Edges
        # A
        corners_A = [
            (params['FDD'] * np.cos(angle_iso_A - beta_A_rad), params['FDD'] * np.sin(angle_iso_A - beta_A_rad), -results['H_det_A']/2),
            (params['FDD'] * np.cos(angle_iso_A + beta_A_rad), params['FDD'] * np.sin(angle_iso_A + beta_A_rad), -results['H_det_A']/2),
            (params['FDD'] * np.cos(angle_iso_A + beta_A_rad), params['FDD'] * np.sin(angle_iso_A + beta_A_rad), results['H_det_A']/2),
            (params['FDD'] * np.cos(angle_iso_A - beta_A_rad), params['FDD'] * np.sin(angle_iso_A - beta_A_rad), results['H_det_A']/2)
        ]
        # Shift corners relative to FA
        shifted_corners_A = [np.array(c) + np.array(FA) for c in corners_A]
        edge_pts_A = [
            shifted_corners_A[0], shifted_corners_A[1], shifted_corners_A[2], shifted_corners_A[3], shifted_corners_A[0]
        ]
        self.items['det_A_edge'].setData(pos=np.array(edge_pts_A))
        
        # B Edge (Real Part only)
        shifted_corners_B = [np.array(c) + np.array(FB) for c in corners_B]
        edge_pts_B = [
            shifted_corners_B[0], shifted_corners_B[1], shifted_corners_B[2], shifted_corners_B[3], shifted_corners_B[0]
        ]
        self.items['det_B_edge'].setData(pos=np.array(edge_pts_B))

        # 3. Cone Lines (Edges)
        # A
        pts_lines_A = []
        for c in corners_A:
            pt = np.array(c) + np.array(FA)
            pts_lines_A.append(FA)
            pts_lines_A.append(pt)
        self.items['lines_A'].setData(pos=np.array(pts_lines_A))
        
        # B
        pts_lines_B = []
        for c in corners_B:
            pt = np.array(c) + np.array(FB)
            pts_lines_B.append(FB)
            pts_lines_B.append(pt)
        self.items['lines_B'].setData(pos=np.array(pts_lines_B))
        
        # 4. SFOV Sphere A
        radius_A = params['RA'] * np.sin(beta_A_rad)
        md = gl.MeshData.sphere(rows=20, cols=40, radius=radius_A) # Smoother sphere
        self.items['sfov'].setMeshData(meshdata=md)
        
        # Equator A
        theta_circ = np.linspace(0, 2*np.pi, 60)
        eq_x = radius_A * np.cos(theta_circ)
        eq_y = radius_A * np.sin(theta_circ)
        eq_z = np.zeros_like(eq_x)
        self.items['sfov_equator'].setData(pos=np.column_stack((eq_x, eq_y, eq_z)))
        
        # 4.1 SFOV Sphere B (Calculate early for reuse)
        if is_asymmetric:
             # Use calculated effective SFOV
             radius_B = results['SFOV_B_effective'] / 2
        else:
             beta_B_rad = np.deg2rad(fan_B) / 2
             radius_B = params['RB'] * np.sin(beta_B_rad)
             
        md_B = gl.MeshData.sphere(rows=20, cols=40, radius=radius_B)
        self.items['sfov_B'].setMeshData(meshdata=md_B)
        
        # Equator B
        eq_x_B = radius_B * np.cos(theta_circ)
        eq_y_B = radius_B * np.sin(theta_circ)
        self.items['sfov_B_equator'].setData(pos=np.column_stack((eq_x_B, eq_y_B, eq_z)))

        # 3.1 Fan Sectors (Solid)
        # A
        verts_fan_A, faces_fan_A = self.create_fan_sector_mesh(FA, [np.array(c) + np.array(FA) for c in corners_A])
        self.items['fan_A'].setMeshData(vertexes=verts_fan_A, faces=faces_fan_A, color=(0.8, 0, 0, 0.2)) # Light red transparent
        
        # B (Uses fan_corners_B - Real part in asymmetric mode)
        verts_fan_B, faces_fan_B = self.create_fan_sector_mesh(FB, [np.array(c) + np.array(FB) for c in fan_corners_B])
        self.items['fan_B'].setMeshData(vertexes=verts_fan_B, faces=faces_fan_B, color=(0, 0, 0.8, 0.3)) # Light blue transparent
        
        # B Virtual
        if fan_corners_B_virtual:
            verts_fan_B_virt, faces_fan_B_virt = self.create_fan_sector_mesh(FB, [np.array(c) + np.array(FB) for c in fan_corners_B_virtual])
            self.items['fan_B_virtual'].setMeshData(vertexes=verts_fan_B_virt, faces=faces_fan_B_virt)
            self.items['fan_B_virtual'].setVisible(True)
            
            # --- Update FOV View Objects ---
            # Use same meshes but in different view context
            # We want to show them centered at ISO or just show the B geometry isolated?
            # User said "associated 3D graphics", "synchronously draw".
            # Let's show B system isolated but in same coordinate system relative to ISO
            
            # 1. SFOV B Sphere
            # Reuse md_B
            self.items_fov['sfov_B'].setMeshData(meshdata=md_B)
            
            # 2. Real Fan B
            self.items_fov['fan_B_real'].setMeshData(vertexes=verts_fan_B, faces=faces_fan_B)
            
            # 3. Missing Fan B
            self.items_fov['fan_B_missing'].setMeshData(vertexes=verts_fan_B_virt, faces=faces_fan_B_virt)
            self.items_fov['fan_B_missing'].setVisible(True)
            self.items_fov['fan_B_missing'].setColor((1, 0, 0, 0.7)) # High contrast Red
            
            # 3.1 Missing Fan Edges
            # Extract lines from corners: Tube -> Corner
            # fan_corners_B_virtual has 4 points at detector
            # Origin is FB
            pts_edge = []
            for c in fan_corners_B_virtual:
                 c_global = np.array(c) + np.array(FB)
                 pts_edge.append(FB)
                 pts_edge.append(c_global)
            # Also connect the detector rect
            c_globals = [np.array(c) + np.array(FB) for c in fan_corners_B_virtual]
            pts_edge.append(c_globals[0]); pts_edge.append(c_globals[1])
            pts_edge.append(c_globals[1]); pts_edge.append(c_globals[2])
            pts_edge.append(c_globals[2]); pts_edge.append(c_globals[3])
            pts_edge.append(c_globals[3]); pts_edge.append(c_globals[0])
            
            self.items_fov['fan_B_missing_edge'].setData(pos=np.array(pts_edge))
            self.items_fov['fan_B_missing_edge'].setVisible(True)
            
            # Update Text
            self.text_fov.setData(text=f"Missing FOV Sector: {results['SFOV_B_effective']:.1f} mm")
            
            # Show Affected Shell (Outer Ring)
            # Inner Sphere is md_B (SFOV B Inner)
            # Wait, md_B above was calculated using 'radius_B'.
            # In asymmetric mode, radius_B was effective/2.
            # So md_B is the Large Sphere.
            
            # We want: 
            # 1. Inner Sphere (Safe) - Green/Cyan
            # 2. Outer Sphere (Affected) - Orange
            
            # Recalculate meshes for FOV view specifically
            rad_inner = results['SFOV_B_inner'] / 2
            rad_outer = results['SFOV_B_effective'] / 2
            
            md_inner = gl.MeshData.sphere(rows=20, cols=40, radius=rad_inner)
            # For outer shell, maybe just a slightly larger sphere? 
            # To see it as a shell, we rely on transparency.
            md_outer = gl.MeshData.sphere(rows=20, cols=40, radius=rad_outer)
            
            self.items_fov['sfov_B'].setMeshData(meshdata=md_inner) # Inner Safe
            self.items_fov['sfov_B'].setColor((0, 1, 1, 0.3)) # Cyan Safe
            
            self.items_fov['sfov_B_affected'].setMeshData(meshdata=md_outer) # Outer Affected
            self.items_fov['sfov_B_affected'].setVisible(True)
            
        else:
            self.items['fan_B_virtual'].setVisible(False)
            
            # If not asymmetric, show normal fan in FOV view?
            # Or hide missing part
            self.items_fov['fan_B_missing'].setVisible(False)
            self.items_fov['fan_B_missing_edge'].setVisible(False)
            self.items_fov['sfov_B_affected'].setVisible(False)
            
            # Update Real Fan
            self.items_fov['fan_B_real'].setMeshData(vertexes=verts_fan_B, faces=faces_fan_B)
            
            # Update SFOV
            self.items_fov['sfov_B'].setMeshData(meshdata=md_B)
            self.items_fov['sfov_B'].setColor((0, 1, 1, 0.1))
            
            self.text_fov.setData(text="Standard Symmetric FOV")
        
        # (Moved up) 4. SFOV Sphere A
        
        # (Moved up) 4.1 SFOV Sphere B
        
        # 5. Safety Lines
        # safe_pts = []
        # safe_pts.append(results['closest_pt_A_TubeB']); safe_pts.append(FB)
        # safe_pts.append(results['closest_pt_B_TubeA']); safe_pts.append(FA)
        # if results['min_dist_Det_Det'] < 200:
        #      safe_pts.append(results['closest_pt_DetA_DetB']); safe_pts.append(results['closest_pt_DetB_DetA'])
        
        # self.items['safe_lines'].setData(pos=np.array(safe_pts))
        
        # Text for Safety Lines
        self.text_items['A_TubeB'].setData(text="")
        self.text_items['B_TubeA'].setData(text="")
        self.text_items['Det_Det'].setData(text="")
        # self.text_items['A_TubeB'].setData(pos=np.array(results['closest_pt_A_TubeB']), text=f"{results['min_dist_A_TubeB']:.0f}")
        # self.text_items['B_TubeA'].setData(pos=np.array(results['closest_pt_B_TubeA']), text=f"{results['min_dist_B_TubeA']:.0f}")
        # if results['min_dist_Det_Det'] < 200:
        #      mid_det = (np.array(results['closest_pt_DetA_DetB']) + np.array(results['closest_pt_DetB_DetA'])) / 2
        #      self.text_items['Det_Det'].setData(pos=mid_det, text=f"{results['min_dist_Det_Det']:.0f}")
        # else:
        #      self.text_items['Det_Det'].setData(text="")

        # 6. Force Vector
        F_vec = results['F_total_vector']
        scale = 0.1
        vec_end = np.array([F_vec[0]*scale, F_vec[1]*scale, 0])
        self.items['force_vec'].setData(pos=np.array([[0,0,0], vec_end]))
        self.items['force_head'].setData(pos=np.array([vec_end]))
        
        # 7. Angle Arc
        # Draw arc from -alpha/2 to alpha/2 at radius min(RA, RB)*0.4
        r_arc = min(params['RA'], params['RB']) * 0.4
        alpha_rad = results['alpha_rad']
        theta_arc = np.linspace(-alpha_rad/2, alpha_rad/2, 30)
        arc_x = r_arc * np.cos(theta_arc)
        arc_y = r_arc * np.sin(theta_arc)
        arc_z = np.zeros_like(arc_x)
        self.items['angle_arc'].setData(pos=np.column_stack((arc_x, arc_y, arc_z)))
        
        # Angle Text
        mid_arc = (arc_x[15], arc_y[15], 0)
        self.text_items['Angle'].setData(pos=np.array(mid_arc), text=f"{params['alpha']}°")
        
        # --- Update Results Text ---
        safe_color = "green" if results['is_safe'] else "red"
        arc_color = "green" if results['is_arc_ok'] else "red"
        
        impact_txt = ""
        if is_asymmetric:
            impact_txt = f"""
            <hr>
            <b><font color='red'>非对称模式影响 (Negative Impact):</font></b><br>
            1. <b>受影响FOV体积:</b> {results['affected_volume_cm3']:.1f} cm³ (Outer Shell)<br>
            2. <b>有效 SFOV B:</b> {results['SFOV_B_effective']:.1f} mm (Inner: {results['SFOV_B_inner']:.1f} mm)<br>
            3. <b>散射干扰系数:</b> {results['scatter_coeff']:.2f} (因体积增加)<br>
            """
        
        txt = f"""
        <h3>计算结果</h3>
        <b>扇区角:</b> A: {results['fan_angle_A']:.2f}° | B: {results['fan_angle_B']:.2f}°<br>
        <b>探测器高度:</b> A: {results['H_det_A']:.1f} | B: {results['H_det_B']:.1f} mm<br>
        <hr>
        <b>最小安全距离:</b> {results['min_dist']:.1f} mm <font color='{safe_color}'><b>({'安全' if results['is_safe'] else '碰撞预警'})</b></font><br>
        <b>DetA-DetB:</b> {results['min_dist_Det_Det']:.1f} mm<br>
        <b>弧长差:</b> {results['arc_diff']:.1f} mm <font color='{arc_color}'>({'OK' if results['is_arc_ok'] else 'Diff < 120'})</font><br>
        <b>物理时间分辨:</b> <font color='blue'>{results['temporal_resolution']*1000:.0f} ms</font><br>
        <b>散射干扰系数:</b> <font color='orange'>{results['scatter_coeff']:.2f}</font><br>
        {impact_txt}
        <hr>
        <b>离心力 (G-Force):</b><br>
        Tube A/B: {results['g_force_TubeA']:.1f} G / {results['g_force_TubeB']:.1f} G<br>
        Det A/B: {results['g_force_DetA']:.1f} G / {results['g_force_DetB']:.1f} G<br>
        <b>系统固定压力 (双球管合力):</b> <font color='purple' size='4'>{results['F_total_mag']:.0f} N ({results['g_force_total']:.1f} G)</font>
        """
        self.result_label.setText(txt)

        # --- Update HUD Text ---
        hud_txt = f"""
        <b>System Geometry Parameters</b><br>
        --------------------------------<br>
        Angle (α)   : {params['alpha']}°<br>
        F-ISO (RA)  : {params['RA']} mm<br>
        F-ISO (RB)  : {params['RB']} mm<br>
        FDD         : {params['FDD']} mm<br>
        SFOV (A)    : {params['SFOV_A']} mm<br>
        SFOV (B)    : {params['SFOV_B']} mm<br>
        Z-Coverage  : {params['Z_coverage']} mm<br>
        Rotation T  : {params['rotation_time']} s<br>
        --------------------------------<br>
        System Load : {results['F_total_mag']:.0f} N
        """
        if is_asymmetric:
             hud_txt += "<br><font color='red'>[Asymmetric Mode ON]</font>"
             
        self.hud_label.setText(hud_txt)
        self.hud_label.adjustSize()

        # Update Reconstruction
        self.recon_widget.update_data(results, params)

if __name__ == '__main__':
    try:
        print("Starting application...")
        app = QApplication(sys.argv)
        print("QApplication created.")
        
        win = MainWindow()
        print("MainWindow created.")
        
        win.show()
        print("Window shown. Entering event loop...")
        
        exit_code = app.exec_()
        print(f"Event loop exited with code: {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        print("CRITICAL ERROR:")
        import traceback
        traceback.print_exc()
        
        # Try to show a GUI error message if possible
        try:
            from PyQt5.QtWidgets import QMessageBox
            # Check if app exists, if not create one
            if not QApplication.instance():
                app = QApplication(sys.argv)
            
            error_msg = f"An unexpected error occurred:\n{str(e)}\n\nSee log for details."
            QMessageBox.critical(None, "Critical Error", error_msg)
        except:
            pass
        
        # input("Press Enter to close...") # Removed to avoid lost sys.stdin error