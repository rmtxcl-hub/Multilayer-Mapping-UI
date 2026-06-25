from pathlib import Path
import sys, traceback, json, numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QObject, QStringListModel, QSortFilterProxyModel, QFileSystemWatcher, QTimer, QEvent, qInstallMessageHandler
from PySide6.QtGui import QDoubleValidator, QIntValidator, QFont
from functools import partial
from PySide6.QtWidgets import *
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patheffects as pe
from nkwrap import NK, _vm_parse
from TMM import TMM
import map_collection as mc
import map_collection2 as mc2
import contour
import submask
import simplepeak

BASE = Path(__file__).resolve().parent
DB = BASE / 'data_f.sqlite'
TXT_DIR = BASE / 'txt'
TAGS = BASE / 'material_keys_tagged.txt'
SETTINGS_JSON = BASE / 'ui_settings.json'
TXT_DIR.mkdir(exist_ok=True)

MAP_CMAPS = ['inferno', 'magma', 'plasma', 'viridis', 'cividis', 'turbo', 'gray', 'coolwarm', 'seismic', 'RdBu_r']

MAP_SCALE_OPTIONS = [
    'Normalized 0-1',
    'Normalized -1~0~1',
    'Normalized |value|',
    'Auto data',
]

DEFAULT_MAP_CMAP = 'turbo'
DEFAULT_MAP_SCALE = 'Normalized 0-1'

DEFAULT_TH0 = '0'
DEFAULT_TH1 = '85'

FIELD_MAP_TYPES = [
    'Field intensity map',
]
FIELD_COMPONENT_OPTIONS = list(mc2.FIELD_COMPONENTS)
FIELD_KIND_OPTIONS = ['E', 'B']
FIELD_SLICE_OPTIONS = list(mc2.FIELD_SLICES)
FIELD_DEPTH_REGION_OPTIONS = list(mc2.FIELD_DEPTH_REGIONS)
FIELD_SCALE_OPTIONS = ['None', 'Incident', 'Maximum', 'Log incident']
SETTINGS_MIN_W = 1180
SETTINGS_STANDARD_H = 170
SETTINGS_FIELD_H = 220

MAP_TYPES = [
    'Emissivity',
    'Reflectivity',
    'Band-averaged Emissivity',
    'Planck-weighted emissive power',
    'Hemispherical emissivity',
    'Polarization contrast',
    'Ellipsometry',
    *FIELD_MAP_TYPES,
]

CONTRAST_OPTIONS = [
    '(εₚ-εₛ)/(εₚ+εₛ)',
    'εₚ-εₛ',
    '(Rₚ-Rₛ)/(Rₚ+Rₛ)',
    'Rₚ-Rₛ',
]

ELLIP_OPTIONS = [
    'Ψ angle (deg)',
    'Δ phase (deg)',
    '|ρ| amplitude',
]

MASK_CFG = dict(thr=None, thr_hi='adaptive', thr_hi_alpha=0.7, thr_hi_base='median', thr_lo=None, thr_lo_pct='auto', sig=1.6, close=4, single=None, peak=None)
SUBMASK_CFG = dict(close=2, fill=True)

DEFAULT_UI_SETTINGS = {
    'tol_im': 0.0,
    'tol_re': 0.0,
    'clip_im_delta': 60.0,
    'cover_n': 1.0,
    'cover_k': 0.0,
    'save_aspect_w': 16.0,
    'save_aspect_h': 9.0,
    'map_cmap': DEFAULT_MAP_CMAP,
    'map_scale': DEFAULT_MAP_SCALE,
}
LAYER_CARD_W = 215
LAYER_CARD_H = 158
LAYER_GAP_W = 30
LAYER_BTN = 22
MAPPING_LAYER_SLIDER_REVISION = 'mapping_layer_drag_ghost_inplace_v10_20260625'

def ensure_checkbox_icons():
    u = BASE / '_cb_unchecked.svg'
    c = BASE / '_cb_checked.svg'
    if not u.exists():
        u.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16"><rect x="1" y="1" width="14" height="14" rx="2" ry="2" fill="white" stroke="black" stroke-width="1.4"/></svg>', encoding='utf-8')
    if not c.exists():
        c.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16"><rect x="1" y="1" width="14" height="14" rx="2" ry="2" fill="white" stroke="black" stroke-width="1.4"/><path d="M4 8.4l2.2 2.4L12 5.5" fill="none" stroke="black" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>', encoding='utf-8')
    return u.as_posix(), c.as_posix()


def load_ui_settings():
    d = dict(DEFAULT_UI_SETTINGS)
    try:
        if SETTINGS_JSON.exists():
            j = json.loads(SETTINGS_JSON.read_text(encoding='utf-8'))
            if isinstance(j, dict):
                d.update(j)
    except Exception:
        pass

    if d.get('map_scale') == 'Fixed 0-1':
        d['map_scale'] = 'Normalized 0-1'

    if d.get('map_cmap') not in MAP_CMAPS or d.get('map_cmap') == 'inferno':
        d['map_cmap'] = DEFAULT_MAP_CMAP

    return d

def save_ui_settings(vals):
    try:
        SETTINGS_JSON.write_text(json.dumps(vals, indent=2), encoding='utf-8')
    except Exception:
        pass

def qt_message_handler(mode, context, message):
    msg = str(message)
    if msg.startswith('QFont::setPointSize: Point size <= 0'):
        return
    sys.stderr.write(msg + '\n')

def normalize_qt_app_font(default_pt=9.0):
    app = QApplication.instance()
    if app is None:
        return
    f = QFont(app.font())
    if f.pointSize() > 0 or f.pointSizeF() > 0:
        return
    px = f.pixelSize()
    if px > 0:
        screen = app.primaryScreen()
        dpi = screen.logicalDotsPerInchY() if screen is not None else 96.0
        f.setPointSizeF(max(1.0, px * 72.0 / max(dpi, 1.0)))
    else:
        f.setPointSizeF(float(default_pt))
    app.setFont(f)
    
def fixed_blank(width, height=40):
    w = QWidget()
    w.setFixedSize(width, height)
    w.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return w

def fixed_label(text, width, height=40):
    w = QLabel(text)
    w.setFixedSize(width, height)
    w.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
    w.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return w

def fixed_stack(width, height=40):
    w = QStackedWidget()
    w.setFixedSize(width, height)
    w.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return w

def read_tagged_keys(path):
    if not path.exists():
        return []
    out = []
    for s in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        s = s.strip()
        if not s or s.startswith('#'):
            continue
        out.append(s.split('\t')[0].strip())
    return out

def key_info(nk, key):
    key = (key or '').strip()
    if not key:
        return {'valid': False, 'input': key, 'base': None, 'kind': None}
    if _vm_parse(key) is not None:
        return {'valid': True, 'input': key, 'base': key, 'kind': 'both'}
    try:
        nk.meta(key)
        return {'valid': True, 'input': key, 'base': key, 'kind': 'both'}
    except Exception:
        pass
    lo = key.lower()
    for suf, kind in (('-n', 'n'), ('_n', 'n'), ('-k', 'k'), ('_k', 'k')):
        if lo.endswith(suf):
            base = key[:-len(suf)]
            if _vm_parse(base) is not None:
                return {'valid': True, 'input': key, 'base': base, 'kind': kind}
            try:
                nk.meta(base)
                return {'valid': True, 'input': key, 'base': base, 'kind': kind}
            except Exception:
                return {'valid': False, 'input': key, 'base': None, 'kind': kind}
    return {'valid': False, 'input': key, 'base': None, 'kind': None}

def load_key_data(nk, key):
    info = key_info(nk, key)
    if not info['valid']:
        raise ValueError(f'Invalid key: {key}')
    if _vm_parse(info['base']) is not None:
        raise ValueError('Virtual material has no catalog dataset.')
    wn, nv, wk, kv = nk._load(info['base'])
    wn = np.asarray(wn); nv = np.asarray(nv); wk = np.asarray(wk); kv = np.asarray(kv)
    if info['kind'] == 'n':
        wk = np.asarray([]); kv = np.asarray([])
    elif info['kind'] == 'k':
        wn = np.asarray([]); nv = np.asarray([])
    return wn, nv, wk, kv

def build_catalog():
    nk = NK(path=str(DB), additional_txt_files={'paths': [], 'dir': str(TXT_DIR)})
    tagged = read_tagged_keys(TAGS)

    def rep_for_base(base):
        try:
            wn, _, wk, _ = nk._load(base)
        except Exception:
            return None
        has_n = np.asarray(wn).size > 0
        has_k = np.asarray(wk).size > 0
        if has_n and has_k:
            return base
        if has_n:
            for cand in (f'{base}-n', f'{base}_n'):
                if cand in tagged:
                    return cand
            return f'{base}-n'
        if has_k:
            for cand in (f'{base}-k', f'{base}_k'):
                if cand in tagged:
                    return cand
            return f'{base}-k'
        return None

    canonical = []
    seen = set()

    for key in tagged:
        info = key_info(nk, key)
        if not info['valid']:
            continue
        base = info['base']
        rep = base if info['kind'] == 'both' else (f'{base}-n' if info['kind'] == 'n' else f'{base}-k')
        if rep not in seen:
            seen.add(rep)
            canonical.append(key)

    for base in nk.keys():
        rep = rep_for_base(base)
        if rep and rep not in seen:
            seen.add(rep)
            canonical.append(rep)

    return nk, canonical

class FilterProxy(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._needle = ''
        self.setFilterCaseSensitivity(Qt.CaseInsensitive)
    def setNeedle(self, s):
        self._needle = (s or '').strip().lower()
        self.invalidateFilter()
    def filterAcceptsRow(self, row, parent):
        if not self._needle:
            return True
        s = self.sourceModel().data(self.sourceModel().index(row, 0, parent), Qt.DisplayRole)
        return self._needle in str(s).lower()

class ArrowComboBox(QComboBox):
    def __init__(self, items=None, parent=None, editable=False):
        super().__init__(parent)
        self.setEditable(editable)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(14)
        self.setMinimumHeight(40)
        self.setMaxVisibleItems(18)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._btn = QToolButton(self)
        self._btn.setCursor(Qt.ArrowCursor)
        self._btn.setFocusPolicy(Qt.NoFocus)
        self._btn.setText('▼')
        self._btn.clicked.connect(self.showPopup)
        self._btn.setStyleSheet('QToolButton { background: transparent; border: none; color: #111; font-size: 10pt; font-weight: 600; padding: 0px; margin: 0px; }')
        if editable:
            self._src = QStringListModel(items or [], self)
            self._proxy = FilterProxy(self)
            self._proxy.setSourceModel(self._src)
            self._comp = QCompleter(self._proxy, self)
            self._comp.setCaseSensitivity(Qt.CaseInsensitive)
            self._comp.setFilterMode(Qt.MatchContains)
            self._comp.setCompletionMode(QCompleter.PopupCompletion)
            self.setCompleter(self._comp)
            self.lineEdit().textEdited.connect(self._proxy.setNeedle)
            self.lineEdit().editingFinished.connect(self._snap)
            self.lineEdit().setStyleSheet('padding-right: 38px;')
        if items:
            self.set_items(items)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        frame = 1
        w = 34
        self._btn.setGeometry(self.width() - w - frame, frame, w, self.height() - 2 * frame)

    def set_items(self, items):
        cur = self.currentText().strip()
        self.blockSignals(True)
        self.clear()
        self.addItems(items)
        if hasattr(self, '_src'):
            self._src.setStringList(items)
        self.blockSignals(False)
        if cur:
            self.setCurrentText(cur)

    def _snap(self):
        if not hasattr(self, '_src'):
            return
        txt = self.currentText().strip()
        items = self._src.stringList()
        if txt in items:
            return
        matches = [k for k in items if txt.lower() in k.lower()]
        if matches:
            self.setCurrentText(matches[0])

class MaterialBox(ArrowComboBox):
    def __init__(self, items, parent=None):
        super().__init__(items=items, parent=parent, editable=True)

class LayerCard(QFrame):
    def __init__(self, idx, total, items):
        super().__init__()
        self.setObjectName('LayerCard')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedSize(LAYER_CARD_W, LAYER_CARD_H)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.idx = idx
        self.setCursor(Qt.OpenHandCursor)

        self.lab = QLabel()
        self.lab.setCursor(Qt.OpenHandCursor)

        self.del_btn = QToolButton()
        self.del_btn.setText('✕')
        self.del_btn.setFixedSize(LAYER_BTN, LAYER_BTN)
        self.del_btn.setFont(QFont('Arial', 8, QFont.Bold))
        self.del_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.del_btn.setCursor(Qt.PointingHandCursor)
        self.del_btn.setStyleSheet(
            'QToolButton { background:#fbfbfb; border:1px solid #999; border-radius:11px; padding:0px; margin:0px; }'
            'QToolButton:hover { background:#ffecec; border-color:#cc4444; }'
            'QToolButton:disabled { color:#aaa; border-color:#ccc; background:#f5f5f5; }'
        )

        self.key = MaterialBox(items)
        self.key.setFixedHeight(36)
        self.key.setCursor(Qt.ArrowCursor)

        self.thk = QLineEdit('0.20')
        self.thk.setValidator(QDoubleValidator(0.0, 1e9, 6, self))
        self.thk.setFixedSize(78, 32)
        self.thk.setCursor(Qt.IBeamCursor)

        self.unit_label = QLabel('µm')
        self.unit_label.setCursor(Qt.OpenHandCursor)

        self.mat_label = QLabel('Material key')
        self.mat_label.setCursor(Qt.OpenHandCursor)

        self.thk_label = QLabel('Thickness')
        self.thk_label.setCursor(Qt.OpenHandCursor)

        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(4)
        head.addWidget(self.lab, 1)
        head.addWidget(self.del_btn)
        v.addLayout(head)

        v.addWidget(self.mat_label)
        v.addWidget(self.key)

        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(5)
        h.addWidget(self.thk)
        h.addWidget(self.unit_label)
        h.addStretch(1)

        v.addWidget(self.thk_label)
        v.addLayout(h)

        self.rename(idx, total)

    def rename(self, idx, total):
        self.idx = idx
        if total == 1:
            t = '↔ Layer 1 (Top = Bottom)'
        elif idx == 0:
            t = '↔ Layer 1 (Top)'
        elif idx == total - 1:
            t = f'↔ Layer {idx + 1} (Bottom)'
        else:
            t = f'↔ Layer {idx + 1}'

        self.lab.setText(t)
        self.del_btn.setEnabled(total > 1)
        
class InsertLayerGap(QWidget):
    def __init__(self, height=LAYER_CARD_H, parent=None):
        super().__init__(parent)
        self.setFixedSize(LAYER_GAP_W, height)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.btn = QToolButton()
        self.btn.setText('+')
        self.btn.setFixedSize(24, 24)
        self.btn.setFont(QFont('Arial', 11, QFont.Bold))
        self.btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setStyleSheet(
            'QToolButton { background:#fbfbfb; border:1px solid #999; border-radius:12px; padding:0px; margin:0px; }'
            'QToolButton:hover { background:#edf7ff; border-color:#3377cc; }'
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addStretch(1)
        lay.addWidget(self.btn, 0, Qt.AlignHCenter)
        lay.addStretch(1)

class InfoCard(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setObjectName('InfoCard')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)
        self.title = QLabel(title)
        self.title.setObjectName('InfoCardTitle')
        self.value = QLabel('-')
        self.value.setObjectName('InfoCardValue')
        self.value.setWordWrap(True)
        self.value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.title)
        lay.addWidget(self.value, 1)
    def setText(self, text):
        self.value.setText(text)

class SettingsDialog(QDialog):
    def __init__(self, vals, parent=None):
        super().__init__(parent)
        self.setWindowTitle('TMM Settings')
        self.setModal(True)
        f = QFormLayout(self)
        self.tol_im = QLineEdit(str(vals.get('tol_im', 0.0)))
        self.tol_re = QLineEdit(str(vals.get('tol_re', 0.0)))
        self.clip = QLineEdit(str(vals.get('clip_im_delta', 60.0)))
        self.cover_n = QLineEdit(str(vals.get('cover_n', 1.0)))
        self.cover_k = QLineEdit(str(vals.get('cover_k', 0.0)))
        self.asp_w = QLineEdit(str(vals.get('save_aspect_w', 16.0)))
        self.asp_h = QLineEdit(str(vals.get('save_aspect_h', 9.0)))
        for w in (self.tol_im, self.tol_re, self.clip, self.cover_n, self.cover_k, self.asp_w, self.asp_h):
            w.setValidator(QDoubleValidator(bottom=-1e9, top=1e9, decimals=9, parent=self))
        f.addRow('tol_im', self.tol_im)
        f.addRow('tol_re', self.tol_re)
        f.addRow('clip_im_delta', self.clip)
        f.addRow('cover_n', self.cover_n)
        f.addRow('cover_k', self.cover_k)
        f.addRow('Save aspect width', self.asp_w)
        f.addRow('Save aspect height', self.asp_h)
        b = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        b.accepted.connect(self.accept)
        b.rejected.connect(self.reject)
        f.addRow(b)
    def values(self):
        aw = max(float(self.asp_w.text() or 16.0), 0.1)
        ah = max(float(self.asp_h.text() or 9.0), 0.1)
        return {'tol_im': float(self.tol_im.text() or 0.0), 'tol_re': float(self.tol_re.text() or 0.0), 'clip_im_delta': float(self.clip.text() or 60.0), 'cover_n': float(self.cover_n.text() or 1.0), 'cover_k': float(self.cover_k.text() or 0.0), 'save_aspect_w': aw, 'save_aspect_h': ah}

class Worker(QObject):
    done = Signal(object, object)
    def __init__(self, payload):
        super().__init__()
        self.payload = payload
    def run(self):
        try:
            p = self.payload
            tmm = TMM(tol_im=p['tol_im'], tol_re=p['tol_re'], clip_im_delta=p['clip_im_delta'])
            cfg = {
                'lam_um': f"{p['lam0']}-{p['lam1']}-{p['lamN']}",
                'theta_rad': np.deg2rad(np.linspace(p['th0'], p['th1'], p['thN'])),
                'pol': p['pol'],
                'n0': complex(p['cover_n'], p['cover_k']),
                'film_keys': p['films'],
                'sub_key': p['sub'],
                'dL': p['dL'],
                'cover_n': p['cover_n'],
                'cover_k': p['cover_k'],
                'mode': p['mode'],
                'band_lam0': p['band_lam0'],
                'band_lam1': p['band_lam1'],
                'band_lamN': p['band_lamN'],
                'temperature': p['temperature'],
                'contrast_metric': p['contrast_metric'],
                'ellip_output': p['ellip_output'],
                'hemi_thetaN': p['hemi_thetaN'],
                'planck_quantity': 'flux',
                'lamN': p['lamN'],
                'thN': p['thN'],
                'field_kind': p.get('field_kind', 'E'),
                'field_component': p.get('field_component', 'total'),
                'field_normalization': p.get('field_normalization', 'incident'),
                'field_slice': p.get('field_slice', 'lambda-theta'),
                'field_depth_region': p.get('field_depth_region', 'stack'),
                'field_layer': p.get('field_layer', 1),
                'field_z_frac': p.get('field_z_frac', 0.5),
                'field_z_points': p.get('field_z_points', 200),
                'field_points_per_layer': p.get('field_points_per_layer', 80),
                'field_fixed_theta_deg': p.get('field_fixed_theta_deg', 0.0),
                'field_fixed_lam_um': p.get('field_fixed_lam_um', 8.0),
            }

            if p['mode'] in FIELD_MAP_TYPES:
                md = mc2.calc(tmm, cfg, mode=p['mode'], nk=p['nk'])
            else:
                md = mc.calc(tmm, cfg, mode=p['mode'], nk=p['nk'], use_gpu=False)

            self.done.emit(md, None)
        except Exception:
            self.done.emit(None, traceback.format_exc())

class MapTab(QWidget):
    def __init__(self):
        super().__init__()
        self.nk, self.catalog = build_catalog()
        self.settings = load_ui_settings()
        self.last_md = None
        self._overlay_cache = None
        self._thread = None
        self._worker = None
        self._watcher = None
        self._layer_widgets = []
        self._gap_widgets = []
        self._slide = None
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)
        top = QHBoxLayout()
        self.layer_count = QSpinBox(); self.layer_count.setRange(1, 999); self.layer_count.setValue(1); self.layer_count.valueChanged.connect(self.rebuild_layers)
        self.refresh_btn = QPushButton('Refresh materials'); self.refresh_btn.clicked.connect(self.refresh_catalog)
        self.settings_btn = QToolButton(); self.settings_btn.setText('⚙'); self.settings_btn.clicked.connect(self.open_settings)
        top.addWidget(QLabel('Number of layers')); top.addWidget(self.layer_count); top.addSpacing(16); top.addWidget(self.refresh_btn); top.addStretch(1); top.addWidget(self.settings_btn)
        outer.addLayout(top)
        self.sub_box = MaterialBox(self.catalog)
        sub_row = QHBoxLayout(); sub_row.addWidget(QLabel('Substrate')); sub_row.addWidget(self.sub_box, 1); outer.addLayout(sub_row)
        self.layer_wrap = QWidget()
        self.layer_wrap.setObjectName('LayerWrap')
        self.layer_wrap.setMinimumHeight(LAYER_CARD_H)
        self.layer_wrap.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.layer_wrap.setAutoFillBackground(False)
        self.layer_scroll = QScrollArea()
        self.layer_scroll.setObjectName('LayerScroll')
        self.layer_scroll.setFrameShape(QFrame.NoFrame)
        self.layer_scroll.setWidgetResizable(False)
        self.layer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.layer_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.layer_scroll.viewport().setObjectName('LayerScrollViewport')
        self.layer_scroll.viewport().setAutoFillBackground(False)
        self.layer_scroll.setWidget(self.layer_wrap)
        self.layer_scroll.setMinimumHeight(165)
        self.layer_scroll.setMaximumHeight(180)
        self.layer_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        outer.addWidget(self.layer_scroll)
        box = QGroupBox()
        g = QGridLayout(box)
        g.setContentsMargins(10, 8, 10, 8)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)

        self.mode = ArrowComboBox(MAP_TYPES)
        self.pol = ArrowComboBox(['both', 's', 'p'])
        self.cmap = ArrowComboBox(MAP_CMAPS)
        self.scale = ArrowComboBox(MAP_SCALE_OPTIONS)
        self.contrast_metric = ArrowComboBox(CONTRAST_OPTIONS)
        self.ellip_output = ArrowComboBox(ELLIP_OPTIONS)

        self.field_kind = ArrowComboBox(FIELD_KIND_OPTIONS)
        self.field_component = ArrowComboBox(FIELD_COMPONENT_OPTIONS)
        self.field_slice = ArrowComboBox(FIELD_SLICE_OPTIONS)
        self.field_depth_region = ArrowComboBox(FIELD_DEPTH_REGION_OPTIONS)
        self.field_layer = QLineEdit('1')
        self.field_z_frac = QLineEdit('0.5')
        self.field_z_points = QLineEdit('200')
        self.field_points_per_layer = QLineEdit('80')
        self.field_fixed_theta = QLineEdit('0')
        self.field_fixed_lam = QLineEdit('8')

        self.cmap.setCurrentText(str(self.settings.get('map_cmap', DEFAULT_MAP_CMAP)))
        self.scale.setCurrentText(str(self.settings.get('map_scale', DEFAULT_MAP_SCALE)))

        self.lam0 = QLineEdit('3')
        self.lam1 = QLineEdit('14')
        self.lamN = QLineEdit('501')
        self.th0 = QLineEdit(DEFAULT_TH0)
        self.th1 = QLineEdit(DEFAULT_TH1)
        self._hemi_forced_angle = False
        self.thN = QLineEdit('301')
        self.tempK = QLineEdit('300')

        dv = QDoubleValidator(-1e9, 1e9, 6, self)
        iv = QIntValidator(2, 10**6, self)
        layer_iv = QIntValidator(1, 10**6, self)

        for w in (self.lam0, self.lam1, self.th0, self.th1, self.tempK, self.field_z_frac, self.field_fixed_theta, self.field_fixed_lam):
            w.setValidator(dv)

        for w in (self.lamN, self.thN, self.field_z_points, self.field_points_per_layer):
            w.setValidator(iv)

        self.field_layer.setValidator(layer_iv)

        LW = 72
        LW2 = 88
        CW = 210
        EW = 125

        self._ui_LW = LW
        self._ui_LW2 = LW2
        self._ui_CW = CW
        self._ui_EW = EW

        for w in (self.mode, self.pol, self.cmap, self.scale, self.contrast_metric, self.ellip_output, self.field_kind, self.field_component, self.field_slice, self.field_depth_region):
            w.setFixedWidth(CW)

        for w in (self.lam0, self.lam1, self.th0, self.th1, self.lamN, self.thN, self.tempK, self.field_layer, self.field_z_frac, self.field_z_points, self.field_points_per_layer, self.field_fixed_theta, self.field_fixed_lam):
            w.setFixedWidth(EW)

        self.lbl_map = fixed_label('Map', LW)
        self.lbl_pol = fixed_label('Pol.', LW)
        self.lbl_cmap = fixed_label('Colormap', LW2)
        self.lbl_scale = fixed_label('Scale', LW)
        self.lbl_lam0 = fixed_label('λ min', LW)
        self.lbl_lam1 = fixed_label('λ max', LW)
        self.lbl_lamN = fixed_label('λ pts', LW2)
        self.lbl_th0 = fixed_label('θ min', LW)
        self.lbl_th1 = fixed_label('θ max', LW)
        self.lbl_thN = fixed_label('θ pts', LW2)
        self.lbl_temp = fixed_label('T (K)', LW)
        self.lbl_contrast = fixed_label('Contrast', LW)
        self.lbl_ellip = fixed_label('Ellip.', LW)

        self.pol_label_stack = fixed_stack(LW)
        self.pol_box_stack = fixed_stack(CW)
        self.scale_label_stack = fixed_stack(LW)
        self.scale_box_stack = fixed_stack(CW)
        self.right_label_stack = fixed_stack(LW)
        self.right_box_stack = fixed_stack(CW)

        self.pol_label_stack.addWidget(self.lbl_pol)
        self.pol_label_stack.addWidget(fixed_blank(LW))
        self.pol_box_stack.addWidget(self.pol)
        self.pol_box_stack.addWidget(fixed_blank(CW))

        self.scale_label_stack.addWidget(self.lbl_scale)
        self.scale_label_stack.addWidget(fixed_blank(LW))
        self.scale_box_stack.addWidget(self.scale)
        self.scale_box_stack.addWidget(fixed_blank(CW))

        self.right_label_stack.addWidget(fixed_blank(LW))
        self.right_label_stack.addWidget(self.lbl_temp)
        self.right_label_stack.addWidget(self.lbl_contrast)
        self.right_label_stack.addWidget(self.lbl_ellip)

        self.right_box_stack.addWidget(fixed_blank(CW))
        self.right_box_stack.addWidget(self.tempK)
        self.right_box_stack.addWidget(self.contrast_metric)
        self.right_box_stack.addWidget(self.ellip_output)

        g.addWidget(self.lbl_map, 0, 0)
        g.addWidget(self.mode, 0, 1)
        g.addWidget(self.pol_label_stack, 0, 2)
        g.addWidget(self.pol_box_stack, 0, 3)
        g.addWidget(self.lbl_cmap, 0, 4)
        g.addWidget(self.cmap, 0, 5)
        g.addWidget(self.scale_label_stack, 0, 6)
        g.addWidget(self.scale_box_stack, 0, 7)

        g.addWidget(self.lbl_lam0, 1, 0)
        g.addWidget(self.lam0, 1, 1)
        g.addWidget(self.lbl_lam1, 1, 2)
        g.addWidget(self.lam1, 1, 3)
        g.addWidget(self.lbl_lamN, 1, 4)
        g.addWidget(self.lamN, 1, 5)
        g.addWidget(self.right_label_stack, 1, 6)
        g.addWidget(self.right_box_stack, 1, 7)

        g.addWidget(self.lbl_th0, 2, 0)
        g.addWidget(self.th0, 2, 1)
        g.addWidget(self.lbl_th1, 2, 2)
        g.addWidget(self.th1, 2, 3)
        g.addWidget(self.lbl_thN, 2, 4)
        g.addWidget(self.thN, 2, 5)

        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(8)

        col_widths = {
            0: LW,
            1: CW,
            2: LW,
            3: CW,
            4: LW2,
            5: CW,
            6: LW,
            7: CW,
        }

        for col, width in col_widths.items():
            g.setColumnMinimumWidth(col, width)
            g.setColumnStretch(col, 0)

        g.setColumnStretch(8, 1)

        g.setRowMinimumHeight(0, 48)
        g.setRowMinimumHeight(1, 48)
        g.setRowMinimumHeight(2, 48)

        FLW = 56
        FLW2 = 88

        self.lbl_field_kind = fixed_label('Field', FLW)
        self.lbl_field_component = fixed_label('Component', FLW2)
        self.lbl_field_slice = fixed_label('Slice', FLW)
        self.lbl_field_region = fixed_label('Depth', FLW)
        self.lbl_field_layer = fixed_label('Layer', FLW)

        self.field_value_label_stack = fixed_stack(FLW2)
        self.field_value_box_stack = fixed_stack(110)
        self.field_zsample_label_stack = fixed_stack(FLW2)
        self.field_zsample_box_stack = fixed_stack(110)

        self.field_value_label_stack.addWidget(fixed_label('z frac', FLW2))
        self.field_value_label_stack.addWidget(fixed_label('fixed θ', FLW2))
        self.field_value_label_stack.addWidget(fixed_label('fixed λ', FLW2))
        self.field_value_box_stack.addWidget(self.field_z_frac)
        self.field_value_box_stack.addWidget(self.field_fixed_theta)
        self.field_value_box_stack.addWidget(self.field_fixed_lam)

        self.field_zsample_label_stack.addWidget(fixed_label('', FLW2))
        self.field_zsample_label_stack.addWidget(fixed_label('z pts', FLW2))
        self.field_zsample_label_stack.addWidget(fixed_label('pts/layer', FLW2))
        self.field_zsample_box_stack.addWidget(fixed_blank(110))
        self.field_zsample_box_stack.addWidget(self.field_z_points)
        self.field_zsample_box_stack.addWidget(self.field_points_per_layer)

        self.map_grid = g

        self.standard_widgets = [
            self.lbl_map, self.mode,
            self.pol_label_stack, self.pol_box_stack,
            self.lbl_cmap, self.cmap,
            self.scale_label_stack, self.scale_box_stack,
            self.lbl_lam0, self.lam0,
            self.lbl_lam1, self.lam1,
            self.lbl_lamN, self.lamN,
            self.lbl_th0, self.th0,
            self.lbl_th1, self.th1,
            self.lbl_thN, self.thN,
            self.right_label_stack, self.right_box_stack,
        ]

        self.field_widgets = [
            self.lbl_field_kind, self.field_kind,
            self.lbl_field_component, self.field_component,
            self.lbl_field_slice, self.field_slice,
            self.lbl_field_region, self.field_depth_region,
            self.lbl_field_layer, self.field_layer,
            self.field_value_label_stack, self.field_value_box_stack,
            self.field_zsample_label_stack, self.field_zsample_box_stack,
        ]

        self._map_control_widgets = list(dict.fromkeys(self.standard_widgets + self.field_widgets))

        self.map_box = box
        box.setMinimumWidth(SETTINGS_MIN_W)
        box.setMinimumHeight(SETTINGS_STANDARD_H)
        box.setMaximumHeight(SETTINGS_STANDARD_H)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.mode.currentTextChanged.connect(lambda *_: self.update_map_controls(force_default=True))
        self.contrast_metric.currentTextChanged.connect(lambda *_: self.update_map_controls(force_default=True))
        self.ellip_output.currentTextChanged.connect(lambda *_: self.update_map_controls(force_default=True))
        self.cmap.currentTextChanged.connect(self.refresh_plot_style)
        self.scale.currentTextChanged.connect(self.refresh_plot_style)
        self.field_slice.currentTextChanged.connect(lambda *_: self.update_map_controls())
        self.field_depth_region.currentTextChanged.connect(lambda *_: self.update_map_controls())

        outer.addWidget(box)
        
        act = QHBoxLayout()
        self.run_btn = QPushButton('Run'); self.save_btn = QPushButton('Save map'); self.save_btn.setEnabled(False); self.run_btn.clicked.connect(self.run); self.save_btn.clicked.connect(self.save_map)
        self.status = QLabel('Ready'); self.contour_cb = QCheckBox('Contour'); self.submask_cb = QCheckBox('Submask'); self.peak_cb = QCheckBox('Peak')
        for cb in (self.contour_cb, self.submask_cb, self.peak_cb): cb.toggled.connect(self.refresh_overlays)
        act.addWidget(self.run_btn); act.addWidget(self.save_btn); act.addSpacing(10); act.addWidget(self.status, 1); act.addWidget(self.contour_cb); act.addWidget(self.submask_cb); act.addWidget(self.peak_cb)
        outer.addLayout(act)
        self.update_map_controls()
        self.fig = Figure(figsize=(8.5, 6.5)); self.ax = self.fig.add_subplot(111); self.canvas = FigureCanvas(self.fig); self.canvas.setMinimumHeight(260); self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); outer.addWidget(self.canvas, 1)
        self.range_info = QLabel('Layer data overlap — n: - | k: -'); self.range_info.setObjectName('RangeInfo'); self.range_info.setTextInteractionFlags(Qt.TextSelectableByMouse); outer.addWidget(self.range_info)
        self.layers = []; self.rebuild_layers(); self.refresh_catalog()
        if 'PI-PI' in self.catalog: self.sub_box.setCurrentText('PI-PI')
        elif 'AlN-Hoffmann' in self.catalog: self.sub_box.setCurrentText('AlN-Hoffmann')
        self.update_overlap_label()
        
    def _layer_state(self):
        return [(c.key.currentText().strip(), c.thk.text().strip()) for c in getattr(self, 'layers', [])]

    def _default_layer_state(self):
        key = self.catalog[0] if getattr(self, 'catalog', None) else ''
        return key, '0.20'

    def _sync_layer_count(self):
        old = self.layer_count.blockSignals(True)
        self.layer_count.setValue(len(self.layers))
        self.layer_count.blockSignals(old)

    def _clear_layer_bar(self):
        if getattr(self, '_slide', None) is not None:
            ghost = self._slide.get('ghost')
            if ghost is not None:
                ghost.hide()
                ghost.deleteLater()
        for w in getattr(self, '_layer_widgets', []):
            w.hide()
            w.setParent(None)
            w.deleteLater()
        self._layer_widgets = []
        self._gap_widgets = []
        self._slide = None

    def _slot_x(self, idx):
        return int(idx) * (LAYER_CARD_W + LAYER_GAP_W)

    def _content_width(self, total):
        return max(1, total * LAYER_CARD_W + total * LAYER_GAP_W)

    def _resize_layer_wrap(self, total):
        vp = self.layer_scroll.viewport().width() if hasattr(self, 'layer_scroll') else 0
        self.layer_wrap.setFixedSize(max(self._content_width(total), vp), LAYER_CARD_H)

    def _delete_layer_gaps(self):
        for w in getattr(self, '_gap_widgets', []):
            w.hide()
            w.setParent(None)
            w.deleteLater()
        self._gap_widgets = []
        self._layer_widgets = list(getattr(self, 'layers', []))

    def _rebuild_layer_gaps(self):
        self._delete_layer_gaps()
        total = len(self.layers)
        if total < 1:
            return
        for i in range(total - 1):
            gap = InsertLayerGap(parent=self.layer_wrap)
            gap.move(self._slot_x(i) + LAYER_CARD_W, 0)
            gap.btn.clicked.connect(partial(self.insert_layer, i + 1))
            gap.show()
            self._gap_widgets.append(gap)
            self._layer_widgets.append(gap)

        end_gap = InsertLayerGap(parent=self.layer_wrap)
        end_gap.move(self._slot_x(total - 1) + LAYER_CARD_W, 0)
        end_gap.btn.clicked.connect(partial(self.insert_layer, total))
        end_gap.show()
        self._gap_widgets.append(end_gap)
        self._layer_widgets.append(end_gap)

    def _rebind_layer_buttons(self):
        total = len(self.layers)
        for i, c in enumerate(self.layers):
            try:
                c.del_btn.clicked.disconnect()
            except (TypeError, RuntimeError):
                pass
            c.del_btn.clicked.connect(partial(self.delete_layer, i))
            c.rename(i, total)

    def _layout_existing_layers(self):
        total = len(self.layers)
        self._resize_layer_wrap(total)
        self._rebind_layer_buttons()
        for i, c in enumerate(self.layers):
            c.move(self._slot_x(i), 0)
            c.show()
        self._rebuild_layer_gaps()

    def _apply_layer_order(self, old_idx, new_idx):
        total = len(self.layers)
        if total <= 1:
            return
        old_idx = max(0, min(int(old_idx), total - 1))
        new_idx = max(0, min(int(new_idx), total - 1))
        if old_idx == new_idx:
            self.layers[old_idx].move(self._slot_x(old_idx), 0)
            return
        card = self.layers.pop(old_idx)
        self.layers.insert(new_idx, card)
        self._layout_existing_layers()
        self._sync_layer_count()
        self.update_overlap_label()

    def _set_layer_state(self, state):
        state = list(state)
        if not state:
            state = [self._default_layer_state()]

        self._clear_layer_bar()
        self.layers = []
        total = len(state)
        self._resize_layer_wrap(total)

        for i, (key, thk) in enumerate(state):
            c = LayerCard(i, total, self.catalog)
            c.setParent(self.layer_wrap)
            c.move(self._slot_x(i), 0)

            if key:
                c.key.setCurrentText(key)
            if thk:
                c.thk.setText(thk)

            c.key.currentTextChanged.connect(self.update_overlap_label)
            c.key.lineEdit().editingFinished.connect(self.update_overlap_label)
            c.del_btn.clicked.connect(partial(self.delete_layer, i))

            c.show()
            self._install_layer_drag_filters(c)
            self.layers.append(c)
            self._layer_widgets.append(c)

        self._rebuild_layer_gaps()
        self._sync_layer_count()
        self.update_overlap_label()

    def rebuild_layers(self):
        state = self._layer_state()
        target = self.layer_count.value()

        if target > len(state):
            state.extend(self._default_layer_state() for _ in range(target - len(state)))
        else:
            state = state[:target]

        self._set_layer_state(state)

    def delete_layer(self, idx):
        state = self._layer_state()
        if len(state) <= 1:
            return
        idx = max(0, min(int(idx), len(state) - 1))
        state.pop(idx)
        self._set_layer_state(state)

    def insert_layer(self, idx):
        state = self._layer_state()
        idx = max(0, min(int(idx), len(state)))
        state.insert(idx, self._default_layer_state())
        self._set_layer_state(state)

    def _install_layer_drag_filters(self, card):
        for w in (card, card.lab, card.mat_label, card.thk_label, card.unit_label):
            w.installEventFilter(self)

    def _mouse_global_pos(self, event):
        if hasattr(event, 'globalPosition'):
            return event.globalPosition().toPoint()
        return event.globalPos()

    def _widget_inside(self, obj, parent):
        w = obj if isinstance(obj, QWidget) else None
        while w is not None:
            if w is parent:
                return True
            w = w.parentWidget()
        return False

    def _card_at_global(self, global_pos):
        if not hasattr(self, 'layer_wrap'):
            return None
        p = self.layer_wrap.mapFromGlobal(global_pos)
        for c in reversed(getattr(self, 'layers', [])):
            if c.geometry().contains(p):
                return c
        return None

    def _is_layer_drag_control(self, obj, card):
        if obj is None or card is None:
            return False
        if self._widget_inside(obj, card.del_btn):
            return False
        if self._widget_inside(obj, card.key):
            return False
        if self._widget_inside(obj, card.thk):
            return False
        return self._widget_inside(obj, card)

    def eventFilter(self, obj, event):
        et = event.type()
        if et in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease):
            if self._handle_layer_slide_event(obj, event):
                return True
        return super().eventFilter(obj, event)

    def _begin_layer_slide_ghost(self):
        s = self._slide
        if s is None or s.get('ghost') is not None:
            return
        card = s['card']
        ghost = QLabel(self.layer_wrap)
        ghost.setObjectName('LayerDragGhost')
        ghost.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        ghost.setPixmap(card.grab())
        ghost.setFixedSize(card.size())
        ghost.move(card.pos())
        ghost.show()
        ghost.raise_()
        s['ghost'] = ghost

    def _handle_layer_slide_event(self, obj, event):
        et = event.type()
        if et == QEvent.Type.MouseButtonPress:
            if event.button() != Qt.LeftButton:
                return False
            gpos = self._mouse_global_pos(event)
            card = self._card_at_global(gpos)
            if card is None or not self._is_layer_drag_control(obj, card):
                return False
            self._slide = {
                'card': card,
                'old_idx': card.idx,
                'start_global_x': gpos.x(),
                'start_x': card.x(),
                'active': False,
                'current_idx': card.idx,
                'state': self._layer_state(),
                'ghost': None,
            }
            card.grabMouse()
            event.accept()
            return True

        if self._slide is None:
            return False

        if et == QEvent.Type.MouseMove:
            if not (event.buttons() & Qt.LeftButton):
                return False
            gpos = self._mouse_global_pos(event)
            dx = gpos.x() - self._slide['start_global_x']
            if not self._slide['active'] and abs(dx) < 4:
                event.accept()
                return True
            self._slide['active'] = True
            self._move_layer_slide(dx)
            event.accept()
            return True

        if et == QEvent.Type.MouseButtonRelease:
            if event.button() != Qt.LeftButton:
                return False
            self._finish_layer_slide()
            event.accept()
            return True

        return False

    def _move_layer_slide(self, dx):
        s = self._slide
        if s is None:
            return
        total = len(self.layers)
        if total <= 1:
            return

        self._begin_layer_slide_ghost()
        ghost = s.get('ghost')
        if ghost is None:
            return

        slot = LAYER_CARD_W + LAYER_GAP_W
        max_x = self._slot_x(total - 1)
        x = max(0, min(s['start_x'] + int(dx), max_x))
        ghost.move(x, 0)
        ghost.raise_()
        s['current_idx'] = max(0, min(int(round(x / slot)), total - 1))

    def _finish_layer_slide(self):
        s = self._slide
        if s is None:
            return
        card = s['card']
        try:
            card.releaseMouse()
        except Exception:
            pass

        old_idx = s['old_idx']
        new_idx = s['current_idx']
        active = bool(s['active'])
        ghost = s.get('ghost')
        self._slide = None

        if ghost is not None:
            ghost.hide()
            ghost.deleteLater()

        if not active or new_idx == old_idx:
            card.move(self._slot_x(old_idx), 0)
            card.show()
            return

        self._apply_layer_order(old_idx, new_idx)

    def _canonical_key_info(self, key):
        return key_info(self.nk, key)
    
    def _dataset_bounds(self, key):
        info = self._canonical_key_info(key)
        if not info['valid']: return {'valid': False, 'n': None, 'k': None}
        base = info['base']; kind = info['kind']
        if _vm_parse(base) is not None:
            if kind == 'n': return {'valid': True, 'n': (-np.inf, np.inf), 'k': None}
            if kind == 'k': return {'valid': True, 'n': None, 'k': (-np.inf, np.inf)}
            return {'valid': True, 'n': (-np.inf, np.inf), 'k': (-np.inf, np.inf)}
        try:
            wn, _, wk, _ = load_key_data(self.nk, key)
        except Exception:
            return {'valid': False, 'n': None, 'k': None}
        nb = (float(wn[0]), float(wn[-1])) if np.asarray(wn).size else None
        kb = (float(wk[0]), float(wk[-1])) if np.asarray(wk).size else None
        return {'valid': True, 'n': nb, 'k': kb}
    
    def _overlap_bounds(self, bounds):
        good = [b for b in bounds if b is not None]
        if not good: return None
        lo = max(b[0] for b in good); hi = min(b[1] for b in good)
        return (lo, hi) if hi >= lo else None
    
    def _fmt_overlap(self, ov, all_virtual=False, missing=False):
        if missing: return 'none'
        if ov is None: return '0'
        if all_virtual or (np.isneginf(ov[0]) and np.isposinf(ov[1])): return 'all wavelengths'
        return f'{ov[0]:.3f}–{ov[1]:.3f} µm'
    
    def update_overlap_label(self):
        if not hasattr(self, 'range_info'): return
        keys = [(c.key.currentText() or '').strip() for c in getattr(self, 'layers', [])]
        if not keys:
            self.range_info.setText('Layer data overlap — n: - | k: -'); return
        n_bounds, k_bounds, invalid = [], [], []; all_vm_n = True; all_vm_k = True
        for k in keys:
            info = self._dataset_bounds(k)
            if not info['valid']:
                invalid.append(k or '(blank)'); continue
            nb, kb = info['n'], info['k']; n_bounds.append(nb); k_bounds.append(kb)
            all_vm_n = all_vm_n and (nb is not None and np.isneginf(nb[0]) and np.isposinf(nb[1]))
            all_vm_k = all_vm_k and (kb is not None and np.isneginf(kb[0]) and np.isposinf(kb[1]))
        if invalid:
            self.range_info.setText('Layer data overlap — invalid key(s): ' + ', '.join(invalid[:3]) + (' ...' if len(invalid) > 3 else '')); return
        n_missing = any(b is None for b in n_bounds); k_missing = any(b is None for b in k_bounds); nov = self._overlap_bounds(n_bounds); kov = self._overlap_bounds(k_bounds)
        self.range_info.setText('Layer data overlap — ' f'n: {self._fmt_overlap(nov, all_virtual=all_vm_n, missing=n_missing)} | ' f'k: {self._fmt_overlap(kov, all_virtual=all_vm_k, missing=k_missing)}')
    
    def refresh_catalog(self):
        cur_sub = self.sub_box.currentText().strip() if hasattr(self, 'sub_box') else ''
        cur_layers = [c.key.currentText().strip() for c in getattr(self, 'layers', [])]
        self.nk, self.catalog = build_catalog(); self.sub_box.set_items(self.catalog)
        if cur_sub: self.sub_box.setCurrentText(cur_sub)
        for c, txt in zip(self.layers, cur_layers):
            c.key.set_items(self.catalog)
            if txt: c.key.setCurrentText(txt)
        self.status.setText(f'Materials loaded: {len(self.catalog)}'); self.update_overlap_label()
    
    def open_settings(self):
        d = SettingsDialog(self.settings, self)
        if d.exec():
            self.settings.update(d.values())
            save_ui_settings(self.settings)
    
    def _valid_key(self, k): return self._canonical_key_info(k)['valid']
    
    def _canonical_key(self, k):
        info = self._canonical_key_info(k)
        return info['base'] if info['valid'] else (k or '').strip()
    
    def payload(self):
        raw_films = [c.key.currentText().strip() for c in self.layers]
        films = [self._canonical_key(k) for k in raw_films]
        dL = [float(c.thk.text()) for c in self.layers]

        if any(not self._valid_key(k) for k in raw_films):
            raise ValueError('One or more layer keys are invalid.')

        sub_raw = self.sub_box.currentText().strip()

        if not self._valid_key(sub_raw):
            raise ValueError('Invalid substrate key.')

        sub = self._canonical_key(sub_raw)
        mode = self.mode.currentText().strip()
        lam0 = float(self.lam0.text())
        lam1 = float(self.lam1.text())
        lamN = int(self.lamN.text())
        th0 = float(self.th0.text())
        th1 = float(self.th1.text())
        thN = int(self.thN.text())
        temperature = float(self.tempK.text())

        is_field = mode in FIELD_MAP_TYPES
        field_slice = self._field_slice_for_current() if is_field else 'lambda-theta'
        field_region = self.field_depth_region.currentText().strip() if is_field else 'stack'

        if (not is_field) or field_slice in ('lambda-theta', 'lambda-depth'):
            if lam1 <= lam0:
                raise ValueError('λ max must be greater than λ min.')

        if (not is_field) or field_slice in ('lambda-theta', 'theta-depth'):
            if th1 < th0:
                raise ValueError('θ max must be greater than or equal to θ min.')

        if temperature <= 0:
            raise ValueError('Temperature must be greater than 0 K.')

        field_layer = int(self.field_layer.text() or 1)
        field_z_frac = float(self.field_z_frac.text() or 0.5)
        field_z_points = int(self.field_z_points.text() or 200)
        field_points_per_layer = int(self.field_points_per_layer.text() or 80)
        field_fixed_theta = float(self.field_fixed_theta.text() or 0.0)
        field_fixed_lam = float(self.field_fixed_lam.text() or ((lam0 + lam1) * 0.5))

        if is_field:
            needs_layer = field_slice == 'lambda-theta' or field_region == 'selected layer'

            if needs_layer and (field_layer < 1 or field_layer > len(dL)):
                raise ValueError('Field layer must be within the current film stack.')

            if field_slice == 'lambda-theta' and not (0.0 <= field_z_frac <= 1.0):
                raise ValueError('Field z frac must be between 0 and 1.')

            if field_slice in ('lambda-depth', 'theta-depth'):
                if field_region == 'selected layer' and field_z_points < 2:
                    raise ValueError('Field z pts must be at least 2.')
                if field_region == 'stack' and field_points_per_layer < 2:
                    raise ValueError('Field pts/layer must be at least 2.')

            if field_slice == 'lambda-depth':
                if not (0.0 <= field_fixed_theta < 90.0):
                    raise ValueError('Fixed θ must be in the range 0 ≤ θ < 90 deg.')

            if field_slice == 'theta-depth':
                if field_fixed_lam <= 0:
                    raise ValueError('Fixed λ must be greater than 0.')

        return {
            'films': films,
            'dL': dL,
            'sub': sub,
            'mode': mode,
            'pol': self.pol.currentText(),
            'lam0': lam0,
            'lam1': lam1,
            'lamN': lamN,
            'th0': th0,
            'th1': th1,
            'thN': thN,
            'band_lam0': lam0,
            'band_lam1': lam1,
            'band_lamN': lamN,
            'hemi_thetaN': thN,
            'temperature': temperature,
            'contrast_metric': self.contrast_metric.currentText(),
            'ellip_output': self.ellip_output.currentText(),
            'field_kind': self.field_kind.currentText(),
            'field_component': self.field_component.currentText(),
            'field_normalization': self.scale.currentText().strip().lower() if is_field else 'incident',
            'field_slice': field_slice,
            'field_depth_region': self.field_depth_region.currentText(),
            'field_layer': field_layer,
            'field_z_frac': field_z_frac,
            'field_z_points': field_z_points,
            'field_points_per_layer': field_points_per_layer,
            'field_fixed_theta_deg': field_fixed_theta,
            'field_fixed_lam_um': field_fixed_lam,
            'nk': self.nk,
            **self.settings,
        }
    
    def run(self):
        try: p = self.payload()
        except Exception as e:
            QMessageBox.critical(self, 'Input error', str(e)); return
        self.run_btn.setEnabled(False); self.save_btn.setEnabled(False); self.status.setText('Running...'); self._thread = QThread(self); self._worker = Worker(p); self._worker.moveToThread(self._thread); self._thread.started.connect(self._worker.run); self._worker.done.connect(self.finish); self._worker.done.connect(self._thread.quit); self._worker.done.connect(self._worker.deleteLater); self._thread.finished.connect(self._thread.deleteLater); self._thread.start()
    
    def finish(self, md, err):
        self.run_btn.setEnabled(True)
        if err:
            self.status.setText('Failed'); QMessageBox.critical(self, 'Run error', err); return
        self.last_md = md; self._overlay_cache = None; self.save_btn.setEnabled(True); self.plot_map(md); self.status.setText('Done')
    
    def _overlay_enabled(self): return self.contour_cb.isChecked() or self.submask_cb.isChecked() or self.peak_cb.isChecked()
    
    def _can_draw_overlays(self, md):
        if np.asarray(md.val).ndim != 2:
            return False
        return str(md.mode).strip().lower() in ('e', 'emissivity', 'r', 'reflectivity')

    def _compute_overlays(self, md):
        if self._overlay_cache is not None:
            return self._overlay_cache

        if not self._can_draw_overlays(md):
            return None

        lam = np.asarray(md.lam_um, float)
        th = np.asarray(md.theta_deg, float)
        val = np.asarray(md.val, float)
        is_reflectivity = str(md.mode).strip().lower() in ('r', 'reflectivity')
        mode_for_mask = 'R' if is_reflectivity else 'E'
        peak_kind = 'low' if is_reflectivity else 'high'

        lam2, th2, mask, thr_used, _ = contour.mask(val, lam, th, mode=mode_for_mask, return_report=True, **MASK_CFG)
        pw = np.ones(np.asarray(lam2).size, float)
        aw = np.ones(np.asarray(th2).size, float)

        mv, mh, _ = submask.peak_masks(
            val, lam2, th2, mask,
            mode=mode_for_mask,
            peak=MASK_CFG.get('peak') or peak_kind,
            thr_used=thr_used,
            return_debug=True,
            pw=pw,
            aw=aw,
            **SUBMASK_CFG
        )

        vmeta, hmeta, _ = simplepeak.locate_1d_peaks(
            val, lam2, th2, mask,
            mode=mode_for_mask,
            peak=peak_kind,
            thr_used=thr_used,
            return_meta=True
        )

        self._overlay_cache = {
            'lam': lam2,
            'theta': th2,
            'contour_mask': mask,
            'submask_v': mv,
            'submask_h': mh,
            'peak_v': vmeta,
            'peak_h': hmeta,
        }

        return self._overlay_cache
    
    def _draw_overlays(self, ax, md):
        if not self._overlay_enabled() or not self._can_draw_overlays(md):
            return

        try:
            ov = self._compute_overlays(md)
        except Exception:
            self.status.setText('Overlay error')
            return
        if ov is None:
            return
        if self.submask_cb.isChecked():
            ax.contourf(ov['lam'], ov['theta'], np.asarray(ov['submask_h'], int), levels=[0.5, 1.5], colors=['none'], hatches=['////'], alpha=0, zorder=3)
            ax.contourf(ov['lam'], ov['theta'], np.asarray(ov['submask_v'], int), levels=[0.5, 1.5], colors=['none'], hatches=['xx'], alpha=0, zorder=3)
        if self.contour_cb.isChecked(): contour.draw(ax, ov['lam'], ov['theta'], ov['contour_mask'], color='#00a0ff', lw=2.0, alpha=0.9, zorder=4, pick='all')
        if self.peak_cb.isChecked():
            fx = [pe.Stroke(linewidth=4.2, foreground='black'), pe.Normal()]
            for o in ov['peak_v']: ax.axvline(float(o['x']), color='white', linestyle='--', linewidth=2.2, alpha=1.0, zorder=5, path_effects=fx)
            for o in ov['peak_h']: ax.axhline(float(o['y']), color='white', linestyle='--', linewidth=2.2, alpha=1.0, zorder=5, path_effects=fx)
    
    def refresh_overlays(self):
        if self.last_md is not None: self.plot_map(self.last_md)

    def _selected_cmap(self):
        cm = self.cmap.currentText().strip()
        return cm if cm in MAP_CMAPS else DEFAULT_MAP_CMAP

    def _save_visual_settings(self):
        self.settings['map_cmap'] = self._selected_cmap()
        self.settings['map_scale'] = self.scale.currentText().strip() or DEFAULT_MAP_SCALE
        save_ui_settings(self.settings)

    def refresh_plot_style(self):
        self._save_visual_settings()
        if self.last_md is not None:
            self.plot_map(self.last_md)

    def _scale_choices_for_current(self):
        mode = self.mode.currentText().strip()
        ellip = self.ellip_output.currentText().strip()

        if mode in FIELD_MAP_TYPES:
            return FIELD_SCALE_OPTIONS

        if mode == 'Polarization contrast':
            return ['Normalized -1~0~1', 'Normalized |value|', 'Auto data']

        if mode == 'Ellipsometry':
            if ellip.startswith('Δ'):
                return ['Normalized -1~0~1', 'Normalized |value|', 'Auto data']
            return ['Normalized 0-1', 'Auto data']

        if mode == 'Planck-weighted emissive power':
            return ['Normalized 0-1', 'Auto data']

        return ['Normalized 0-1', 'Auto data']

    def _default_scale_for_current(self):
        mode = self.mode.currentText().strip()
        ellip = self.ellip_output.currentText().strip()

        if mode in FIELD_MAP_TYPES:
            return 'Incident'

        if mode == 'Polarization contrast':
            return 'Normalized -1~0~1'

        if mode == 'Ellipsometry' and ellip.startswith('Δ'):
            return 'Normalized -1~0~1'

        return 'Normalized 0-1'

    def _sync_scale_options(self, force_default=False):
        choices = self._scale_choices_for_current()
        cur = self.scale.currentText().strip()

        if cur == 'Fixed 0-1':
            cur = 'Normalized 0-1'

        if force_default or cur not in choices:
            cur = self._default_scale_for_current()

        old = self.scale.blockSignals(True)
        self.scale.clear()
        self.scale.addItems(choices)
        self.scale.setCurrentText(cur)
        self.scale.blockSignals(old)

        self.settings['map_scale'] = cur
        save_ui_settings(self.settings)

    def _auto_limits(self, arr):
        a = np.asarray(arr, float)
        a = a[np.isfinite(a)]

        if a.size == 0:
            return 0.0, 1.0

        vmin = float(np.nanmin(a))
        vmax = float(np.nanmax(a))

        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return 0.0, 1.0

        if vmin == vmax:
            pad = 0.5 if vmin == 0 else abs(vmin) * 0.05
            return vmin - pad, vmax + pad

        return vmin, vmax

    def _normalize_01(self, arr):
        a = np.asarray(arr, float)
        out = np.full_like(a, np.nan, dtype=float)
        m = np.isfinite(a)

        if not np.any(m):
            return out

        lo = float(np.nanmin(a[m]))
        hi = float(np.nanmax(a[m]))

        if hi == lo:
            out[m] = 0.0
        else:
            out[m] = (a[m] - lo) / (hi - lo)

        return out

    def _normalize_signed(self, arr):
        a = np.asarray(arr, float)
        out = np.full_like(a, np.nan, dtype=float)
        m = np.isfinite(a)

        if not np.any(m):
            return out

        s = float(np.nanmax(np.abs(a[m])))

        if s == 0.0:
            out[m] = 0.0
        else:
            out[m] = a[m] / s

        return out

    def _normalize_abs(self, arr):
        a = np.asarray(arr, float)
        out = np.full_like(a, np.nan, dtype=float)
        m = np.isfinite(a)

        if not np.any(m):
            return out

        s = float(np.nanmax(np.abs(a[m])))

        if s == 0.0:
            out[m] = 0.0
        else:
            out[m] = np.abs(a[m]) / s

        return out

    def _scaled_map_data(self, arr, md):
        scale = self.scale.currentText().strip() if hasattr(self, 'scale') else DEFAULT_MAP_SCALE
        meta = md.meta or {}
        label = meta.get('cbar_label', str(md.mode))
        mode = str(md.mode).strip().lower()

        if meta.get('field_kind') in ('E', 'B'):
            vmin, vmax = (0.0, 1.0) if scale == 'Maximum' else self._auto_limits(arr)
            return arr, vmin, vmax, label

        if scale == 'Auto data':
            vmin, vmax = self._auto_limits(arr)
            return arr, vmin, vmax, label

        if scale == 'Normalized -1~0~1':
            return self._normalize_signed(arr), -1.0, 1.0, f'Normalized signed value ({label})'

        if scale == 'Normalized |value|':
            return self._normalize_abs(arr), 0.0, 1.0, f'Normalized |value| ({label})'

        if mode in ('emissivity', 'reflectivity'):
            return arr, 0.0, 1.0, label

        return self._normalize_01(arr), 0.0, 1.0, f'Normalized intensity ({label})'



    
    def _title_text(self, md):
        meta = md.meta or {}
        base = meta.get('title', str(md.mode))
        mats = ' | '.join(meta.get('film_keys') or [])
        dL = ' | '.join(f'{float(v):.2f}' for v in (meta.get('dL') or []))
        if mats:
            return f'{base} of {mats}\n{dL}'
        return base

    def _draw_result(self, fig, md, title_fs):
        fig.clear()
        val = np.asarray(md.val, float)
        meta = md.meta or {}

        if val.ndim == 1:
            ax = fig.add_subplot(111)
            x = meta.get('x', None)
            if x is None:
                x = md.lam_um if np.asarray(md.lam_um).size == val.size else np.arange(val.size)
            x = np.asarray(x, float)

            ax.plot(x, val, linewidth=1.8)
            ax.set_xlabel(meta.get('x_label', 'x'), labelpad=4)
            ax.set_ylabel(meta.get('y_label', str(md.mode)), labelpad=6)
            ax.grid(True, alpha=0.3)
            ax.set_title(self._title_text(md), fontsize=title_fs, pad=6)
            fig.subplots_adjust(left=0.10, right=0.97, bottom=0.14, top=0.86)
            return fig

        gs = fig.add_gridspec(1, 2, width_ratios=[48, 1.35], left=0.075, right=0.925, bottom=0.14, top=0.88, wspace=0.12)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        plot_val, vmin, vmax, cbar_label = self._scaled_map_data(val, md)

        x = np.asarray(meta.get('x', md.lam_um), float)
        y = np.asarray(meta.get('y', md.theta_deg), float)

        im = ax.pcolormesh(
            x,
            y,
            plot_val.T,
            shading='auto',
            vmin=vmin,
            vmax=vmax,
            cmap=self._selected_cmap()
        )

        ax.set_xlabel(meta.get('x_label', 'Wavelength (µm)'), labelpad=4)
        ax.set_ylabel(meta.get('y_label', 'Angle (deg)'), labelpad=6)
        ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
        ax.set_ylim(float(np.nanmin(y)), float(np.nanmax(y)))
        ax.set_title(self._title_text(md), fontsize=title_fs, pad=6)

        self._draw_overlays(ax, md)

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label, labelpad=6)
        return fig

    def plot_map(self, md):
        self._draw_result(self.fig, md, 11)
        self.canvas.draw_idle()

    def save_map(self):
        if self.last_md is None:
            return

        path, _ = QFileDialog.getSaveFileName(self, 'Save map', str(BASE / 'map.png'), 'PNG (*.png);;PDF (*.pdf);;SVG (*.svg)')

        if not path:
            return

        md = self.last_md
        aw = max(float(self.settings.get('save_aspect_w', 16.0)), 0.1)
        ah = max(float(self.settings.get('save_aspect_h', 9.0)), 0.1)
        fig = Figure(figsize=(aw, ah), dpi=120)

        self._draw_result(fig, md, 16)

        ext = Path(path).suffix.lower()
        fig.savefig(path, dpi=120 if ext == '.png' else None, facecolor='white')
        self.status.setText(f'Saved: {path}')
        
    def _set_many_visible(self, widgets, visible):
        for w in widgets:
            w.setVisible(visible)

    def _pair(self, label, widget, row, pair):
        c = pair * 2
        self.map_grid.addWidget(label, row, c)
        self.map_grid.addWidget(widget, row, c + 1)
        label.setVisible(True)
        widget.setVisible(True)

    def _clear_map_grid_visibility(self):
        self._set_many_visible(self._map_control_widgets, False)

    def _set_row_heights(self, active_rows):
        for r in range(6):
            self.map_grid.setRowMinimumHeight(r, 0)
        for r in active_rows:
            self.map_grid.setRowMinimumHeight(r, 44)

    def _resize_standard_controls(self):
        self.mode.setFixedWidth(self._ui_CW)
        self.pol.setFixedWidth(self._ui_CW)
        self.cmap.setFixedWidth(self._ui_CW)
        self.scale.setFixedWidth(self._ui_CW)
        self.contrast_metric.setFixedWidth(self._ui_CW)
        self.ellip_output.setFixedWidth(self._ui_CW)

        self.pol_box_stack.setFixedWidth(self._ui_CW)
        self.scale_box_stack.setFixedWidth(self._ui_CW)
        self.right_box_stack.setFixedWidth(self._ui_CW)

        for w in (self.lam0, self.lam1, self.th0, self.th1, self.lamN, self.thN, self.tempK):
            w.setFixedWidth(self._ui_EW)

    def _resize_field_controls(self):
        self.mode.setFixedWidth(self._ui_CW)
        self.pol.setFixedWidth(self._ui_CW)
        self.cmap.setFixedWidth(self._ui_CW)
        self.scale.setFixedWidth(self._ui_CW)

        self.field_kind.setFixedWidth(self._ui_CW)
        self.field_component.setFixedWidth(self._ui_CW)
        self.field_slice.setFixedWidth(self._ui_CW)
        self.field_depth_region.setFixedWidth(self._ui_CW)

        self.pol_box_stack.setFixedWidth(self._ui_CW)
        self.scale_box_stack.setFixedWidth(self._ui_CW)

        for w in (self.lam0, self.lam1, self.th0, self.th1, self.lamN, self.thN):
            w.setFixedWidth(self._ui_EW)

        for w in (
            self.field_layer,
            self.field_z_frac,
            self.field_z_points,
            self.field_points_per_layer,
            self.field_fixed_theta,
            self.field_fixed_lam,
        ):
            w.setFixedWidth(self._ui_EW)

        self.field_value_box_stack.setFixedWidth(self._ui_EW)
        self.field_zsample_box_stack.setFixedWidth(self._ui_EW)

        for w in (self.lbl_field_kind, self.lbl_field_slice, self.lbl_field_region, self.lbl_field_layer):
            w.setFixedWidth(self._ui_LW)

        self.lbl_field_component.setFixedWidth(self._ui_LW2)
        self.field_value_label_stack.setFixedWidth(self._ui_LW2)
        self.field_zsample_label_stack.setFixedWidth(self._ui_LW2)

    def _layout_standard_map_controls(self):
        self._resize_standard_controls()
        self._clear_map_grid_visibility()

        for w in self.standard_widgets:
            w.setVisible(True)

        g = self.map_grid

        g.addWidget(self.lbl_map, 0, 0)
        g.addWidget(self.mode, 0, 1)
        g.addWidget(self.pol_label_stack, 0, 2)
        g.addWidget(self.pol_box_stack, 0, 3)
        g.addWidget(self.lbl_cmap, 0, 4)
        g.addWidget(self.cmap, 0, 5)
        g.addWidget(self.scale_label_stack, 0, 6)
        g.addWidget(self.scale_box_stack, 0, 7)

        g.addWidget(self.lbl_lam0, 1, 0)
        g.addWidget(self.lam0, 1, 1)
        g.addWidget(self.lbl_lam1, 1, 2)
        g.addWidget(self.lam1, 1, 3)
        g.addWidget(self.lbl_lamN, 1, 4)
        g.addWidget(self.lamN, 1, 5)
        g.addWidget(self.right_label_stack, 1, 6)
        g.addWidget(self.right_box_stack, 1, 7)

        g.addWidget(self.lbl_th0, 2, 0)
        g.addWidget(self.th0, 2, 1)
        g.addWidget(self.lbl_th1, 2, 2)
        g.addWidget(self.th1, 2, 3)
        g.addWidget(self.lbl_thN, 2, 4)
        g.addWidget(self.thN, 2, 5)

        self._set_row_heights([0, 1, 2])

    def _layout_axis_range(self, row, axis):
        if axis == 'lambda':
            self._pair(self.lbl_lam0, self.lam0, row, 0)
            self._pair(self.lbl_lam1, self.lam1, row, 1)
            self._pair(self.lbl_lamN, self.lamN, row, 2)
        else:
            self._pair(self.lbl_th0, self.th0, row, 0)
            self._pair(self.lbl_th1, self.th1, row, 1)
            self._pair(self.lbl_thN, self.thN, row, 2)

    def _layout_field_map_controls(self):
        self._resize_field_controls()
        self._clear_map_grid_visibility()

        sl = self._field_slice_for_current()
        selected_layer = self.field_depth_region.currentText().strip() == 'selected layer'

        self.lbl_scale.setText('Norm.')
        self.pol_label_stack.setCurrentIndex(0)
        self.pol_box_stack.setCurrentIndex(0)
        self.scale_label_stack.setCurrentIndex(0)
        self.scale_box_stack.setCurrentIndex(0)

        self._pair(self.lbl_map, self.mode, 0, 0)
        self._pair(self.lbl_field_kind, self.field_kind, 0, 1)
        self._pair(self.lbl_field_component, self.field_component, 0, 2)
        self._pair(self.pol_label_stack, self.pol_box_stack, 0, 3)

        self._pair(self.lbl_cmap, self.cmap, 1, 0)
        self._pair(self.scale_label_stack, self.scale_box_stack, 1, 1)
        self._pair(self.lbl_field_slice, self.field_slice, 1, 2)

        if sl == 'lambda-theta':
            self._pair(self.lbl_field_layer, self.field_layer, 1, 3)

            self._layout_axis_range(2, 'lambda')
            self._layout_axis_range(3, 'theta')

            self.field_value_label_stack.setCurrentIndex(0)
            self.field_value_box_stack.setCurrentIndex(0)
            self._pair(self.field_value_label_stack, self.field_value_box_stack, 3, 3)

            self._set_row_heights([0, 1, 2, 3])
            return

        self._pair(self.lbl_field_region, self.field_depth_region, 1, 3)

        if sl == 'lambda-depth':
            self._layout_axis_range(2, 'lambda')
            self.field_value_label_stack.setCurrentIndex(1)
            self.field_value_box_stack.setCurrentIndex(1)
        else:
            self._layout_axis_range(2, 'theta')
            self.field_value_label_stack.setCurrentIndex(2)
            self.field_value_box_stack.setCurrentIndex(2)

        self._pair(self.field_value_label_stack, self.field_value_box_stack, 2, 3)

        zidx = 1 if selected_layer else 2
        self.field_zsample_label_stack.setCurrentIndex(zidx)
        self.field_zsample_box_stack.setCurrentIndex(zidx)

        self._pair(self.field_zsample_label_stack, self.field_zsample_box_stack, 3, 0)

        if selected_layer:
            self._pair(self.lbl_field_layer, self.field_layer, 3, 1)

        self._set_row_heights([0, 1, 2, 3])
        
    def _field_slice_for_current(self):
        return self.field_slice.currentText().strip()

    def _sync_field_controls(self):
        if self.mode.currentText().strip() not in FIELD_MAP_TYPES:
            return

        self.field_kind.setEnabled(True)
        self.field_slice.setEnabled(True)

        sl = self._field_slice_for_current()
        depth_map = sl in ('lambda-depth', 'theta-depth')
        selected_layer = self.field_depth_region.currentText().strip() == 'selected layer'

        self.field_depth_region.setEnabled(depth_map)
        self.field_layer.setEnabled(sl == 'lambda-theta' or selected_layer)
        self.field_z_frac.setEnabled(sl == 'lambda-theta')

    def update_map_controls(self, force_default=False):
        mode = self.mode.currentText().strip()
        is_field = mode in FIELD_MAP_TYPES

        self.lbl_lam0.setText('λ min')
        self.lbl_lam1.setText('λ max')
        self.lbl_lamN.setText('λ pts')
        self.lbl_th0.setText('θ min')
        self.lbl_th1.setText('θ max')
        self.lbl_thN.setText('θ pts')
        self.lbl_scale.setText('Scale')

        self.lam0.setEnabled(True)
        self.lam1.setEnabled(True)
        self.lamN.setEnabled(True)
        self.th0.setEnabled(True)
        self.th1.setEnabled(True)
        self.thN.setEnabled(True)

        show_pol = mode not in ('Polarization contrast', 'Ellipsometry')
        show_scale = mode not in ('Band-averaged Emissivity', 'Hemispherical emissivity')

        self.pol_label_stack.setCurrentIndex(0 if show_pol else 1)
        self.pol_box_stack.setCurrentIndex(0 if show_pol else 1)
        self.scale_label_stack.setCurrentIndex(0 if show_scale else 1)
        self.scale_box_stack.setCurrentIndex(0 if show_scale else 1)

        if mode == 'Planck-weighted emissive power':
            self.right_label_stack.setCurrentIndex(1)
            self.right_box_stack.setCurrentIndex(1)
        elif mode == 'Polarization contrast':
            self.right_label_stack.setCurrentIndex(2)
            self.right_box_stack.setCurrentIndex(2)
        elif mode == 'Ellipsometry':
            self.right_label_stack.setCurrentIndex(3)
            self.right_box_stack.setCurrentIndex(3)
        else:
            self.right_label_stack.setCurrentIndex(0)
            self.right_box_stack.setCurrentIndex(0)

        if mode == 'Band-averaged Emissivity':
            self.lbl_lam0.setText('band λ min')
            self.lbl_lam1.setText('band λ max')
            self.lbl_lamN.setText('λ int pts')

        if mode == 'Hemispherical emissivity':
            self.th0.setText('0')
            self.th1.setText('90')
            self.th0.setEnabled(False)
            self.th1.setEnabled(False)
            self.lbl_thN.setText('θ int pts')
            self._hemi_forced_angle = True
        else:
            if getattr(self, '_hemi_forced_angle', False) and self.th1.text().strip() == '90':
                self.th1.setText(DEFAULT_TH1)
            self._hemi_forced_angle = False

        if is_field:
            self._sync_field_controls()
            self._layout_field_map_controls()
            h = SETTINGS_FIELD_H
        else:
            self._layout_standard_map_controls()
            h = SETTINGS_STANDARD_H

        if hasattr(self, 'map_box'):
            self.map_box.setMinimumWidth(SETTINGS_MIN_W)
            self.map_box.setMinimumHeight(h)
            self.map_box.setMaximumHeight(h)

        overlay_allowed = mode in ('Emissivity', 'Reflectivity')

        if hasattr(self, 'contour_cb'):
            for cb in (self.contour_cb, self.submask_cb, self.peak_cb):
                if not overlay_allowed:
                    cb.setChecked(False)
                cb.setVisible(overlay_allowed)
                cb.setEnabled(overlay_allowed)

        self._sync_scale_options(force_default=force_default)
    



class DataTab(QWidget):
    def __init__(self):
        super().__init__()
        self.nk, self.catalog = build_catalog()
        self._watcher = None
        self._build_ui(); self.refresh_catalog(select_first=True)
    def _build_ui(self):
        outer = QVBoxLayout(self); outer.setContentsMargins(10, 10, 10, 10); outer.setSpacing(8)
        top = QHBoxLayout(); self.search = QLineEdit(); self.search.setPlaceholderText('Search material key'); self.refresh_btn = QPushButton('Refresh materials'); self.refresh_btn.clicked.connect(self.refresh_catalog); top.addWidget(QLabel('Catalog')); top.addWidget(self.search, 1); top.addWidget(self.refresh_btn); outer.addLayout(top)
        splitter = QSplitter(Qt.Horizontal); outer.addWidget(splitter, 1)
        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(6); self.list_model = QStringListModel([], self); self.proxy = FilterProxy(self); self.proxy.setSourceModel(self.list_model); self.list = QListView(); self.list.setModel(self.proxy); self.list.setEditTriggers(QAbstractItemView.NoEditTriggers); self.list.setSelectionMode(QAbstractItemView.SingleSelection); self.search.textEdited.connect(self.proxy.setNeedle); lv.addWidget(self.list); splitter.addWidget(left)
        right = QWidget(); rv = QVBoxLayout(right); rv.setContentsMargins(0, 0, 0, 0); rv.setSpacing(8); self.title = QLabel('Select a material key'); self.title.setStyleSheet('font-size:10pt; font-weight:600;'); rv.addWidget(self.title)
        self.fig = Figure(figsize=(8.2, 5.8)); self.canvas = FigureCanvas(self.fig); self.canvas.setMinimumHeight(380); rv.addWidget(self.canvas, 1)
        range_row = QHBoxLayout(); self.avg0 = QLineEdit(); self.avg1 = QLineEdit(); dv = QDoubleValidator(-1e9, 1e9, 6, self); self.avg0.setValidator(dv); self.avg1.setValidator(dv); self.avg0.setPlaceholderText('None'); self.avg1.setPlaceholderText('None'); self.avg_apply = QPushButton('Apply range'); self.avg_clear = QPushButton('Clear'); self.avg_apply.clicked.connect(self.update_current); self.avg_clear.clicked.connect(self._clear_avg_range); range_row.addWidget(QLabel('Average range (µm)')); range_row.addWidget(self.avg0); range_row.addWidget(QLabel('to')); range_row.addWidget(self.avg1); range_row.addWidget(self.avg_apply); range_row.addWidget(self.avg_clear); range_row.addStretch(1); rv.addLayout(range_row)
        self.stats_wrap = QWidget(); sg = QGridLayout(self.stats_wrap); sg.setContentsMargins(0, 0, 0, 0); sg.setHorizontalSpacing(8); sg.setVerticalSpacing(8)
        self.cards = {'n_avail': InfoCard('n available'), 'n_max': InfoCard('n max'), 'n_min': InfoCard('n min'), 'n_avg': InfoCard('Average n'), 'k_avail': InfoCard('k available'), 'k_max': InfoCard('k max'), 'k_min': InfoCard('k min'), 'k_avg': InfoCard('Average k')}
        order = [('n_avail', 0, 0), ('n_max', 0, 1), ('n_min', 0, 2), ('n_avg', 0, 3), ('k_avail', 1, 0), ('k_max', 1, 1), ('k_min', 1, 2), ('k_avg', 1, 3)]
        for name, r, c in order: sg.addWidget(self.cards[name], r, c)
        for c in range(4): sg.setColumnStretch(c, 1)
        rv.addWidget(self.stats_wrap)
        splitter.addWidget(right); splitter.setSizes([320, 1180]); self.list.selectionModel().currentChanged.connect(self._on_pick)
    def _clear_avg_range(self): self.avg0.clear(); self.avg1.clear(); self.update_current()
    def refresh_catalog(self, select_first=False):
        current = self.current_key() if not select_first else None; self.nk, self.catalog = build_catalog(); keys = [k for k in self.catalog if _vm_parse(k) is None]; self.list_model.setStringList(keys)
        if current and current in keys: self.select_key(current)
        elif keys: self.select_key(keys[0])
    def current_key(self):
        idx = self.list.currentIndex()
        if not idx.isValid(): return None
        src = self.proxy.mapToSource(idx)
        return self.list_model.data(src, Qt.DisplayRole)
    def select_key(self, key):
        items = self.list_model.stringList()
        if key not in items: return
        row = items.index(key); src = self.list_model.index(row, 0); idx = self.proxy.mapFromSource(src); self.list.setCurrentIndex(idx); self.list.scrollTo(idx); self.update_current()
    def _on_pick(self, cur, prev): self.update_current()
    def _set_card(self, name, text): self.cards[name].setText(text)
    def _extrema_text(self, wl, y):
        if wl.size == 0 or y.size == 0: return None
        ymax = float(np.nanmax(y)); ymin = float(np.nanmin(y)); wl_max = wl[np.isclose(y, ymax)]; wl_min = wl[np.isclose(y, ymin)]; fmt = lambda a: ', '.join(f'{float(v):.4g}' for v in a[:4]) + (' ...' if a.size > 4 else '')
        return ymax, ymin, fmt(wl_max), fmt(wl_min)
    def _avg_text(self, wl, y, lo, hi):
        if wl.size == 0 or y.size == 0: return 'Not available'
        if lo is None or hi is None: return 'Range = None'
        m = (wl >= lo) & (wl <= hi)
        if not np.any(m): return f'No data in {lo:.4g}–{hi:.4g} µm'
        return f'{float(np.mean(y[m])):.6g} over {lo:.4g}–{hi:.4g} µm'
    def _parse_avg_range(self, wl_all):
        a = self.avg0.text().strip(); b = self.avg1.text().strip()
        if not a and not b: return None, None
        if not a or not b: raise ValueError('Enter both average-range limits or leave both blank.')
        lo, hi = float(a), float(b)
        if hi < lo: raise ValueError('Average-range upper limit must be >= lower limit.')
        wlmin, wlmax = float(np.min(wl_all)), float(np.max(wl_all))
        if lo < wlmin or hi > wlmax: raise ValueError(f'Average range must stay within {wlmin:.4g}–{wlmax:.4g} µm.')
        return lo, hi
    def _reset_cards(self):
        for card in self.cards.values(): card.setText('-')
    def update_current(self):
        key = self.current_key()
        if not key: return
        self.title.setText(key)
        try:
            wn, nv, wk, kv = load_key_data(self.nk, key); arrs = [a for a in (wn, wk) if np.asarray(a).size]
            if not arrs: raise ValueError('No wavelength data found.')
            wl_all = np.unique(np.concatenate(arrs)); lo, hi = self._parse_avg_range(wl_all); self._plot_key(key, wn, nv, wk, kv)
            if wn.size:
                self._set_card('n_avail', f'{float(wn[0]):.4g}–{float(wn[-1]):.4g} µm\n{wn.size} points')
                nmax, nmin, wlmax, wlmin = self._extrema_text(wn, nv)
                self._set_card('n_max', f'{nmax:.6g}\nat {wlmax} µm')
                self._set_card('n_min', f'{nmin:.6g}\nat {wlmin} µm')
            else:
                self._set_card('n_avail', 'None'); self._set_card('n_max', 'Not available'); self._set_card('n_min', 'Not available')
            if wk.size:
                self._set_card('k_avail', f'{float(wk[0]):.4g}–{float(wk[-1]):.4g} µm\n{wk.size} points')
                kmax, kmin, wlmax, wlmin = self._extrema_text(wk, kv)
                self._set_card('k_max', f'{kmax:.6g}\nat {wlmax} µm')
                self._set_card('k_min', f'{kmin:.6g}\nat {wlmin} µm')
            else:
                self._set_card('k_avail', 'None'); self._set_card('k_max', 'Not available'); self._set_card('k_min', 'Not available')
            self._set_card('n_avg', self._avg_text(wn, nv, lo, hi)); self._set_card('k_avg', self._avg_text(wk, kv, lo, hi))
        except Exception as e:
            self._reset_cards(); self._set_card('n_avail', str(e)); self.fig.clear(); self.canvas.draw_idle()
    def _plot_key(self, key, wn, nv, wk, kv):
        self.fig.clear(); ax = self.fig.add_subplot(111); has_n = wn.size > 0; has_k = wk.size > 0
        if has_n: ax.plot(wn, nv, linewidth=1.8, label='n')
        if has_k: ax.plot(wk, kv, linewidth=1.8, linestyle='--', label='k')
        ax.set_xlabel('Wavelength, µm'); ax.set_ylabel('n, k'); xs = [a for a in (wn, wk) if np.asarray(a).size]
        if xs: ax.set_xlim(float(np.min(np.concatenate(xs))), float(np.max(np.concatenate(xs))))
        ys = [a for a in (nv, kv) if np.asarray(a).size]
        if ys:
            y = np.concatenate(ys); ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y)); pad = 0.08 * max(ymax - ymin, 1e-6); ax.set_ylim(ymin - pad, ymax + pad)
        ax.grid(True, alpha=0.3); ax.set_title(key, fontsize=11, pad=8)
        if has_n or has_k: ax.legend(loc='best')
        self.fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.92); self.canvas.draw_idle()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        normalize_qt_app_font()
        self.setWindowTitle('Optical Mapping UI - Mapping Release')

        tabs = QTabWidget()
        self.map_tab = MapTab()
        self.data_tab = DataTab()

        tabs.addTab(self.map_tab, 'Mapping')
        tabs.addTab(self.data_tab, 'Data')
        self.setCentralWidget(tabs)

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._thaw_updates)

        self._freeze = QLabel(self)
        self._freeze.hide()
        self._freeze.setScaledContents(True)
        self._freeze.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._resize_frozen = False

        unchecked_svg, checked_svg = ensure_checkbox_icons()
        self.setStyleSheet(f"""
            QWidget {{ background:#ececec; color:#222; font-size:9pt; }}
            QGroupBox, QFrame#LayerCard, QScrollArea, QTabWidget::pane, QListView, QFrame#InfoCard {{ background:#f5f5f5; border:1px solid #cfcfcf; border-radius:8px; }}
            QFrame#LayerCard {{ background:#f5f5f5; border:1px solid #cfcfcf; border-radius:8px; }}
            QScrollArea#LayerScroll, QWidget#LayerScrollViewport, QWidget#LayerWrap {{ background:#ececec; border:0px; }}
            QLineEdit, QToolButton, QPushButton {{ background:#fbfbfb; border:1px solid #c6c6c6; border-radius:6px; padding:6px; }}
            QComboBox {{ background:#fbfbfb; border:1px solid #c6c6c6; border-radius:6px; padding:4px 34px 4px 8px; min-height:32px; }}
            QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right; width: 34px; border-left:1px solid #c6c6c6; background:#fbfbfb; }}
            QComboBox::down-arrow {{ image:none; width:0px; height:0px; }}
            QComboBox QAbstractItemView {{ background:#fbfbfb; border:1px solid #c6c6c6; }}
            QLabel#RangeInfo {{ color:#444; padding:4px 2px 0 2px; }}
            QSpinBox {{ background:#fbfbfb; border:1px solid #c6c6c6; border-radius:6px; padding:2px 18px 2px 6px; }}
            QPushButton {{ min-height:30px; }}
            QTabBar::tab {{ background:#dddddd; border:1px solid #c8c8c8; padding:8px 16px; margin-right:2px; border-top-left-radius:6px; border-top-right-radius:6px; }}
            QTabBar::tab:selected {{ background:#f7f7f7; }}
            QCheckBox {{ color:#111; spacing:6px; }}
            QCheckBox::indicator {{ width:16px; height:16px; }}
            QCheckBox::indicator:unchecked {{ image: url({unchecked_svg}); }}
            QCheckBox::indicator:checked {{ image: url({checked_svg}); }}
            QLabel#InfoCardTitle {{ font-size:9pt; font-weight:600; color:#555; }}
            QLabel#InfoCardValue {{ font-size:10pt; }}
        """)

        QTimer.singleShot(0, self._set_initial_compact_size)

    def _set_initial_compact_size(self):
        self.adjustSize()
        screen = QApplication.primaryScreen().availableGeometry()
        w = min(max(self.sizeHint().width(), 1260), screen.width() - 80)
        h = min(930, screen.height() - 80)
        self.setMinimumWidth(w)
        self.resize(w, h)
        
    def refresh_all_materials(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if hasattr(self, 'map_tab'):
                self.map_tab.refresh_catalog()
            if hasattr(self, 'data_tab'):
                self.data_tab.refresh_catalog()
        finally:
            QApplication.restoreOverrideCursor()

    def _freeze_updates(self):
        c = self.centralWidget()
        if c is None or self._resize_frozen:
            return
        self._freeze.setPixmap(c.grab())
        self._freeze.setGeometry(c.geometry())
        self._freeze.show()
        self._freeze.raise_()
        c.setUpdatesEnabled(False)
        self._resize_frozen = True

    def resizeEvent(self, event):
        self._freeze_updates()
        super().resizeEvent(event)
        c = self.centralWidget()
        if self._resize_frozen and c is not None:
            self._freeze.setGeometry(c.geometry())
        self._resize_timer.start(140)

    def _thaw_updates(self):
        c = self.centralWidget()
        if c is not None and self._resize_frozen:
            c.setUpdatesEnabled(True)
            self._freeze.hide()
            c.repaint()
            self._resize_frozen = False

def main():
    qInstallMessageHandler(qt_message_handler)
    app = QApplication(sys.argv); w = MainWindow(); w.show(); sys.exit(app.exec())
if __name__ == '__main__':
    main()
