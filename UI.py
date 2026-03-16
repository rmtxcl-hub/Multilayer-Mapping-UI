from pathlib import Path
import sys, traceback, json, numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QObject, QStringListModel, QSortFilterProxyModel, QFileSystemWatcher, QTimer
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import *
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patheffects as pe
from nkwrap import NK, _vm_parse
from TMM import TMM
import map_mod as mm
import contour
import submask
import simplepeak

BASE = Path(__file__).resolve().parent
DB = BASE / 'data_f.sqlite'
TXT_DIR = BASE / 'txt'
TAGS = BASE / 'material_keys_tagged.txt'
SETTINGS_JSON = BASE / 'ui_settings.json'
TXT_DIR.mkdir(exist_ok=True)

MAP_CMAP = 'inferno'
MASK_CFG = dict(thr=None, thr_hi='adaptive', thr_hi_alpha=0.7, thr_hi_base='median', thr_lo=None, thr_lo_pct='auto', sig=1.6, close=4, single=None, peak=None)
SUBMASK_CFG = dict(close=2, fill=True)
DEFAULT_UI_SETTINGS = {'tol_im': 0.0, 'tol_re': 0.0, 'clip_im_delta': 60.0, 'cover_n': 1.0, 'cover_k': 0.0, 'save_aspect_w': 16.0, 'save_aspect_h': 9.0}

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
    return d

def save_ui_settings(vals):
    try:
        SETTINGS_JSON.write_text(json.dumps(vals, indent=2), encoding='utf-8')
    except Exception:
        pass

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
        self._btn.setStyleSheet('QToolButton { background: transparent; border: none; color: #111; font-size: 14px; font-weight: 600; padding: 0px; margin: 0px; }')
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
        self.lab = QLabel()
        self.key = MaterialBox(items)
        self.key.setMinimumHeight(40)
        self.thk = QLineEdit('0.20')
        self.thk.setValidator(QDoubleValidator(0.0, 1e9, 6, self))
        self.thk.setFixedWidth(90)
        self.thk.setMinimumHeight(38)
        u = QLabel('µm')
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)
        v.addWidget(self.lab)
        v.addWidget(QLabel('Material key'))
        v.addWidget(self.key)
        h = QHBoxLayout()
        h.addWidget(self.thk)
        h.addWidget(u)
        h.addStretch(1)
        v.addWidget(QLabel('Thickness'))
        v.addLayout(h)
        self.rename(idx, total)
        self.setMinimumWidth(250)
    def rename(self, idx, total):
        if total == 1:
            t = 'Layer 1 (Top = Bottom)'
        elif idx == 0:
            t = 'Layer 1 (Top)'
        elif idx == total - 1:
            t = f'Layer {idx + 1} (Bottom)'
        else:
            t = f'Layer {idx + 1}'
        self.lab.setText(t)

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
            cfg = {'lam_um': f"{p['lam0']}-{p['lam1']}-{p['lamN']}", 'theta_rad': np.deg2rad(np.linspace(p['th0'], p['th1'], p['thN'])), 'pol': p['pol'], 'n0': complex(p['cover_n'], p['cover_k']), 'film_keys': p['films'], 'sub_key': p['sub'], 'dL': p['dL'], 'cover_n': p['cover_n'], 'cover_k': p['cover_k']}
            md = mm.calc(tmm, cfg, mode=p['mode'], nk=p['nk'], use_gpu=False)
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
        self._watcher = QFileSystemWatcher([str(TXT_DIR)], self)
        self._watcher.directoryChanged.connect(self.refresh_catalog)
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
        self.layer_wrap = QWidget(); self.layer_bar = QHBoxLayout(self.layer_wrap); self.layer_bar.setContentsMargins(0, 0, 0, 0); self.layer_bar.setSpacing(8); self.layer_bar.addStretch(1)
        self.layer_scroll = QScrollArea(); self.layer_scroll.setWidgetResizable(True); self.layer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded); self.layer_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff); self.layer_scroll.setWidget(self.layer_wrap); self.layer_scroll.setMinimumHeight(175); self.layer_scroll.setMaximumHeight(220); self.layer_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        outer.addWidget(self.layer_scroll)
        box = QGroupBox(); g = QGridLayout(box)
        self.mode = ArrowComboBox(['E', 'R'])
        self.pol = ArrowComboBox(['both', 's', 'p'])
        self.lam0 = QLineEdit('3'); self.lam1 = QLineEdit('14'); self.lamN = QLineEdit('501')
        self.th0 = QLineEdit('0'); self.th1 = QLineEdit('80'); self.thN = QLineEdit('301')
        dv = QDoubleValidator(-1e9, 1e9, 6, self); iv = QIntValidator(2, 10**6, self)
        for w in (self.lam0, self.lam1, self.th0, self.th1): w.setValidator(dv)
        for w in (self.lamN, self.thN): w.setValidator(iv)
        g.addWidget(QLabel('Map'), 0, 0); g.addWidget(self.mode, 0, 1); g.addWidget(QLabel('Polarization'), 0, 2); g.addWidget(self.pol, 0, 3)
        g.addWidget(QLabel('λ min (µm)'), 1, 0); g.addWidget(self.lam0, 1, 1); g.addWidget(QLabel('λ max (µm)'), 1, 2); g.addWidget(self.lam1, 1, 3); g.addWidget(QLabel('λ points'), 1, 4); g.addWidget(self.lamN, 1, 5)
        g.addWidget(QLabel('θ min (deg)'), 2, 0); g.addWidget(self.th0, 2, 1); g.addWidget(QLabel('θ max (deg)'), 2, 2); g.addWidget(self.th1, 2, 3); g.addWidget(QLabel('θ points'), 2, 4); g.addWidget(self.thN, 2, 5)
        outer.addWidget(box); box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        act = QHBoxLayout()
        self.run_btn = QPushButton('Run'); self.save_btn = QPushButton('Save map'); self.save_btn.setEnabled(False); self.run_btn.clicked.connect(self.run); self.save_btn.clicked.connect(self.save_map)
        self.status = QLabel('Ready'); self.contour_cb = QCheckBox('Contour'); self.submask_cb = QCheckBox('Submask'); self.peak_cb = QCheckBox('Peak')
        for cb in (self.contour_cb, self.submask_cb, self.peak_cb): cb.toggled.connect(self.refresh_overlays)
        act.addWidget(self.run_btn); act.addWidget(self.save_btn); act.addSpacing(10); act.addWidget(self.status, 1); act.addWidget(self.contour_cb); act.addWidget(self.submask_cb); act.addWidget(self.peak_cb)
        outer.addLayout(act)
        self.fig = Figure(figsize=(8.5, 6.5)); self.ax = self.fig.add_subplot(111); self.canvas = FigureCanvas(self.fig); self.canvas.setMinimumHeight(260); self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); outer.addWidget(self.canvas, 1)
        self.range_info = QLabel('Layer data overlap — n: - | k: -'); self.range_info.setObjectName('RangeInfo'); self.range_info.setTextInteractionFlags(Qt.TextSelectableByMouse); outer.addWidget(self.range_info)
        self.layers = []; self.rebuild_layers(); self.refresh_catalog()
        if 'PI-PI' in self.catalog: self.sub_box.setCurrentText('PI-PI')
        elif 'AlN-Hoffmann' in self.catalog: self.sub_box.setCurrentText('AlN-Hoffmann')
        self.update_overlap_label()
    def rebuild_layers(self):
        prev = [(c.key.currentText().strip(), c.thk.text().strip()) for c in getattr(self, 'layers', [])]
        while self.layer_bar.count():
            item = self.layer_bar.takeAt(0); w = item.widget()
            if w: w.deleteLater()
        self.layers = []; n = self.layer_count.value()
        for i in range(n):
            c = LayerCard(i, n, self.catalog)
            if i < len(prev):
                key, thk = prev[i]
                if key: c.key.setCurrentText(key)
                if thk: c.thk.setText(thk)
            c.key.currentTextChanged.connect(self.update_overlap_label); c.key.lineEdit().editingFinished.connect(self.update_overlap_label)
            self.layers.append(c); self.layer_bar.addWidget(c)
        self.layer_bar.addStretch(1); self.update_overlap_label()
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
            self.settings = d.values(); save_ui_settings(self.settings)
    def _valid_key(self, k): return self._canonical_key_info(k)['valid']
    def _canonical_key(self, k):
        info = self._canonical_key_info(k)
        return info['base'] if info['valid'] else (k or '').strip()
    def payload(self):
        raw_films = [c.key.currentText().strip() for c in self.layers]; films = [self._canonical_key(k) for k in raw_films]; dL = [float(c.thk.text()) for c in self.layers]
        if any(not self._valid_key(k) for k in raw_films): raise ValueError('One or more layer keys are invalid.')
        sub_raw = self.sub_box.currentText().strip()
        if not self._valid_key(sub_raw): raise ValueError('Invalid substrate key.')
        sub = self._canonical_key(sub_raw); lam0, lam1, lamN = float(self.lam0.text()), float(self.lam1.text()), int(self.lamN.text()); th0, th1, thN = float(self.th0.text()), float(self.th1.text()), int(self.thN.text())
        if lam1 <= lam0 or th1 < th0: raise ValueError('Boundary values are invalid.')
        return {'films': films, 'dL': dL, 'sub': sub, 'mode': self.mode.currentText(), 'pol': self.pol.currentText(), 'lam0': lam0, 'lam1': lam1, 'lamN': lamN, 'th0': th0, 'th1': th1, 'thN': thN, 'nk': self.nk, **self.settings}
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
    def _compute_overlays(self, md):
        if self._overlay_cache is not None: return self._overlay_cache
        lam = np.asarray(md.lam_um, float); th = np.asarray(md.theta_deg, float); val = np.asarray(md.val, float); peak_kind = 'high' if str(md.mode).upper() in ('E', 'A') else 'low'
        lam2, th2, mask, thr_used, _ = contour.mask(val, lam, th, mode=md.mode, return_report=True, **MASK_CFG)
        pw = np.ones(np.asarray(lam2).size, float); aw = np.ones(np.asarray(th2).size, float)
        mv, mh, _ = submask.peak_masks(val, lam2, th2, mask, mode=md.mode, peak=MASK_CFG.get('peak') or peak_kind, thr_used=thr_used, return_debug=True, pw=pw, aw=aw, **SUBMASK_CFG)
        vmeta, hmeta, _ = simplepeak.locate_1d_peaks(val, lam2, th2, mask, mode=md.mode, peak=peak_kind, thr_used=thr_used, return_meta=True)
        self._overlay_cache = {'lam': lam2, 'theta': th2, 'contour_mask': mask, 'submask_v': mv, 'submask_h': mh, 'peak_v': vmeta, 'peak_h': hmeta}
        return self._overlay_cache
    def _draw_overlays(self, ax, md):
        if not self._overlay_enabled(): return
        try: ov = self._compute_overlays(md)
        except Exception:
            self.status.setText('Overlay error'); return
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
    def _map_limits(self, arr): return 0.0, 1.0
    def _draw_map(self, fig, md, title_fs):
        fig.clear(); gs = fig.add_gridspec(1, 2, width_ratios=[48, 1.35], left=0.075, right=0.925, bottom=0.14, top=0.88, wspace=0.12); ax = fig.add_subplot(gs[0, 0]); cax = fig.add_subplot(gs[0, 1]); vmin, vmax = self._map_limits(md.val)
        im = ax.pcolormesh(md.lam_um, md.theta_deg, md.val.T, shading='auto', vmin=vmin, vmax=vmax, cmap=MAP_CMAP)
        ax.set_xlabel('Wavelength (µm)', labelpad=4); ax.set_ylabel('Angle (deg)', labelpad=6); ax.set_xlim(float(np.min(md.lam_um)), float(np.max(md.lam_um))); ax.set_ylim(float(np.min(md.theta_deg)), float(np.max(md.theta_deg)))
        mode = 'E' if md.mode == 'E' else 'R'; mats = ' | '.join(md.meta.get('film_keys') or []); dL = ' | '.join(f'{float(v):.2f}' for v in (md.meta.get('dL') or [])); ax.set_title(f'{mode} map of {mats}\n{dL}', fontsize=title_fs, pad=6)
        self._draw_overlays(ax, md); cbar = fig.colorbar(im, cax=cax); cbar.set_label(mode, labelpad=6); return fig
    def plot_map(self, md): self._draw_map(self.fig, md, 11); self.canvas.draw_idle()
    def save_map(self):
        if self.last_md is None: return
        path, _ = QFileDialog.getSaveFileName(self, 'Save map', str(BASE / 'map.png'), 'PNG (*.png);;PDF (*.pdf);;SVG (*.svg)')
        if not path: return
        md = self.last_md; aw = max(float(self.settings.get('save_aspect_w', 16.0)), 0.1); ah = max(float(self.settings.get('save_aspect_h', 9.0)), 0.1)
        fig = Figure(figsize=(aw, ah), dpi=120); gs = fig.add_gridspec(1, 2, width_ratios=[48, 1.35], left=0.075, right=0.925, bottom=0.10, top=0.90, wspace=0.12); ax = fig.add_subplot(gs[0, 0]); cax = fig.add_subplot(gs[0, 1]); vmin, vmax = self._map_limits(md.val)
        im = ax.pcolormesh(md.lam_um, md.theta_deg, md.val.T, shading='auto', vmin=vmin, vmax=vmax, cmap=MAP_CMAP)
        ax.set_xlabel('Wavelength (µm)', labelpad=4); ax.set_ylabel('Angle (deg)', labelpad=6); ax.set_xlim(float(np.min(md.lam_um)), float(np.max(md.lam_um))); ax.set_ylim(float(np.min(md.theta_deg)), float(np.max(md.theta_deg)))
        mode = 'E' if md.mode == 'E' else 'R'; mats = ' | '.join(md.meta.get('film_keys') or []); dL = ' | '.join(f'{float(v):.2f}' for v in (md.meta.get('dL') or [])); ax.set_title(f'{mode} map of {mats}\n{dL}', fontsize=16)
        self._draw_overlays(ax, md); cbar = fig.colorbar(im, cax=cax); cbar.set_label(mode, labelpad=6); ext = Path(path).suffix.lower(); fig.savefig(path, dpi=120 if ext == '.png' else None, facecolor='white'); self.status.setText(f'Saved: {path}')

class DataTab(QWidget):
    def __init__(self):
        super().__init__()
        self.nk, self.catalog = build_catalog()
        self._watcher = QFileSystemWatcher([str(TXT_DIR)], self)
        self._watcher.directoryChanged.connect(self.refresh_catalog)
        self._build_ui(); self.refresh_catalog(select_first=True)
    def _build_ui(self):
        outer = QVBoxLayout(self); outer.setContentsMargins(10, 10, 10, 10); outer.setSpacing(8)
        top = QHBoxLayout(); self.search = QLineEdit(); self.search.setPlaceholderText('Search material key'); self.refresh_btn = QPushButton('Refresh materials'); self.refresh_btn.clicked.connect(self.refresh_catalog); top.addWidget(QLabel('Catalog')); top.addWidget(self.search, 1); top.addWidget(self.refresh_btn); outer.addLayout(top)
        splitter = QSplitter(Qt.Horizontal); outer.addWidget(splitter, 1)
        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(6); self.list_model = QStringListModel([], self); self.proxy = FilterProxy(self); self.proxy.setSourceModel(self.list_model); self.list = QListView(); self.list.setModel(self.proxy); self.list.setEditTriggers(QAbstractItemView.NoEditTriggers); self.list.setSelectionMode(QAbstractItemView.SingleSelection); self.search.textEdited.connect(self.proxy.setNeedle); lv.addWidget(self.list); splitter.addWidget(left)
        right = QWidget(); rv = QVBoxLayout(right); rv.setContentsMargins(0, 0, 0, 0); rv.setSpacing(8); self.title = QLabel('Select a material key'); self.title.setStyleSheet('font-size:14px; font-weight:600;'); rv.addWidget(self.title)
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
        super().__init__(); self.setWindowTitle('Optical Mapping UI'); self.resize(1500, 930); tabs = QTabWidget(); tabs.addTab(MapTab(), 'Mapping'); tabs.addTab(DataTab(), 'Data'); self.setCentralWidget(tabs)
        self._resize_timer = QTimer(self); self._resize_timer.setSingleShot(True); self._resize_timer.timeout.connect(self._thaw_updates); self._freeze = QLabel(self); self._freeze.hide(); self._freeze.setScaledContents(True); self._freeze.setAttribute(Qt.WA_TransparentForMouseEvents, True); self._resize_frozen = False
        unchecked_svg, checked_svg = ensure_checkbox_icons()
        self.setStyleSheet(f"""
            QWidget {{ background:#ececec; color:#222; font-size:12px; }}
            QGroupBox, QFrame#LayerCard, QScrollArea, QTabWidget::pane, QListView, QFrame#InfoCard {{ background:#f5f5f5; border:1px solid #cfcfcf; border-radius:8px; }}
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
            QLabel#InfoCardTitle {{ font-size:12px; font-weight:600; color:#555; }}
            QLabel#InfoCardValue {{ font-size:13px; }}
        """)
    def _freeze_updates(self):
        c = self.centralWidget()
        if c is None or self._resize_frozen: return
        self._freeze.setPixmap(c.grab()); self._freeze.setGeometry(c.geometry()); self._freeze.show(); self._freeze.raise_(); c.setUpdatesEnabled(False); self._resize_frozen = True
    def resizeEvent(self, event):
        self._freeze_updates(); super().resizeEvent(event); c = self.centralWidget()
        if self._resize_frozen and c is not None: self._freeze.setGeometry(c.geometry())
        self._resize_timer.start(140)
    def _thaw_updates(self):
        c = self.centralWidget()
        if c is not None and self._resize_frozen: c.setUpdatesEnabled(True); self._freeze.hide(); c.repaint(); self._resize_frozen = False

def main():
    app = QApplication(sys.argv); w = MainWindow(); w.show(); sys.exit(app.exec())
if __name__ == '__main__':
    main()
