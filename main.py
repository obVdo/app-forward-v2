"""
app-forward-v2: Compute forward solution for MEG/EEG source reconstruction.

Inputs : sensor data (epochs or raw FIF), source space (src.fif from app-source-space-v2),
         trans.fif (from app-coreg-v2), bem-sol.fif (from app-bem-v2).
Outputs: forward-fwd.fif
"""

import os
import sys

# Resolve brainlife_utils — try local copy first, then parent monorepo
app_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(app_dir)
for search_path in [app_dir, parent_dir]:
    if os.path.isdir(os.path.join(search_path, 'brainlife_utils')):
        sys.path.insert(0, search_path)
        break

from brainlife_utils import (
    setup_matplotlib_backend,
    load_config,
    ensure_output_dirs,
    add_info_to_product,
    add_image_to_product,
    create_product_json,
)

setup_matplotlib_backend()
import matplotlib.pyplot as plt

import mne

# == SETUP ==
ensure_output_dirs('out_dir', 'out_figs', 'out_report')
report_items = []

# == LOAD CONFIG ==
config = load_config()

# == LOAD SENSOR DATA (for channel info only — any data type works) ==
# Priority: evoked > epochs > raw (most to least downstream/processed).
# Brainlife config keys: evoked='evoked', epochs='epo', raw='mne'
epochs_file = config.get('epo') or None
raw_file    = config.get('mne') or None
evoked_file = config.get('evoked') or None

info = None
try:
    if evoked_file and os.path.isfile(evoked_file):
        evoked = mne.read_evokeds(evoked_file)[0]
        info   = evoked.info
        add_info_to_product(report_items, f"Loaded evoked: {len(info['ch_names'])} channels", "info")
    elif epochs_file and os.path.isfile(epochs_file):
        data = mne.read_epochs(epochs_file, preload=False)
        info = data.info
        add_info_to_product(report_items, f"Loaded epochs: {len(data.ch_names)} channels", "info")
    elif raw_file and os.path.isfile(raw_file):
        info = mne.io.read_info(raw_file)
        add_info_to_product(report_items, f"Loaded raw: {len(info['ch_names'])} channels", "info")
    else:
        add_info_to_product(
            report_items,
            "FATAL: No sensor data found. Set 'evoked', 'epochs', or 'raw' in config.json.",
            "error"
        )
        create_product_json(report_items)
        sys.exit(1)
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not load sensor data: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == DETECT MODALITY ==
ch_types  = info.get_channel_types()
meg_types = {'mag', 'grad', 'ref_meg'}
eeg_count = sum(1 for t in ch_types if t == 'eeg')
meg_count = sum(1 for t in ch_types if t in meg_types)

if meg_count > 0 and eeg_count > 0:
    modality = 'meeg'
elif meg_count > 0:
    modality = 'meg'
elif eeg_count > 0:
    modality = 'eeg'
else:
    add_info_to_product(
        report_items,
        f"FATAL: No MEG or EEG channels found. Types present: {set(ch_types)}",
        "error"
    )
    create_product_json(report_items)
    sys.exit(1)

use_meg = modality in ('meg', 'meeg')
use_eeg = modality in ('eeg', 'meeg')

add_info_to_product(
    report_items,
    f"Modality: {modality} | EEG: {eeg_count} | MEG: {meg_count}",
    "info"
)

# == LOAD SOURCE SPACE ==
# Brainlife maps raw source_space datatype to config key 'output' (a directory).
# Find the src.fif file inside that directory.
_src_dir  = config.get('output') or None
src_file  = None
if _src_dir and os.path.isdir(_src_dir):
    for _f in os.listdir(_src_dir):
        if _f.endswith('.fif') and 'src' in _f:
            src_file = os.path.join(_src_dir, _f)
            break
    if not src_file:
        # fallback: any .fif in the directory
        _fifs = [f for f in os.listdir(_src_dir) if f.endswith('.fif')]
        if _fifs:
            src_file = os.path.join(_src_dir, _fifs[0])

try:
    if src_file:
        src = mne.read_source_spaces(src_file)
        add_info_to_product(
            report_items,
            f"Loaded source space: {sum(s['nuse'] for s in src)} sources",
            "info"
        )
    else:
        add_info_to_product(report_items, "FATAL: No source space file found. Set 'src' in config.json.", "error")
        create_product_json(report_items)
        sys.exit(1)
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not load source space: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == LOAD TRANS AND BEM ==
# Brainlife config keys: trans='trans', bem='fif' (meg/fif datatype)
trans_file = config.get('trans') or None
bem_file   = config.get('fif') or None
mindist    = float(config.get('mindist') or 5.0)

if trans_file and not os.path.isfile(trans_file):
    trans_file = None
if bem_file and not os.path.isfile(bem_file):
    bem_file = None

# == COMPUTE FORWARD SOLUTION ==
try:
    if trans_file and bem_file:
        add_info_to_product(report_items, "Computing forward solution (trans + BEM)...", "info")
        bem_sol   = mne.read_bem_solution(bem_file)
        trans_obj = mne.read_trans(trans_file)
        fwd = mne.make_forward_solution(
            info, trans=trans_obj, src=src, bem=bem_sol,
            meg=use_meg, eeg=use_eeg,
            mindist=mindist, n_jobs=1, verbose=True
        )

    elif modality == 'eeg':
        # EEG fallback: sphere model
        add_info_to_product(
            report_items,
            "No trans/BEM — using sphere model for EEG (fallback, less accurate).",
            "warning"
        )
        sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.095)
        fwd = mne.make_forward_solution(
            info, trans=None, src=src, bem=sphere,
            meg=False, eeg=True,
            mindist=mindist, n_jobs=1, verbose=True
        )

    else:
        add_info_to_product(
            report_items,
            "FATAL: MEG forward solution requires trans.fif and bem-sol.fif.",
            "error"
        )
        create_product_json(report_items)
        sys.exit(1)

    add_info_to_product(
        report_items,
        f"Forward solution: {fwd['nsource']} sources, "
        f"{'fixed' if fwd['surf_ori'] else 'free'} orientation",
        "info"
    )
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Forward solution failed: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == SAVE OUTPUT ==
fwd_path = os.path.join('out_dir', 'fwd.fif')
try:
    mne.write_forward_solution(fwd_path, fwd, overwrite=True)
    add_info_to_product(report_items, f"Saved: {fwd_path}", "info")
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not save forward solution: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == QC FIGURE — sensor topomap ==
try:
    fig_sensors = mne.viz.plot_sensors(info, show_names=False, show=False)
    fig_path = os.path.join('out_figs', 'forward_sensors.png')
    fig_sensors.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig_sensors)
    add_image_to_product(report_items, 'Sensor Layout', filepath=fig_path)
except Exception as e:
    add_info_to_product(report_items, f"Could not plot sensors: {e}", "warning")

# == SAVE REPORT ==
report = mne.Report(title='Forward Solution Report')
report.save(os.path.join('out_report', 'report.html'), overwrite=True)

add_info_to_product(report_items, "Forward solution computed successfully.", "success")
create_product_json(report_items)
print("Done.")
