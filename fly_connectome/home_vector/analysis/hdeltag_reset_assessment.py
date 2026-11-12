"""Offline measurement assessment only: no reset dynamics, fitting, or simulation."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..__main__ import begin, fail, finish
from ..data import digest, verify_bundle, write_json


def extract_mean_curve(rgb, calibration):
    """Sample only a manually isolated black trace; absent pixels stay missing."""
    c = calibration
    x0, x1 = c['x_pixels']
    t0, t1 = c['x_seconds']
    y0, y1 = c['y_pixels']
    r0, r1 = c['y_hz']
    lo, hi = c['crop_y']
    def rate(y):
        return float(r0 + (y - y0) * (r1 - r0) / (y1 - y0))
    rows = []
    for seconds in c['sample_seconds']:
        x = x0 + (seconds - t0) * (x1 - x0) / (t1 - t0)
        xi = round(x)
        half = c['half_strip_width_pixels']
        strip = rgb[lo:hi, xi-half:xi+half+1]
        ys = np.nonzero((strip.max(axis=2) < c['dark_threshold']).any(axis=1))[0] + lo
        row = dict(time_s=seconds, x_pixel=x, status='missing', y_pixel=None,
                   rate_hz=None, envelope_low_hz=None, envelope_high_hz=None)
        if len(ys):
            center = float(np.median(ys))
            pad = c['envelope_padding_pixels']
            bounds = [rate(ys.min()-pad), rate(ys.max()+pad)]
            row.update(status='extracted', y_pixel=center, rate_hz=rate(center),
                       envelope_low_hz=min(bounds), envelope_high_hz=max(bounds))
        rows.append(row)
    return rows


def affine_normalize(values, before, after):
    """Remove a fixed gain/offset using explicit anchors, not independent curve extrema."""
    if not np.isfinite([before, after]).all() or before == after:
        raise ValueError('Need distinct finite before/after anchors.')
    return (np.asarray(values, dtype=float) - after) / (before - after)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assess(args):
    pdf, model, calibration_path, output = [Path(getattr(args, k)).resolve()
                                           for k in ('pdf', 'model', 'calibration', 'output')]
    if output == model or output.is_relative_to(model) or model.is_relative_to(output):
        raise ValueError('Output must be separate from the model archive.')
    if output.exists():
        raise FileExistsError(output)
    if verify_bundle(model)['kind'] != 'home_vector_repeated_loops_v1':
        raise ValueError('Require the completed two-loop probe bundle.')
    c = json.loads(calibration_path.read_text())
    if digest(pdf) != c['source_pdf_sha256']:
        raise ValueError('PDF hash differs from the calibrated thesis.')
    inputs = {str(p): digest(p) for p in [pdf, calibration_path, *[p for p in model.iterdir() if p.is_file()]]}
    output, manifest = begin(output, 'home_vector_hdeltag_reset_assessment_v1', model=str(model),
                             calibration=str(calibration_path), pdf=str(pdf))
    manifest.update(immutable_input_sha256=inputs, analysis_script_sha256=digest(__file__),
                    simulation=False, refitting=False, reset_implemented=False)
    write_json(output / 'manifest.json', manifest)
    try:
        subprocess.run(['pdftoppm', '-f', str(c['pdf_page']), '-l', str(c['pdf_page']),
                        '-scale-to', str(c['render_scale_to']), '-png', '-singlefile',
                        str(pdf), str(output / 'source_page')], check=True, capture_output=True)
        rgb = np.asarray(Image.open(output / 'source_page.png').convert('RGB'))
        if list(rgb.shape[1::-1]) != c['image_size']:
            raise ValueError('Rendered page dimensions differ from calibrated page.')
        points = extract_mean_curve(rgb, c)
        if not all(r['status'] == 'extracted' for r in points):
            raise ValueError('A calibrated sample is missing; inspect the source before proceeding.')
        write_csv(output / 'digitized_spike_mean.csv', points)
        pre = [r['rate_hz'] for r in points if r['time_s'] < 0]
        early = [r['rate_hz'] for r in points if 40 <= r['time_s'] <= 60]
        summary = dict(figure=c['figure'], pdf_sha256=digest(pdf),
            pre_sample_median_hz=float(np.median(pre)),
            early_40_60s_sample_median_hz=float(np.median(early)),
            coarse_residual_ratio=float(np.median(early) / np.median(pre)),
            seconds_per_horizontal_pixel=(c['x_seconds'][1]-c['x_seconds'][0])/(c['x_pixels'][1]-c['x_pixels'][0]),
            fast_reset_time_constant_identified=False,
            limitations=['Pooled single-cell spike mean, not simultaneous population baseline.',
                'Early samples are already recovering, not the instantaneous post-sugar minimum.',
                'Pixel envelopes are not biological uncertainty or trial-level replicates.',
                'No calibration from model units to Hz is estimated.',
                'No bump amplitude is inferred without an anatomical phase map.',
                'No reset is fitted or implemented.'])
        with (model / 'neurons.csv').open() as stream:
            neurons = list(csv.DictReader(stream))
        selected = [i for i, r in enumerate(neurons) if r['cell_type'] == 'hDeltaG']
        if len(selected) != 8:
            raise ValueError('Require the eight modeled hDeltaG neurons.')
        summary['model_hdeltag_ids'] = [int(neurons[j]['body_id']) for j in selected]
        rows = []
        for name in ('path_08', 'path_09', 'path_14', 'path_15'):
            with np.load(model / f'{name}.npz', allow_pickle=False) as saved:
                rates, time = saved['rates'][:, selected].astype(float), saved['time_s']
            for seconds in (0., 1., 8., 16.):
                tick = int(np.argmin(np.abs(time - seconds)))
                if abs(time[tick] - seconds) > 1e-9:
                    raise ValueError('Missing prescribed comparison time.')
                values = rates[tick]
                rows.append(dict(path=name, time_s=seconds, mean_model_activity=float(values.mean()),
                    cross_cell_rms_about_mean=float(np.sqrt(np.mean((values-values.mean())**2))),
                    maximum_model_activity=float(values.max()), sugar_event_present=False))
        write_csv(output / 'current_model_hdeltag.csv', rows)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
        t = np.array([r['time_s'] for r in points]); y = np.array([r['rate_hz'] for r in points])
        axes[0].errorbar(t, y, yerr=[y-[r['envelope_low_hz'] for r in points],
                        np.array([r['envelope_high_hz'] for r in points])-y], fmt='o-', ms=3)
        axes[0].axvline(0, color='gray', ls=':')
        axes[0].set(xlabel='Time relative to sugar (s)', ylabel='Pooled spike mean (Hz)',
                    title='Janke Fig. 3.4b: coarse curve samples')
        for name in ('path_08', 'path_09', 'path_14', 'path_15'):
            rs = [r for r in rows if r['path'] == name]
            axes[1].plot([r['time_s'] for r in rs], [r['mean_model_activity'] for r in rs], 'o-', label=name)
        axes[1].set(xlabel='Elapsed movement time (s)', ylabel='Mean hDeltaG model activity',
                    title='Existing model: no sugar input, no reset')
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=.2)
        for extension in ('png', 'pdf'):
            fig.savefig(output / f'measurement_assessment.{extension}', dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(rgb)
        ax.scatter([r['x_pixel'] for r in points], [r['y_pixel'] for r in points],
                   facecolors='none', edgecolors='red', s=30, lw=.8)
        ax.set(xlim=(420, 625), ylim=(630, 425), title='Sample overlay on Fig. 3.4b; points are displayed-mean estimates')
        fig.savefig(output / 'digitization_overlay.png', dpi=160, bbox_inches='tight')
        plt.close(fig)
        write_json(output / 'assessment.json', summary)
        shutil.copyfile(calibration_path, output / 'calibration.json')
        shutil.copyfile(__file__, output / 'analysis_script.py')
        (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
            '<title>hDeltaG reset measurement assessment</title><style>body{font:16px system-ui;'
            'max-width:1200px;margin:2em auto}img{max-width:100%}</style>'
            '<h1>hDeltaG reset: physiology and current model</h1>'
            '<p>Different time axes and units are intentional. No model-to-Hz conversion, '
            'reset fit, simulation or decoding optimization.</p>'
            '<p><a href="digitized_spike_mean.csv">Displayed mean coordinates</a> · '
            '<a href="current_model_hdeltag.csv">Existing model measurements</a> · '
            '<a href="assessment.json">Summary and limitations</a></p>'
            '<img src="measurement_assessment.png" alt="Separate physiology and model measurements">'
            '<img src="digitization_overlay.png" alt="Coordinate extraction overlay">')
        if any(digest(path) != sha for path, sha in inputs.items()):
            raise ValueError('An immutable input changed during assessment.')
        finish(output, manifest)
    except BaseException as error:
        fail(output, manifest, error)
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pdf', 'model', 'calibration', 'output'):
        parser.add_argument(f'--{name}', required=True)
    return assess(parser.parse_args(argv))


if __name__ == '__main__':
    main()
