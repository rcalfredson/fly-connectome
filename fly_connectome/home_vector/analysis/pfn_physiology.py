"""Digitize published PFNd curves; fit descriptive laws, never a circuit encoder.

Only the hash-identified Lu ED4 image is supported by the supplied calibration.
Missing/occluded traces stay missing. Pixel intervals are not biological CIs.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import label
from scipy.optimize import least_squares, nnls


MODELS = ('published_shape', 'proposed_shape', 'positive_only', 'signed', 'residual_floor')
EQUATIONS = {
    'published_shape': 'a * (1 + 5 * max(0, p))',
    'proposed_shape': 'a * max(0, 1 + 5 * p)',
    'positive_only': 'b + g * max(0, p)',
    'signed': 'max(0, b + g * p)',
    'residual_floor': 'f + max(0, b + g * p)',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def affine(values, pixels, coordinates):
    """Two tick centers define a linear axis, including truncated/reversed axes."""
    a, b = pixels
    c, d = coordinates
    if a == b:
        raise ValueError('Axis calibration needs distinct pixels.')
    return c + (np.asarray(values) - a) * (d - c) / (b - a)


def trace_mask(rgb, condition, variant=0):
    rgb = np.asarray(rgb, dtype=float)
    r, g, b = np.moveaxis(rgb, -1, 0)
    if condition == 'preferred':
        mask = (r > 160 - variant * 10) & (g > 95) & (b < 155 + variant * 10)
        mask &= (r - g > 15 - variant * 3) & (g - b > 35 - variant * 5)
    elif condition == 'antipreferred':
        mask = (rgb.max(axis=-1) - rgb.min(axis=-1) < 12 + variant * 3)
        mask &= (r > 125 - variant * 5) & (r < 195 + variant * 5)
    else:
        raise ValueError('Unknown heading condition.')
    # Remove isolated lettering/antialiasing; never fill a missing trace.
    components, n = label(mask, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, dtype=bool)
    for k in range(1, n + 1):
        ys, xs = np.nonzero(components == k)
        keep[k] = len(xs) >= 20 and np.ptp(xs) >= 15
    return keep[components]


def digitize(rgb, calibration, variant=0):
    points = []
    tolerance = calibration['pixel_tolerance']
    for panel in calibration['panels']:
        x0, x1 = panel['x_pixels']
        y0, y1 = panel['crop_y']
        # Include edge columns to recover colored endpoints beside black axes.
        left, right = int(x0), int(x1) + 1
        for condition in ('preferred', 'antipreferred'):
            mask = trace_mask(rgb[y0:y1, left:right], condition, variant)
            # Gray antialiased edges of the black horizontal axis can satisfy
            # the gray-curve color test. Treat this band as occluded, not data.
            if condition == 'antipreferred':
                axis_y = int(round(panel['y_pixels'][0])) - y0
                mask[max(0, axis_y-3):axis_y+4] = False
                # The vertical axis has gray antialiasing too. At p=-1 the
                # gray curve can be inseparable from it; leave that sample out.
                mask[:, :4] = False
            for p in np.arange(-1., 5.01, .5):
                xp = float(affine(p, panel['x_values'], panel['x_pixels']))
                xi = int(round(xp))
                lo, hi = max(left, xi - 1), min(right, xi + 2)
                ys = np.nonzero(mask[:, lo-left:hi-left])[0] + y0
                row = dict(cell=panel['cell'], condition=condition, projection_mm_s=float(p),
                           x_pixel=xp, status='missing_or_occluded', y_pixel=None,
                           rate_spikes_s=None, rate_low=None, rate_high=None,
                           projection_pixel_tolerance_mm_s=abs(float(affine(xp + tolerance,
                               panel['x_pixels'], panel['x_values'])) - p))
                if len(ys):
                    # Separate distant candidates are ambiguous, not averaged together.
                    unique = np.unique(ys)
                    if np.diff(unique).max(initial=0) > 6:
                        row['status'] = 'multiple_candidates'
                    else:
                        yp = float(np.median(ys))
                        rate = float(affine(yp, panel['y_pixels'], panel['y_values']))
                        bounds = affine([ys.min()-tolerance, ys.max()+tolerance],
                                        panel['y_pixels'], panel['y_values'])
                        row.update(status='extracted', y_pixel=yp, rate_spikes_s=max(0., rate),
                                   rate_low=max(0., float(min(bounds))),
                                   rate_high=max(0., float(max(bounds))))
                points.append(row)
    return points


def predict(model, parameters, x):
    x = np.asarray(x)
    if model == 'published_shape':
        return parameters[0] * (1 + 5 * np.maximum(0, x))
    if model == 'proposed_shape':
        return parameters[0] * np.maximum(0, 1 + 5 * x)
    if model == 'positive_only':
        return parameters[0] + parameters[1] * np.maximum(0, x)
    if model == 'signed':
        return np.maximum(0, parameters[0] + parameters[1] * x)
    if model == 'residual_floor':
        return parameters[0] + np.maximum(0, parameters[1] + parameters[2] * x)
    raise ValueError('Unknown response law.')


def fit_curve(model, x, y):
    """Unweighted descriptive fit to fixed-grid curve coordinates, not trials."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 4 or x.shape != y.shape or not np.isfinite([x, y]).all() or (y < 0).any():
        raise ValueError('Need at least four finite nonnegative curve samples.')
    if model in ('published_shape', 'proposed_shape'):
        basis = predict(model, [1.], x)
        parameters = np.array([max(0., float(basis @ y / (basis @ basis)))])
    elif model == 'positive_only':
        parameters = nnls(np.column_stack([np.ones(len(x)), np.maximum(0, x)]), y)[0]
    else:
        base = max(float(np.interp(0., x, y)), 1e-4)
        slope = max(float((y[-1] - y[0]) / (x[-1] - x[0])), 1e-4)
        starts = ([base, slope], [base, base*5], [base, slope*.1], [base, slope*5])
        if model == 'residual_floor':
            starts = [[fraction*base, (1-fraction)*base, gain]
                      for fraction in (.05, .5, .95) for gain in (slope, base*5)]
        results = [least_squares(lambda q: predict(model, q, x)-y, initial,
                                bounds=(0, np.inf), max_nfev=1000,
                                ftol=1e-10, xtol=1e-10, gtol=1e-10) for initial in starts]
        converged = [r for r in results if r.success]
        if not converged:
            raise RuntimeError('Response-law fit did not converge.')
        parameters = min(converged, key=lambda r: np.sum(r.fun**2)).x
    return parameters


def write_csv(path, rows):
    with Path(path).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(points):
    fits, predictions, sensitivity = [], [], []
    for cell in range(1, 15):
        for condition in ('preferred', 'antipreferred'):
            rows = [r for r in points if r['cell'] == cell and r['condition'] == condition
                    and r['status'] == 'extracted']
            x = np.array([r['projection_mm_s'] for r in rows])
            y = np.array([r['rate_spikes_s'] for r in rows])
            if len(x) < 5 or len(np.unique(x[x >= 0])) < 3:
                continue
            for model in MODELS:
                q = fit_curve(model, x, y)
                fitted = predict(model, q, x)
                held = np.array([predict(model, fit_curve(model, np.delete(x, k), np.delete(y, k)),
                                         x[k]) for k in range(len(x))])
                baseline = float(predict(model, q, 0.))
                gain = (float(5*q[0]) if model in ('published_shape', 'proposed_shape')
                        else float(q[-1]))
                threshold = None
                if model == 'proposed_shape':
                    threshold = -.2
                elif model in ('signed', 'residual_floor') and gain > 1e-8:
                    threshold = -float(q[-2])/gain
                fits.append(dict(cell=cell, condition=condition, model=model, n_points=len(x),
                    parameters=json.dumps(q.tolist()), baseline_at_zero_projection=baseline,
                    positive_gain=gain, floor_onset_mm_s=threshold,
                    floor_onset_within_sampled_range=(threshold is not None and x.min() < threshold < x.max()),
                    rmse=float(np.sqrt(np.mean((fitted-y)**2))),
                    leave_one_sample_out_rmse=float(np.sqrt(np.mean((held-y)**2)))))
                for row, pred, loo in zip(rows, fitted, held):
                    predictions.append(dict(cell=cell, condition=condition, model=model,
                        projection_mm_s=row['projection_mm_s'], observed_rate=row['rate_spikes_s'],
                        fitted_rate=float(pred), held_sample_prediction=float(loo)))
                # Deterministic extraction envelopes, not resampling biological observations.
                low = np.array([r['rate_low'] for r in rows])
                high = np.array([r['rate_high'] for r in rows])
                dx = np.array([r['projection_pixel_tolerance_mm_s'] for r in rows])
                for name, px, perturbed in [('low', x, low), ('high', x, high),
                        ('low_to_high', x, np.where(x <= 0, low, high)),
                        ('high_to_low', x, np.where(x <= 0, high, low)),
                        ('x_low', x-dx, y), ('x_high', x+dx, y)]:
                    qp = fit_curve(model, px, perturbed)
                    sensitivity.append(dict(cell=cell, condition=condition, model=model,
                        scenario=name, parameters=json.dumps(qp.tolist()),
                        baseline_at_zero_projection=float(predict(model, qp, 0.)),
                        rmse_against_central=float(np.sqrt(np.mean((predict(model, qp, x)-y)**2)))))
    return fits, predictions, sensitivity


def plot_overlay(rgb, calibration, points, output):
    fig, ax = plt.subplots(figsize=(13, 12))
    ax.imshow(rgb)
    for panel in calibration['panels']:
        for xp, xv in zip(panel['x_pixels'], panel['x_values']):
            yp = panel['y_pixels'][0]
            ax.plot(xp, yp, '+', color='cyan', ms=5)
        for yp, yv in zip(panel['y_pixels'], panel['y_values']):
            ax.plot(panel['x_pixels'][0], yp, '+', color='cyan', ms=5)
    for condition, color in [('preferred', 'red'), ('antipreferred', 'blue')]:
        rows = [r for r in points if r['condition'] == condition and r['status'] == 'extracted']
        ax.scatter([r['x_pixel'] for r in rows], [r['y_pixel'] for r in rows],
                   s=9, facecolors='none', edgecolors=color, linewidths=.7, label=condition)
    ax.set_xlim(50, 1290)
    ax.set_ylim(1900, 0)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_title('Digitized curve samples; cyan crosses = calibrated ticks')
    fig.savefig(output / 'digitization_overlay.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_fits(points, fits, output):
    colors = dict(published_shape='purple', proposed_shape='red', positive_only='gray',
                  signed='blue', residual_floor='green')
    for condition in ('preferred', 'antipreferred'):
        fig, axes = plt.subplots(7, 2, figsize=(11, 20))
        for cell, ax in enumerate(axes.flat, 1):
            rows = [r for r in points if r['cell'] == cell and r['condition'] == condition
                    and r['status'] == 'extracted']
            x = np.array([r['projection_mm_s'] for r in rows])
            y = np.array([r['rate_spikes_s'] for r in rows])
            ax.errorbar(x, y, yerr=[y-[r['rate_low'] for r in rows],
                         np.array([r['rate_high'] for r in rows])-y], fmt='ko', ms=3,
                        capsize=2, label='digitized ± pixel envelope')
            for fit in fits:
                if fit['cell'] == cell and fit['condition'] == condition:
                    grid = np.linspace(x.min(), x.max(), 200)
                    ax.plot(grid, predict(fit['model'], json.loads(fit['parameters']), grid),
                            color=colors[fit['model']], lw=1, label=fit['model'])
            ax.set_title(f'Cell {cell} — {condition}')
            ax.set_xlabel('Projected velocity (mm/s)')
            ax.set_ylabel('Firing rate (spikes/s)')
            ax.set_ylim(bottom=0)
            ax.axvline(0, color='black', alpha=.2)
        axes.flat[0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(output / f'{condition}_fits.png', dpi=140)
        fig.savefig(output / f'{condition}_fits.pdf')
        plt.close(fig)


def write_report(summary, output):
    lines = ['# PFNd plot digitization and descriptive response fits', '',
        'Source: [Lu et al. 2022, Extended Data Fig. 4a]('+summary['source_url']+').', '',
        'This is a comparison with digitized displayed curves, not a fit to raw recordings. '
        'No prescribed encoder, circuit parameter, checkpoint or archived result is changed.', '',
        f"Extracted {summary['points_extracted']} coordinates; retained "
        f"{summary['points_unresolved']} unresolved samples as missing. All 14 cells and both "
        'heading conditions have descriptive fits. No curve was excluded for a poor fit.', '',
        '## Extraction', '',
        'Tick centers and per-panel scales are in calibration.json. Cells 5 and 12 have '
        'truncated vertical axes (lower ticks 10 and 5 spikes/s). Sample the displayed lines '
        'on a fixed half-mm/s grid; original bin centers are not known. Some coordinates '
        'therefore lie on connecting segments. Color masks separate gold/gray curves; '
        'gray traces hidden by axes or the gold line remain missing. Review the overlay.', '',
        'The pixel envelope includes observed stroke extent in a three-column strip plus '
        'two vertical pixels. Six deterministic coordinate perturbations and two color-mask '
        'variants assess extraction sensitivity; they do not estimate biological uncertainty.', '',
        '## Response laws', '', '```text']
    lines += [f'{model}: {equation}' for model, equation in EQUATIONS.items()]
    lines += ['```', '', 'All parameters are nonnegative. a is a free rate scale; b and f '
        'have firing-rate units; g has rate-per-velocity units. Fits are separate for each '
        'cell and heading band, with equal weight per extracted coordinate. The first '
        'two models have one parameter, the next two have two, and the last has three.', '',
        '## Descriptive errors', '',
        '| Law | Preferred RMSE | Anti-preferred RMSE | Preferred held sample | Anti-preferred held sample |',
        '| --- | ---: | ---: | ---: | ---: |']
    for model in MODELS:
        a, b = summary['preferred'][model], summary['antipreferred'][model]
        lines.append(f"| {model} | {a['median_rmse']:.3f} | {b['median_rmse']:.3f} | "
                     f"{a['median_held_sample_rmse']:.3f} | {b['median_held_sample_rmse']:.3f} |")
    checks = summary['extraction_checks']
    lines += ['', 'Entries are medians across 14 curves, in spikes/s. Held samples use '
        'leave-one-coordinate-out refitting. Because samples can lie on connecting lines, '
        'this measures descriptive curve prediction, not independent experimental validation. '
        'Training error alone rewards flexibility; this table does not select a biological law.', '',
        '## Interpretation', '',
        f"Of {checks['preferred_samples_below_proposed_cutoff']} preferred-heading samples at "
        'p=-1 or -0.5 mm/s, '
        f"{checks['preferred_samples_below_cutoff_with_positive_pixel_lower_bound']} have "
        'positive lower pixel bounds. These occur in '
        f"{len(checks['cells_with_positive_samples_below_cutoff'])} cells. The proposed fixed "
        'law predicts exact zero there, irrespective of the fitted scale. These are curve '
        'coordinates, not counts of statistically independent experimental observations.', '',
        'Allowing baseline/gain flexibility substantially improves descriptive fit. The '
        'signed and residual-floor families do not establish a unique superior law across '
        'both heading conditions. Several curves are nonmonotonic; a nonnegative b also '
        'prevents a positive-velocity onset threshold. The candidate set is deliberately limited.', '',
        'No shared baseline/gain is inferred. A fitted zero/floor threshold outside the '
        'observed range is extrapolation, and floor parameters can be unidentifiable when '
        'no floor is sampled. Absence of plotted error bars prevents estimating recording '
        'noise from these curves. A zero-projection bin need not represent stationary behavior.', '',
        'PFNv is not calibrated here. Population-level heading modulation and preservation '
        'of both body-velocity dimensions cannot be established from separate PFNd curve '
        'fits. No successor encoder is approved by this analysis.', '',
        '## Artifacts', '',
        '- [Source and extraction overlay](digitization_overlay.png)',
        '- [Preferred-heading fits](preferred_fits.pdf)',
        '- [Anti-preferred-heading fits](antipreferred_fits.pdf)',
        '- [Digitized coordinates](digitized_points.csv)',
        '- [Per-curve fits](curve_fits.csv)',
        '- [Coordinate sensitivity](extraction_sensitivity.csv)',
        '- [Color-mask sensitivity](mask_sensitivity.csv)',
        '- [Summary](summary.json)',
        '- [Calibration](calibration.json)',
        '- [Exact analysis script](analysis_script.py)', '']
    (output / 'report.md').write_text('\n'.join(lines))
    (output / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>PFNd digitization</title>'
        '<style>body{max-width:1100px;margin:2em auto;font-family:system-ui}img{max-width:100%}</style>'
        '<h1>PFNd digitized response curves</h1><p>Exploratory curve fits; no encoder changes.</p>'
        '<p><a href="report.md">Methods, numerical results and limitations</a> · '
        '<a href="digitized_points.csv">Coordinates</a> · <a href="curve_fits.csv">Fits</a></p>'
        '<h2>Extraction overlay</h2><img src="digitization_overlay.png" alt="Digitized curve coordinates over source image">'
        '<h2>Preferred heading</h2><img src="preferred_fits.png" alt="Preferred-heading curve fits">'
        '<h2>Anti-preferred heading</h2><img src="antipreferred_fits.png" alt="Anti-preferred-heading curve fits">')


def run(source, calibration_path, output):
    source, calibration_path, output = map(Path, (source, calibration_path, output))
    calibration = json.loads(calibration_path.read_text())
    if sha256(source) != calibration['source_sha256']:
        raise ValueError('Source image hash differs from calibrated image.')
    rgb = np.array(Image.open(source).convert('RGB'))
    if list(rgb.shape[1::-1]) != calibration['image_size']:
        raise ValueError('Image dimensions differ from calibrated image.')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output / 'source.webp')
    shutil.copyfile(calibration_path, output / 'calibration.json')
    shutil.copyfile(__file__, output / 'analysis_script.py')
    points = digitize(rgb, calibration)
    fits, predictions, sensitivity = analyze(points)
    write_csv(output / 'digitized_points.csv', points)
    write_csv(output / 'curve_fits.csv', fits)
    write_csv(output / 'fit_predictions.csv', predictions)
    write_csv(output / 'extraction_sensitivity.csv', sensitivity)
    variants = []
    for variant in (-1, 1):
        alternative = digitize(rgb, calibration, variant)
        for central, other in zip(points, alternative):
            variants.append(dict(cell=central['cell'], condition=central['condition'],
                projection_mm_s=central['projection_mm_s'], mask_variant=variant,
                central_status=central['status'], alternative_status=other['status'],
                alternative_rate=other['rate_spikes_s'],
                rate_difference=(other['rate_spikes_s']-central['rate_spikes_s']
                    if central['status'] == other['status'] == 'extracted' else None)))
    write_csv(output / 'mask_sensitivity.csv', variants)
    plot_overlay(rgb, calibration, points, output)
    plot_fits(points, fits, output)
    summary = dict(source_url=calibration['source_url'], equations=EQUATIONS,
        points_extracted=sum(r['status'] == 'extracted' for r in points),
        points_unresolved=sum(r['status'] != 'extracted' for r in points),
        scope='Digitized PFNd ED4a curves only; no PFNv calibration or circuit simulation.',
        limitations=['Pixel envelopes are not physiological confidence intervals.',
            'Zero projected velocity is not a measured stationary baseline.',
            'Fixed half-mm/s sampling can fall on connecting lines; no original bin locations/counts recovered.',
            'Held samples assess curve interpolation, not independent-cell/fly validation.',
            'Each curve fitted separately; no population encoder parameters inferred.'])
    common = [abs(r['rate_difference']) for r in variants if r['rate_difference'] is not None]
    negative = [r for r in points if r['condition'] == 'preferred' and r['status'] == 'extracted'
                and r['projection_mm_s'] < -.2]
    summary['extraction_checks'] = dict(
        mask_common_comparisons=len(common),
        mask_median_absolute_change=float(np.median(common)),
        mask_max_absolute_change=float(max(common)),
        preferred_samples_below_proposed_cutoff=len(negative),
        preferred_samples_below_cutoff_with_positive_pixel_lower_bound=sum(r['rate_low'] > 0 for r in negative),
        cells_with_positive_samples_below_cutoff=sorted({r['cell'] for r in negative if r['rate_low'] > 0}))
    for condition in ('preferred', 'antipreferred'):
        summary[condition] = {}
        for model in MODELS:
            selected = [r for r in fits if r['condition'] == condition and r['model'] == model]
            summary[condition][model] = dict(curves=len(selected),
                median_rmse=float(np.median([r['rmse'] for r in selected])),
                median_held_sample_rmse=float(np.median([r['leave_one_sample_out_rmse'] for r in selected])))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    write_report(summary, output)
    manifest = dict(status='complete', analysis='PFNd physiological plot digitization',
        script_sha256=sha256(__file__), artifacts={p.name: sha256(p) for p in sorted(output.iterdir())})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.image, args.calibration, args.output), indent=2))


if __name__ == '__main__':
    main()
