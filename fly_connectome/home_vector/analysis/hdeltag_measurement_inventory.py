"""Extract coarse displayed calcium means for physiology measurement discovery.

No raw flies/events are reconstructed and no model is fitted or simulated.
"""

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
from ..data import digest, write_json


def extract_dot(rgb, patch, color, pixels, values, padding):
    """Estimate a manually isolated bold mean-dot center, never identify a fly."""
    x0, y0, x1, y1 = patch
    tile = np.asarray(rgb[y0:y1, x0:x1], dtype=float)
    r, g, b = np.moveaxis(tile, -1, 0)
    if color == 'green':
        mask = (g > r+20) & (g > b+15) & (g < 150) & (r < 120)
    elif color == 'purple':
        mask = (r > g+15) & (b > g+15) & (g < 130)
    else:
        raise ValueError('Unknown dot color.')
    yy, xx = np.nonzero(mask)
    if not len(yy):
        raise ValueError('No dark mean-dot pixels in calibrated patch.')
    y, x = float(np.median(yy)+y0), float(np.median(xx)+x0)
    def coordinate(py):
        return float(values[0] + (py-pixels[0]) * (values[1]-values[0]) / (pixels[1]-pixels[0]))
    envelope = [coordinate(yy.min()+y0-padding), coordinate(yy.max()+y0+padding)]
    return dict(x_pixel=x, y_pixel=y, value=coordinate(y),
                pixel_envelope_low=min(envelope), pixel_envelope_high=max(envelope))


def assess(args):
    pdf, calibration, output = [Path(getattr(args, k)).resolve() for k in ('pdf', 'calibration', 'output')]
    if output.exists():
        raise FileExistsError(output)
    c = json.loads(calibration.read_text())
    if digest(pdf) != c['source_pdf_sha256']:
        raise ValueError('PDF differs from the calibrated thesis.')
    inputs = {str(p): digest(p) for p in (pdf, calibration)}
    output, manifest = begin(output, 'hdeltag_measurement_inventory_v1', pdf=str(pdf), calibration=str(calibration))
    manifest.update(immutable_input_sha256=inputs, simulation=False, fitting=False,
                    analysis_script_sha256=digest(__file__))
    write_json(output / 'manifest.json', manifest)
    try:
        images = {}
        for page in c['review_pages']:
            stem = output / f'pdf_page_{page:03d}'
            subprocess.run(['pdftoppm', '-f', str(page), '-l', str(page), '-scale-to',
                            str(c['render_scale_to']), '-png', '-singlefile', str(pdf), str(stem)],
                           check=True, capture_output=True)
            rgb = np.asarray(Image.open(stem.with_suffix('.png')).convert('RGB'))
            if list(rgb.shape[1::-1]) != c['image_size']:
                raise ValueError('Unexpected rendered page dimensions.')
            images[page] = rgb
        points, summaries = [], []
        fig, axes = plt.subplots(2, 3, figsize=(12, 9), layout='constrained')
        for ax, panel in zip(axes.flat, c['panels']):
            found = {}
            rgb = images[panel['page']]
            for condition in ('before', 'after'):
                dot = extract_dot(rgb, panel[condition+'_patch'], panel['color'],
                                  panel['y_pixels'], panel['y_values'], c['padding_pixels'])
                found[condition] = dot
                points.append(dict(panel=panel['name'], figure=panel['figure'], pdf_page=panel['page'],
                                   condition=condition, **dot))
            before, after = found['before']['value'], found['after']['value']
            summaries.append(dict(panel=panel['name'], figure=panel['figure'], before=before,
                after=after, change=after-before, plotted_after_before_ratio=after/before if before != 0 else None))
            ax.imshow(rgb)
            xlimits, ylimits = [], []
            for condition in ('before', 'after'):
                x0,y0,x1,y1 = panel[condition+'_patch'];xlimits += [x0,x1];ylimits += [y0,y1]
                ax.add_patch(plt.Rectangle((x0,y0), x1-x0, y1-y0, fill=False, edgecolor='red', lw=.8))
                ax.plot(found[condition]['x_pixel'], found[condition]['y_pixel'], '+', color='blue', ms=8)
            ax.set(xlim=(min(xlimits)-10,max(xlimits)+10), ylim=(max(ylimits)+15,min(ylimits)-15),
                   title=panel['name'].replace('_', '\n'))
        fig.savefig(output / 'mean_dot_overlay.png', dpi=160)
        plt.close(fig)
        for name, rows in [('mean_dot_coordinates.csv',points), ('mean_dot_summary.csv',summaries)]:
            with (output/name).open('w',newline='') as stream:
                writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write_json(output/'measurement_scope.json',dict(source_pdf_sha256=digest(pdf),units=c['units'],
            note=c['note'], calibrated_gate_parameters=False, recovered_individual_flies=False,
            warning='Ratios describe normalized plotted ordinates, not fractions of remaining spike rate.'))
        shutil.copyfile(calibration,output/'calibration.json')
        shutil.copyfile(__file__,output/'analysis_script.py')
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8">'
            '<title>hDeltaG measurement inventory</title><style>body{font:16px system-ui;max-width:1200px;'
            'margin:2em auto}img{max-width:100%}</style><h1>Displayed physiology measurements</h1>'
            '<p>Coarse population-mean dot estimates only. Normalized calcium units; '
            'no raw events, individual-fly reconstruction, parameter fit or simulation.</p>'
            '<p><a href="mean_dot_summary.csv">Mean summary</a> · '
            '<a href="mean_dot_coordinates.csv">Coordinates and pixel envelopes</a> · '
            '<a href="measurement_scope.json">Limitations</a></p>'
            '<img src="mean_dot_overlay.png" alt="Mean-dot extraction review">'
            + ''.join(f'<h2>PDF page {page}</h2><img src="pdf_page_{page:03d}.png" alt="Source page {page}">'
                      for page in c['review_pages']))
        if any(digest(p)!=h for p,h in inputs.items()):
            raise ValueError('An immutable input changed.')
        finish(output,manifest)
    except BaseException as error:
        fail(output,manifest,error)
        raise
    return summaries


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pdf','calibration','output'):
        parser.add_argument(f'--{name}',required=True)
    return assess(parser.parse_args(argv))


if __name__=='__main__':
    main()
