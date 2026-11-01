"""Recompute saved metrics, independent geometric targets, and diagnostic plots."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..__main__ import metrics
from ..data import write_json

def independent_target(protocol, n, dt):
    """Integrate each geometric segment directly, independently of the sampled cumulative-sum implementation."""
    times = np.arange(n) * dt
    position = np.zeros((n, 2))
    for row, following in zip(protocol[:-1], protocol[1:]):
        t, heading, vf, vr = row
        theta = np.deg2rad(heading)
        v = np.array([vf*np.cos(theta)-vr*np.sin(theta), vf*np.sin(theta)+vr*np.cos(theta)])
        interval = np.clip(times-t, 0, following[0]-t)
        position += interval[:, None]*v
    return -position

def analyze(audit):
    run, out = audit.run, audit.out
    groups, task = audit.groups, audit.task
    saved = json.loads((run / 'metrics.json').read_text())
    report = dict(splits={}, source_manifest_matches=True, refitting=False)
    arrays = {}
    for split in ['validation', 'test', 'challenge']:
        predictions, targets = audit.predictions(split)
        arrays[split] = (predictions, targets)
        geometry_error = 0.
        for row, p, y in zip(groups[split], predictions, targets):
            expected = independent_target(row['protocol'], len(y), task['dt_s'])
            geometry_error = max(geometry_error, float(np.max(np.abs(y-expected))))
            np.testing.assert_allclose(y, expected, rtol=1e-6, atol=5e-6)
        replay = metrics(predictions, targets)
        for key in ['normalized_rmse', 'cartesian_rmse_mm', 'magnitude_rmse_mm']:
            np.testing.assert_allclose(replay[key], saved[split][key], rtol=1e-6)
        mse = np.mean([np.mean(np.sum((p-y)**2, axis=1)) for p,y in zip(predictions,targets)])
        energy = np.mean([np.mean(np.sum(y**2, axis=1)) for y in targets])
        np.testing.assert_allclose(mse/task['length_scale_mm']**2, saved[split]['equal_trajectory_scaled_mse'], rtol=1e-6)
        summary = dict(recomputed_metrics_match=True, geometry_max_abs_error_mm=geometry_error,
            pooled_metrics=replay, equal_trajectory_rmse_mm=float(np.sqrt(mse)),
            equal_trajectory_normalized_rmse=float(np.sqrt(mse/energy)),
            sqrt_logged_scaled_mse=float(np.sqrt(saved[split]['equal_trajectory_scaled_mse'])),
            groups={})
        for kind in sorted({r['kind'] for r in groups[split]}):
            ix = [i for i,r in enumerate(groups[split]) if r['kind']==kind]
            ps, ys = [predictions[i] for i in ix], [targets[i] for i in ix]
            summary['groups'][kind] = dict(trajectories=len(ix), metrics=metrics(ps,ys),
                equal_trajectory_mse_mm2=float(np.mean([np.mean(np.sum((p-y)**2,axis=1)) for p,y in zip(ps,ys)])),
                terminal_error_median_mm=float(np.median([np.linalg.norm(p[-1]-y[-1]) for p,y in zip(ps,ys)])))
        pairs = {}
        for i,row in enumerate(groups[split]):
            if row['kind']=='matched_pause':
                pairs.setdefault(row['family'], []).append(i)
        pair_errors, pair_true_energy, separations = [], [], []
        for family, ix in pairs.items():
            if len(ix) != 2:
                raise ValueError('Matched history families must contain exactly two trajectories.')
            a,b = ix
            ra,rb = groups[split][a],groups[split][b]
            if ra['suffix_start_s'] != rb['suffix_start_s']:
                raise ValueError('Matched history trajectories require identical suffix timing.')
            # Same suffix timing and controls, independently of the geometric target.
            np.testing.assert_array_equal(np.array(ra['protocol'])[1:], np.array(rb['protocol'])[1:])
            count = round(1/task['dt_s'])+1
            predicted_difference = predictions[a][-count:] - predictions[b][-count:]
            actual_difference = targets[a][-count:] - targets[b][-count:]
            pair_errors.append(float(np.sum((predicted_difference-actual_difference)**2)))
            pair_true_energy.append(float(np.sum(actual_difference**2)))
            separations.append(dict(family=family,
                true_terminal_separation_mm=float(np.linalg.norm(actual_difference[-1])),
                predicted_terminal_separation_mm=float(np.linalg.norm(predicted_difference[-1]))))
        summary['matched_history_difference_nrmse_last_second'] = (
            float(np.sqrt(sum(pair_errors)/sum(pair_true_energy))) if pairs else None)
        summary['matched_pairs'] = separations
        report['splits'][split] = summary

    # Exact protocol copies must not occur between training/validation/test.
    fingerprints = {s: {json.dumps(r['protocol'], sort_keys=True) for r in groups[s]}
                    for s in ['train','validation','test']}
    report['exact_protocol_overlap'] = {f'{a}/{b}': len(fingerprints[a]&fingerprints[b])
        for a,b in [('train','validation'),('train','test'),('validation','test')]}
    if any(report['exact_protocol_overlap'].values()):
        raise ValueError('Exact trajectory protocols overlap between train/validation/test.')
    report['challenges'] = []
    for i,row in enumerate(groups['challenge']):
        p,y = arrays['challenge'][0][i], arrays['challenge'][1][i]
        report['challenges'].append(dict(index=i, family=row['family'], kind=row['kind'],
            duration_s=row['protocol'][-1][0], true_terminal_vector_mm=y[-1].tolist(),
            predicted_terminal_vector_mm=p[-1].tolist(),
            true_terminal_length_mm=float(np.linalg.norm(y[-1])),
            predicted_terminal_length_mm=float(np.linalg.norm(p[-1])),
            terminal_error_mm=float(np.linalg.norm(p[-1]-y[-1])), metrics=metrics([p],[y])))

    # Plot predesignated diagnostic protocols, not post-hoc best examples.
    fig,axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    pred,truth=arrays['challenge']
    for i,row in enumerate(groups['challenge']):
        t=np.arange(len(pred[i]))*task['dt_s']
        if row['family']=='challenge-long-64':
            axes[0,0].plot(t,np.linalg.norm(truth[i],axis=1),'k--',label='True')
            axes[0,0].plot(t,np.linalg.norm(pred[i],axis=1),label='Predicted')
        if row['family']=='challenge-pause-5':
            sign='negative prefix' if row['protocol'][0][2] < 0 else 'positive prefix'
            line=axes[0,1].plot(t,pred[i][:,0],label=f'Predicted, {sign}')[0]
            axes[0,1].plot(t,truth[i][:,0],'--',color=line.get_color())
        if row['family']=='challenge-square':
            axes[1,0].plot(t,truth[i][:,0],'k--',label='True x')
            axes[1,0].plot(t,pred[i][:,0],label='Predicted x')
            axes[1,0].plot(t,truth[i][:,1],color='gray',linestyle='--',label='True y')
            axes[1,0].plot(t,pred[i][:,1],label='Predicted y')
        if row['family']=='challenge-world-equivalent':
            axes[1,1].plot(t,np.linalg.norm(pred[i]-truth[i],axis=1),label=row['kind'])
    for ax,title,ylabel in zip(axes.flat,
        ['Long straight path: magnitude saturates','Opposite histories, identical pause inputs',
         'Square loop: residual at return home','Same world path, different body motions'],
        ['Home-vector magnitude (mm)','Home-vector x component (mm)',
         'Home-vector component (mm)','Vector error (mm)']):
        ax.set(title=title,xlabel='Time (s)',ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    pause = next(r['suffix_start_s'] for r in groups['challenge'] if r['family']=='challenge-pause-5')
    axes[0,1].axvline(pause,color='gray',linewidth=1)
    fig.savefig(out/'diagnostics.png',dpi=180)
    fig.savefig(out/'diagnostics.pdf')
    plt.close(fig)
    write_json(out/'audit.json',report)
