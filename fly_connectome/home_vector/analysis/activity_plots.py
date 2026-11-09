"""Shareable activity atlases and signed population readouts from saved traces."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig, output, stem):
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{stem}.{extension}', dpi=160)
    plt.close(fig)


def decorate(axes, events, end):
    for ax in np.asarray(axes).flat:
        for event in events:
            ax.axvline(event, color='gray', linewidth=.6, alpha=.6)
        ax.set_xlim(0, end)
    np.asarray(axes).flat[-1].set_xlabel('Time (s); vertical lines mark control transitions')


def plot_atlas(arrays, circuit, populations, events, title, vmax, output, stem, detail=False):
    time, rates = arrays['time_s'], arrays['rates']
    controls = arrays['controls']
    fig, axes = plt.subplots(len(populations) + 3, 1, figsize=(12, 3 + 2.2 * len(populations)),
                             sharex=True, layout='constrained',
                             gridspec_kw=dict(height_ratios=[.7, .7] + [2] * len(populations) + [1]))
    scale = 'shared within this trajectory' if detail else 'shared across all trajectories'
    fig.suptitle(title + '\nRecorded activity: model units; neuron order is not anatomical\nColor scale ' + scale)
    axes[0].step(time, controls[:, 0], where='post', color='black')
    axes[0].set_ylabel('Heading (deg)')
    for k, label in ((1, 'Forward'), (2, 'Rightward')):
        axes[1].step(time, controls[:, k], where='post', label=label)
    axes[1].set_ylabel('Velocity (mm/s)')
    axes[1].legend(fontsize=7, loc='upper right')
    dt = time[1] - time[0]
    for ax, population in zip(axes[2:-1], populations):
        indices = sorted([j for j, p in enumerate(circuit.dynamic_types) if p == population],
                         key=lambda j: circuit.dynamic_ids[j])
        im = ax.imshow(rates[:, indices].T, aspect='auto', origin='lower', interpolation='nearest',
                       extent=(-dt / 2, time[-1] + dt / 2, -.5, len(indices) - .5),
                       vmin=0, vmax=max(vmax, 1e-12), cmap='viridis', rasterized=True)
        ax.set_yticks(range(len(indices)), [str(circuit.dynamic_ids[j]) for j in indices], fontsize=5)
        ax.set_ylabel(population + '\nBody ID')
    fig.colorbar(im, ax=list(axes[2:-1]), label='Rate (model units); scale ' + scale, pad=.01)
    for k, axis in enumerate(('x', 'y')):
        axes[-1].plot(time, arrays['prediction'][:, k], color=f'C{k}', label=f'Decoded {axis}')
        axes[-1].plot(time, arrays['target'][:, k], '--', color=f'C{k}', label=f'True {axis}')
    axes[-1].set_ylabel('Home (mm)')
    axes[-1].legend(fontsize=7, ncol=4)
    decorate(axes, events, time[-1])
    save(fig, output, stem + ('_atlas_detail' if detail else '_atlas'))


def plot_readout(arrays, populations, intercept, events, title, output, stem):
    time = arrays['time_s']
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout='constrained')
    fig.suptitle(title + '\nSigned population contributions through the frozen decoder')
    for k, (ax, axis) in enumerate(zip(axes, ('x', 'y'))):
        for p, population in enumerate(populations):
            ax.plot(time, arrays['population_contribution'][:, p, k], label=population, color=f'C{p}')
        ax.axhline(intercept[k], color='gray', linestyle=':', label='Decoder intercept')
        ax.plot(time, arrays['prediction'][:, k], color='black', linewidth=1.8, label='Total decoded')
        ax.plot(time, arrays['target'][:, k], color='black', linestyle='--', linewidth=1.3, label='True')
        ax.set_ylabel(f'Home-vector {axis} (mm)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=4)
    decorate(axes, events, time[-1])
    save(fig, output, stem + '_readout')
