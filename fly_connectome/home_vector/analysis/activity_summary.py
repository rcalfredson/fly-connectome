"""Array-only accounting of a frozen decoder's observed population activity."""

import numpy as np


def population_readout(rates, decoder, intercept, types, populations):
    """Return [time, population, xy] contributions, with the intercept separate."""
    rates, decoder, intercept = map(np.asarray, (rates, decoder, intercept))
    types = np.asarray(types)
    if (rates.ndim != 2 or decoder.shape != (2, rates.shape[1])
            or intercept.shape != (2,) or types.shape != (rates.shape[1],)
            or len(set(populations)) != len(populations) or set(types) != set(populations)):
        raise ValueError('Inconsistent activity, decoder or population axes.')
    if not all(np.isfinite(a).all() for a in (rates, decoder, intercept)):
        raise ValueError('Activity and decoder must be finite.')
    contributions = np.stack([rates[:, types == p] @ decoder[:, types == p].T
                              for p in populations], axis=1)
    return contributions, contributions.sum(axis=1) + intercept


def neuron_summary(rates, decoder, ids, types):
    """Full-resolution activity and actual readout contributions, one row per unit."""
    rows = []
    for j, (body_id, cell_type) in enumerate(zip(ids, types)):
        rate = rates[:, j]
        row = dict(neuron_index=j, body_id=body_id, cell_type=cell_type,
                   rate_min=float(rate.min()), rate_max=float(rate.max()),
                   rate_mean=float(rate.mean()), rate_initial=float(rate[0]),
                   rate_final=float(rate[-1]))
        for k, axis in enumerate(('x', 'y')):
            contribution = rate * decoder[k, j]
            row.update({f'contribution_{axis}_mean_mm': float(contribution.mean()),
                        f'contribution_{axis}_final_mm': float(contribution[-1]),
                        f'contribution_{axis}_max_abs_mm': float(np.abs(contribution).max())})
        rows.append(row)
    return rows
