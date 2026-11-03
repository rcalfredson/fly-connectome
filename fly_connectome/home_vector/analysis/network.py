"""Compare trained edge coefficients, effective weights and intrinsic parameters."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..data import write_json
from ..dynamics import initial_coefficients
from ..model import HomeVectorNetwork

def analyze(audit):
    run, out, manifest = audit.run, audit.out, audit.manifest
    audit.record['torch'] = torch.__version__
    if not manifest['settings']['recurrence']:
        raise ValueError('Network-weight comparison requires a recurrent training run.')
    circuit = audit.circuit
    population_types = circuit.population_types
    config=json.loads((run/'training_config.json').read_text())
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    state=checkpoint['model']
    seed=manifest['settings']['seed']
    model=HomeVectorNetwork(circuit,seed=seed)
    expected_initial=initial_coefficients(circuit,seed).astype(np.float32)
    expected_counts=np.asarray([e['contacts']/e['denominator'] for e in circuit.edges],dtype=np.float32)
    np.testing.assert_array_equal(state['a_initial'].numpy(),expected_initial)
    np.testing.assert_array_equal(state['counts'].numpy(),expected_counts)
    np.testing.assert_array_equal(state['pre'].numpy(),[e['pre'] for e in circuit.edges])
    np.testing.assert_array_equal(state['post'].numpy(),[e['post'] for e in circuit.edges])
    np.testing.assert_array_equal(model.bias.detach().numpy(),np.full(len(population_types),config['initial']['bias'],dtype=np.float32))
    np.testing.assert_array_equal(model.tau.detach().numpy(),np.full(len(population_types),config['initial']['tau_s'],dtype=np.float32))

    a=state['a'].numpy().astype(float)
    a0=state['a_initial'].numpy().astype(float)
    scale=state['counts'].numpy().astype(float)
    # Reproduce actual float32 coefficient multiplication before summarizing in float64.
    w=(state['a']*state['counts']).numpy().astype(float)
    w0=(state['a_initial']*state['counts']).numpy().astype(float)
    ids=circuit.input_ids+circuit.dynamic_ids
    all_types=circuit.input_types+circuit.dynamic_types
    table=pd.DataFrame([dict(source_id=ids[e['pre']],target_id=circuit.dynamic_ids[e['post']],
        source_type=e['source_type'],target_type=e['target_type'],
        group='PFN input' if e['pre']<len(circuit.input_ids) else 'internal',
        contacts=e['contacts'],normalization_denominator=e['denominator'],
        normalized_contacts=scale[i],initial_coefficient=a0[i],trained_coefficient=a[i],
        coefficient_change=a[i]-a0[i],initial_effective_weight=w0[i],trained_effective_weight=w[i],
        effective_weight_change=w[i]-w0[i],sign_changed=bool(a[i]*a0[i]<0))
        for i,e in enumerate(circuit.edges)])
    assert not table.duplicated(['source_id','target_id']).any()

    def norm(x):
        return float(np.linalg.norm(x))

    def similarity(x,y):
        return float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))) if norm(x)*norm(y)>0 else None

    def summary(mask):
        u,v=w0[mask],w[mask]
        delta=v-u
        return dict(edges=int(mask.sum()),raw_contacts=int(table.loc[mask,'contacts'].sum()),
            initial_l2=norm(u),trained_l2=norm(v),change_l2=norm(delta),
            relative_change_l2=norm(delta)/norm(u),trained_to_initial_l2=norm(v)/norm(u),
            signed_cosine=similarity(u,v),magnitude_cosine=similarity(np.abs(u),np.abs(v)),
            coefficient_change_rms=float(np.sqrt(np.mean((a[mask]-a0[mask])**2))),
            median_absolute_coefficient_change=float(np.median(np.abs(a[mask]-a0[mask]))),
            changed_coefficients_above_1e_minus6=int((np.abs(a[mask]-a0[mask])>1e-6).sum()),
            sign_changes=int((a[mask]*a0[mask]<0).sum()),
            positive_trained=int((a[mask]>0).sum()),negative_trained=int((a[mask]<0).sum()),
            zero_trained=int((a[mask]==0).sum()),
            coefficient_bound_hits=int((np.abs(a[mask])>=5).sum()))

    groups={name:summary(mask) for name,mask in [
        ('all',np.ones(len(a),dtype=bool)),('PFN input',table.group.eq('PFN input').to_numpy()),
        ('internal',table.group.eq('internal').to_numpy())]}
    blocks=[]
    for pair in dict.fromkeys((e['source_type'],e['target_type']) for e in circuit.edges):
        mask=(table.source_type.eq(pair[0])&table.target_type.eq(pair[1])).to_numpy()
        b,v=scale[mask],w[mask]
        # Descriptive least-squares projection onto the count-proportional pattern, not task training.
        gain=float(np.dot(b,v)/np.dot(b,b))
        residual=norm(v-gain*b)/norm(v)
        blocks.append(dict(source_type=pair[0],target_type=pair[1],initial_coefficient=float(a0[mask][0]),
            trained_coefficient_median=float(np.median(a[mask])),
            trained_coefficient_min=float(a[mask].min()),trained_coefficient_max=float(a[mask].max()),
            best_count_proportional_gain=gain,proportionality_relative_residual=residual,
            count_proportional_fraction_weight_energy=float(1-residual**2),
            proportionality_interpretation='not informative for a singleton' if mask.sum()==1 else 'multi-edge projection',
            **summary(mask)))

    intrinsics=pd.DataFrame(dict(cell_type=list(population_types),
        initial_tau_s=config['initial']['tau_s'],trained_tau_s=state['tau'].numpy(),
        initial_bias=config['initial']['bias'],trained_bias=state['bias'].numpy()))
    decoder=pd.DataFrame(dict(body_id=circuit.dynamic_ids,cell_type=circuit.dynamic_types,
        decoder_x=state['D'].numpy()[0],decoder_y=state['D'].numpy()[1]))

    initial_matrix=np.zeros((len(ids),len(circuit.dynamic_ids)),dtype=np.float32)
    trained_matrix=initial_matrix.copy()
    count_matrix=np.zeros(initial_matrix.shape,dtype=int)
    for i,e in enumerate(circuit.edges):
        initial_matrix[e['pre'],e['post']]=w0[i]
        trained_matrix[e['pre'],e['post']]=w[i]
        count_matrix[e['pre'],e['post']]=e['contacts']
    model.load_state_dict(state)
    np.testing.assert_array_equal(model.weights().detach().numpy(),trained_matrix)
    initial_model=HomeVectorNetwork(circuit,seed=seed)
    np.testing.assert_array_equal(initial_model.weights().detach().numpy(),initial_matrix)
    assert np.array_equal(count_matrix>0,trained_matrix!=0)

    dominant=int(np.argmax(w0**2))
    report=dict(run=str(run),selected_update=checkpoint['update'],seed=seed,groups=groups,projections=blocks,
        structural_checks=dict(input_bodies=len(circuit.input_ids),dynamic_bodies=len(circuit.dynamic_ids),
            positive_anatomical_pairs=len(a),edge_mask_unchanged=True,counts_buffer_unchanged=True,
            initial_coefficients_reconstructed_exactly=True,model_matrix_reproduced_exactly=True,
            no_reported_vDeltaE_to_hDeltaB_edges=not any(e['source_type']=='vDeltaE' and e['target_type']=='hDeltaB' for e in circuit.edges)),
        strongest_initial_edge=dict(**table.iloc[dominant].to_dict(),
            initial_fraction_global_squared_weight_norm=float(w0[dominant]**2/np.sum(w0**2)),
            trained_fraction_global_squared_weight_norm=float(w[dominant]**2/np.sum(w**2))),
        intrinsic_parameters=intrinsics.to_dict(orient='records'),
        decoder=dict(initial='zero weights and intercepts',trained_weight_l2=norm(state['D'].numpy()),
            trained_intercepts_mm=state['d'].numpy().tolist()),
        definitions=dict(relative_change_l2='||W_trained-W_initial||_F / ||W_initial||_F on the same edges',
            trained_to_initial_l2='||W_trained||_F / ||W_initial||_F; not a ratio for each individual edge',
            proportionality_relative_residual='||W_trained-g_best*(C/S)||_2 / ||W_trained||_2 within one type-pair block',
            gain='g_best minimizes the descriptive squared residual for this saved block; no model parameter was replaced',
            weights='Float32 effective weights from checkpoint: W_ij=(C_ij/S_typepair)*a_ij',
            signs='Relative to assumed seeded initialization, not measured biological signs',
            rank_order='body IDs within cell type; no shared column/angle assignment'))

    table.to_csv(out/'edge_changes.csv',index=False)
    pd.DataFrame(blocks).to_csv(out/'projection_summary.csv',index=False)
    intrinsics.to_csv(out/'intrinsic_changes.csv',index=False)
    decoder.to_csv(out/'decoder_weights.csv',index=False)
    for name,matrix in [('raw_contacts',count_matrix),('initial_effective_weights',initial_matrix),
                        ('trained_effective_weights',trained_matrix),('effective_weight_changes',trained_matrix-initial_matrix)]:
        pd.DataFrame(matrix,index=pd.Index(ids,name='presynaptic_body_id'),
                     columns=pd.Index(circuit.dynamic_ids,name='postsynaptic_body_id')).to_csv(out/f'{name}.csv')
    write_json(out/'summary.json',report)

    if circuit.dynamic_types != [t for t in population_types for _ in range(circuit.dynamic_types.count(t))]:
        raise ValueError('Network plots require neurons grouped in declared population order.')
    sizes=[circuit.dynamic_types.count(t) for t in population_types]
    boundaries=np.cumsum(sizes)
    centers=boundaries-(np.asarray(sizes)+1)/2
    fig,axes=plt.subplots(1,3,figsize=(11,4.1),layout='constrained')
    recurrent=[initial_matrix[len(circuit.input_ids):],trained_matrix[len(circuit.input_ids):],trained_matrix[len(circuit.input_ids):]-initial_matrix[len(circuit.input_ids):]]
    limit=max(float(np.max(np.abs(x))) for x in recurrent)
    for ax,matrix,title in zip(axes,recurrent,['Initial internal coupling','Trained internal coupling','Trained minus initial']):
        im=ax.imshow(matrix,cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
        for boundary in boundaries[:-1]:
            ax.axhline(boundary-.5,color='black',linewidth=.6)
            ax.axvline(boundary-.5,color='black',linewidth=.6)
        ax.set(xticks=centers,yticks=centers,xticklabels=population_types,
               yticklabels=population_types,title=title,xlabel='Postsynaptic population')
    axes[0].set_ylabel('Presynaptic population')
    fig.colorbar(im,ax=axes,label='Effective coupling (shared scale; arbitrary units)',shrink=.75)
    fig.savefig(out/'internal_weight_changes.png',dpi=180)
    fig.savefig(out/'internal_weight_changes.pdf')
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,6),layout='constrained',sharey=True)
    labels=[r['source_type']+' → '+r['target_type'] for r in blocks]
    y=np.arange(len(blocks))
    colors=['tab:blue' if r['source_type'].startswith('PFN') else 'tab:orange' for r in blocks]
    axes[0].barh(y,[r['trained_to_initial_l2'] for r in blocks],color=colors)
    axes[0].set_xscale('log')
    axes[0].axvline(1,color='black',linestyle='--',linewidth=1)
    axes[0].set(yticks=y,yticklabels=labels,xlabel='Trained / initial weight norm (log scale)',
                title='Projection strength changes')
    axes[1].barh(y,[100*r['proportionality_relative_residual'] for r in blocks],color=colors)
    axes[1].set(xlim=(0,105),xlabel='Relative residual after best count scaling (%)',
                title='Departure from within-projection count ratios')
    axes[0].invert_yaxis()
    for ax in axes: ax.grid(axis='x',alpha=.2)
    fig.savefig(out/'projection_changes.png',dpi=180)
    fig.savefig(out/'projection_changes.pdf')
    plt.close(fig)
