"""Opt-in paired training from shared initialization with a fixed sugar rule.

The original production network/trainer and archived fitted checkpoint are untouched.
"""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .__main__ import batch, begin, fail, finish
from .data import Circuit, digest, verify_bundle, write_json
from .model import HomeVectorNetwork, objective
from .training_batches import length_groups
from .analysis.sugar_decoder_data import validate_splits
from .analysis.sugar_decoder_fit import family_weights
from .analysis.sugar_response import SugarResponse, event_gate


def sugar_block(state, drives, gates, recurrent, bias, tau, mask, strength, reference, dt):
    """Full differentiable recurrence; zero gate selects native arithmetic exactly."""
    alpha=-torch.expm1(-dt/tau)
    states=[]
    for drive,gate in zip(drives,gates):
        target=torch.relu(bias+drive+state@recurrent)
        native=state+alpha*(target-state)
        extra=gate[:,None]*strength
        decay=1./tau+extra
        equilibrium=(target/tau+extra*reference)/decay
        modified=state+(-torch.expm1(-dt*decay))*(equilibrium-state)
        state=torch.where(mask & (extra>0),modified,native)
        states.append(state)
    return state,torch.stack(states,dim=1)


class SugarTrainingNetwork(HomeVectorNetwork):
    """Same parameters/anatomy as the production network; explicit sensory gate."""

    def __init__(self,circuit,response,seed=0):
        super().__init__(circuit,seed)
        self.response=response
        self._sugar_mask=tuple(t=='hDeltaG' for t in circuit.dynamic_types)
        self._sugar_compiled=None

    def compile_sugar(self,steps=32,backend='inductor'):
        if type(steps) is not int or steps<1:raise ValueError('Require positive block steps.')
        self._sugar_steps=steps
        # Length grouping changes the batch dimension. Mark that dimension
        # explicitly at calls below; keep neuron dimensions/block length fixed.
        self._sugar_compiled=torch.compile(sugar_block,backend=backend,fullgraph=True,dynamic=None)
        return self

    def forward(self,pfn,dt,gates):
        if (pfn.ndim!=3 or pfn.shape[2]!=self.input_size or pfn.shape[1]<2
                or gates.shape!=pfn.shape[:2] or not np.isfinite(dt) or not 0<dt<=.0025
                or gates.device!=pfn.device or not torch.isfinite(gates).all()
                or (gates<0).any() or (gates>1).any()
                or not torch.isfinite(pfn).all() or (pfn<0).any()):
            raise ValueError('Require aligned nonnegative PFNs, gates in [0,1], dt <= .0025.')
        w=self.weights();state=pfn.new_zeros(pfn.shape[0],self.size)
        drives=(pfn@w[:self.input_size]).transpose(0,1).contiguous().unbind(0)
        gate_steps=gates.transpose(0,1).contiguous().unbind(0)
        mask=torch.tensor(self._sugar_mask,device=pfn.device)
        tau=self.tau[self.type_index];bias=self.bias[self.type_index]
        chunks=[state[:,None]]
        steps=self._sugar_steps if self._sugar_compiled is not None else 32
        for start in range(0,len(drives)-1,steps):
            end=min(start+steps,len(drives)-1)
            # Initial state has requires_grad=False, continuation has True.
            # Execute the first block eagerly so compiled calls only see the
            # differentiable continuation. No state/history is detached.
            compiled=(start>0 and torch.is_grad_enabled() and self._sugar_compiled is not None and end-start==steps)
            if compiled:
                for tensor in (state,*drives[start:end],*gate_steps[start:end]):
                    # Size-one dimensions specialize separately in Dynamo.
                    if tensor.shape[0]>1:torch._dynamo.mark_dynamic(tensor,0)
            block=self._sugar_compiled if compiled else sugar_block
            state,states=block(state,drives[start:end],gate_steps[start:end],w[self.input_size:],
                bias,tau,mask,self.response.relaxation_rate_per_s,self.response.reference_activity,dt)
            chunks.append(states)
        rates=torch.cat(chunks,dim=1)
        return rates@self.D.T+self.d,rates


def prepared_batch(circuit,rows,dt,device,response):
    x,y,mask=batch(circuit,rows,dt,device,torch.float32)
    gates=np.zeros(x.shape[:2])
    for j,r in enumerate(rows):
        n=round(r['protocol'][-1][0]/dt)+1
        gates[j,:n]=event_gate(np.arange(n)*dt,r['sugar_events_s'],response)
        mask[j,n-1]=0  # Final state is evaluated separately, not time-averaged.
    return x,y,mask,torch.as_tensor(gates,device=device,dtype=x.dtype)


def evaluate(network,circuit,rows,dt,device,response,enabled,batch_size=16):
    weights=family_weights(rows);total=terminal=0.
    order=sorted(range(len(rows)),key=lambda i:rows[i]['protocol'][-1][0])
    with torch.no_grad():
        for start in range(0,len(order),batch_size):
            indices=order[start:start+batch_size];selected=[rows[i] for i in indices]
            x,y,mask,gates=prepared_batch(circuit,selected,dt,device,response)
            prediction,rates=network(x,dt,gates if enabled else torch.zeros_like(gates))
            if not torch.isfinite(rates).all() or rates.max()>1e6 or not torch.isfinite(prediction).all():
                raise ValueError('Evaluation exceeded numerical guard.')
            error=(prediction-y).square().sum(-1)
            per=((error*mask).sum(1)/mask.sum(1)).cpu().numpy()
            for j,i in enumerate(indices):
                total+=weights[i]*float(per[j])
                n=round(rows[i]['protocol'][-1][0]/dt)
                terminal+=weights[i]*float(error[j,n])
    return dict(vector_rmse_mm=float(np.sqrt(total)),endpoint_rms_error_mm=float(np.sqrt(terminal)))


def train(args):
    if args.updates<1 or args.batch_size<1 or args.validate_every<1 or args.compile_block<0:
        raise ValueError('Require positive update/batch/validation counts and nonnegative compile size.')
    if args.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable.')
    if args.smoke and args.benchmark:raise ValueError('Choose smoke or full-workload benchmark, not both.')
    data,output=Path(args.data).resolve(),Path(args.output).resolve()
    manifest=verify_bundle(data)
    if manifest['kind']!='home_vector_sugar_decoder_data_v1':raise ValueError('Require derived sugar task.')
    task=json.loads((data/'task.json').read_text());source=Path(task['source'])
    for folder in (data,source,Path(task['activity']),Path(task['diagnostics'])):
        if output==folder or output.is_relative_to(folder) or folder.is_relative_to(output):
            raise ValueError('Output must be separate from every original archive.')
    if output.exists():raise FileExistsError(output)
    circuit=Circuit(**json.loads((source/'circuit.json').read_text()))
    scale=json.loads((source/'task.json').read_text())['length_scale_mm'];dt=task['dt_s']
    groups=json.loads((data/'trajectories.json').read_text());validate_splits(groups)
    response=SugarResponse(**json.loads((data/'response_config.json').read_text())['response'])
    if args.smoke:
        # Exercise actual post-event dynamics, not a truncated pre-sugar prefix.
        for split in ('train','validation'):
            rows=groups[split]
            groups[split]=[min((r for r in rows if r['group']=='existing'),key=lambda r:r['protocol'][-1][0]),
                min((r for r in rows if r['arm']=='continuous_sugar'),key=lambda r:r['protocol'][-1][0])]
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    networks={k:SugarTrainingNetwork(circuit,response,args.seed).to(args.device) for k in ('original','sugar')}
    for key,value in networks['original'].state_dict().items():
        torch.testing.assert_close(value,networks['sugar'].state_dict()[key],atol=0,rtol=0)
    if args.compile_block:
        for net in networks.values():net.compile_sugar(args.compile_block,args.backend)
    optimizers={k:torch.optim.Adam(net.parameters(),lr=.001) for k,net in networks.items()}
    frozen=dict(manifest['immutable_input_sha256'])
    frozen.update({str(p):digest(p) for p in data.iterdir() if p.is_file()})
    if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('An archived input changed.')
    output,record=begin(output,'home_vector_sugar_paired_training_v1',data=str(data),seed=args.seed,
        updates=args.updates,batch_size=args.batch_size,device=args.device,smoke=args.smoke,
        benchmark=args.benchmark,
        validate_every=args.validate_every,compile_block=args.compile_block,backend=args.backend)
    runtime=[Path(__file__),Path(__file__).with_name('model.py'),
             Path(__file__).with_name('training_batches.py'),Path(__file__).with_name('data.py'),
             Path(__file__).with_name('__main__.py'),
             Path(__file__).parent/'analysis'/'sugar_response.py',
             Path(__file__).parent/'analysis'/'sugar_decoder_data.py',
             Path(__file__).parent/'analysis'/'sugar_decoder_fit.py']
    code_hashes={str(p):digest(p) for p in runtime}
    record.update(immutable_input_sha256=frozen,implementation_sha256=code_hashes,torch=torch.__version__)
    write_json(output/'manifest.json',record)
    try:
        for path in runtime:shutil.copyfile(path,output/path.name)
        write_json(output/'circuit.json',circuit.as_dict())
        write_json(output/'training_config.json',dict(response=response.__dict__,dt_s=dt,length_scale_mm=scale,
            initialization='Matched from scratch, seed-shared; not the previously optimized checkpoint.',
            sampling='50/50 existing/return domain, equal family then variant; same draws in both arms.',
            trainable=['all anatomical edge coefficients','population bias','population tau','decoder','intercept'],
            fixed=['PFN encoder','anatomical edge support and counts','sugar response parameters'],
            optimizer='Adam',learning_rate=.001,clip_norm=1.,full_trajectory_bptt=True,
            penalties=dict(activity=1e-5,edge_prior=1e-4,decoder=1e-5),
            selection='Independent validation-minimum balanced MSE at identical update checkpoints.',
            test_policy='Test evaluated once after selection; diagnostic/challenge never select.',
            smoke=args.smoke,benchmark=args.benchmark))
        for name,net in networks.items():torch.save(net.state_dict(),output/(name+'_initial.pt'))
        best={k:float('inf') for k in networks};history=[];started=time.monotonic()
        probability=family_weights(groups['train'])
        for update in range(1,args.updates+1):
            # For smoke ensure both a treated and an untreated example are exercised.
            indices=np.arange(2) if args.smoke else rng.choice(len(groups['train']),args.batch_size,p=probability)
            rows=[groups['train'][i] for i in indices]
            partitions,_=length_groups(rows,dt)
            for opt in optimizers.values():opt.zero_grad(set_to_none=True)
            losses={k:0. for k in networks}
            for positions in partitions:
                selected=[rows[i] for i in positions]
                x,y,mask,gates=prepared_batch(circuit,selected,dt,args.device,response)
                for name,net in networks.items():
                    prediction,rates=net(x,dt,gates if name=='sugar' else torch.zeros_like(gates))
                    loss,_=objective(prediction,y,rates,mask,scale,net)
                    if not torch.isfinite(loss) or not torch.isfinite(rates).all() or rates.max()>1e6:
                        raise ValueError('Training exceeded numerical guard.')
                    weighted=loss*len(positions)/len(rows)
                    weighted.backward();losses[name]+=float(weighted.detach())
                    del prediction,rates,loss,weighted
            row=dict(update=update,indices=indices.tolist(),loss=losses)
            for name,net in networks.items():
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                optimizers[name].step();net.project_parameters()
                row[name+'_gradient_norm']=float(norm)
            if not args.benchmark and (update%args.validate_every==0 or update==args.updates):
                for name,net in networks.items():
                    score=evaluate(net,circuit,groups['validation'],dt,args.device,response,name=='sugar')
                    row[name+'_validation']=score
                    if score['vector_rmse_mm']<best[name]:
                        best[name]=score['vector_rmse_mm']
                        torch.save(dict(model=net.state_dict(),update=update,validation=score),output/(name+'_best.pt'))
            row['elapsed_s']=time.monotonic()-started;history.append(row)
            write_json(output/'history.json',history)
            # Save only after both arms complete their optimizer update. An
            # atomic paired snapshot preserves progress even if a later batch
            # fails before the first validation/selected checkpoint.
            pending=output/'paired_last.pt.tmp'
            torch.save(dict(update=update,
                arms={name:dict(model=net.state_dict(),optimizer=optimizers[name].state_dict())
                      for name,net in networks.items()},
                numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                **({'torch_cuda_rng':torch.cuda.get_rng_state_all()} if args.device=='cuda' else {})),pending)
            pending.replace(output/'paired_last.pt')
            print(f'Paired update {update}/{args.updates}: {losses}',flush=True)
        result=dict(smoke=args.smoke,benchmark=args.benchmark,completed_updates=args.updates,elapsed_s=time.monotonic()-started)
        for name,net in networks.items():
            torch.save(dict(model=net.state_dict(),optimizer=optimizers[name].state_dict(),update=args.updates,
                numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                **({'torch_cuda_rng':torch.cuda.get_rng_state_all()} if args.device=='cuda' else {})),output/(name+'_last.pt'))
            if args.benchmark:
                result[name]=dict(selected_update=None,selection='Unselected engineering benchmark; no validation or test evaluation.')
                continue
            chosen=torch.load(output/(name+'_best.pt'),weights_only=False,map_location=args.device)
            result[name]=dict(selected_update=chosen['update'],validation=chosen['validation'])
            if not args.smoke:
                net.load_state_dict(chosen['model'])
                result[name]['test']=evaluate(net,circuit,groups['test'],dt,args.device,response,name=='sugar')
        write_json(output/'comparison.json',result)
        if any(digest(p)!=h for p,h in frozen.items()):raise ValueError('An archived input changed.')
        if any(digest(p)!=h for p,h in code_hashes.items()):raise ValueError('Training implementation changed during execution.')
        finish(output,record)
    except BaseException as error:fail(output,record,error);raise
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--updates',type=int,default=2000);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--validate-every',type=int,default=50);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--compile-block',type=int,default=0)
    p.add_argument('--backend',choices=('inductor','aot_eager'),default='inductor')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--benchmark',action='store_true',help='Real training batches; no validation, test evaluation or selected checkpoint.')
    return train(p.parse_args(argv))


if __name__=='__main__':main()
