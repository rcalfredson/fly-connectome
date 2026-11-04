"""Prepare offline data, fit baselines, benchmark and train isolated capacity models."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np

from .data import (Circuit, assemble, datasets, digest, encode, rewire, sample_protocol,
                   target_scale, verify_bundle, write_json)


def code_hashes():
    return {str(p.relative_to(Path(__file__).parents[2])): digest(p)
            for p in Path(__file__).parent.glob("*.py")}


def begin(output, kind, **settings):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status="running", kind=kind, started_at_utc=datetime.now(timezone.utc).isoformat(),
                  settings=settings, code_sha256=code_hashes(), python=platform.python_version(),
                  numpy=np.__version__)
    write_json(output / "manifest.json", record)
    return output, record


def finish(output, record):
    settings = record.get("settings", {})
    if "source_manifest_sha256" in settings:
        source = Path(settings["source"])
        verify_bundle(source)
        if digest(source / "manifest.json") != settings["source_manifest_sha256"]:
            raise ValueError("Prepared source changed during experiment.")
    record.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  artifacts={p.name: digest(p) for p in output.iterdir()
                             if p.is_file() and p.name != "manifest.json"})
    write_json(output / "manifest.json", record)


def fail(output, record, error):
    record.update(status="failed", error=f"{type(error).__name__}: {error}")
    write_json(output / "manifest.json", record)


def prepare(args):
    if not np.isfinite(args.dt) or not 0 < args.dt <= .0025:
        raise ValueError("Use a preparation timestep in (0, 0.0025] seconds.")
    circuit, provenance = assemble(args.movement, args.census, args.inputs,
                                   hdeltac=getattr(args, "hdeltac", None))
    groups = datasets(args.seed, dict(train=args.train, validation=args.validation, test=args.test))
    output, record = begin(args.output, "home_vector_prepared_v1", seed=args.seed, dt_s=args.dt,
                           circuit_variant="hdeltac" if getattr(args, "hdeltac", None) else "core")
    try:
        write_json(output / "circuit.json", circuit.as_dict())
        write_json(output / "trajectories.json", groups)
        write_json(output / "sources.json", provenance)
        write_json(output / "task.json", dict(dt_s=args.dt, length_scale_mm=target_scale(groups["train"], args.dt),
                   coordinates="clockwise +theta; +x forward at theta=0; +y right/down",
                   target="home - position; external held-velocity integral; initial target zero",
                   family_counts={s: len({r['family'] for r in rows}) for s, rows in groups.items()},
                   trajectory_counts={s: len(rows) for s, rows in groups.items()}))
        # Preserve the exact original feedforward transform as a frozen baseline.
        import pandas as pd
        legacy = pd.read_csv(Path(args.movement) / "effective_weight_matrix.csv", index_col=0)
        legacy.index, legacy.columns = legacy.index.astype(int), legacy.columns.astype(int)
        legacy = legacy.loc[circuit.input_ids, circuit.dynamic_ids[:19]].to_numpy()
        np.save(output / "legacy_weights.npy", legacy)
        for path, expected in provenance["source_sha256"].items():
            if digest(path) != expected:
                raise ValueError(f"Anatomical source changed during preparation: {path}")
        finish(output, record)
    except Exception as error:
        fail(output, record, error)
        raise
    return dict(output=str(output), inputs=len(circuit.input_ids), dynamic=len(circuit.dynamic_ids),
                edges=len(circuit.edges), trajectories={s: len(rows) for s, rows in groups.items()})


def load(source):
    verify_bundle(source)
    source = Path(source)
    return (Circuit(**json.loads((source / "circuit.json").read_text())),
            json.loads((source / "trajectories.json").read_text()),
            json.loads((source / "task.json").read_text()))


def metrics(predictions, targets):
    errors = np.concatenate([p - y for p, y in zip(predictions, targets)])
    truth = np.concatenate(targets)
    error_energy, target_energy = float(np.sum(errors ** 2)), float(np.sum(truth ** 2))
    valid = np.linalg.norm(truth, axis=1) >= 1.
    prediction = np.concatenate(predictions)
    nonzero = np.linalg.norm(prediction, axis=1) >= 1e-6
    angles = np.abs(np.angle(np.exp(1j * (np.arctan2(prediction[:, 1], prediction[:, 0])
                                          - np.arctan2(truth[:, 1], truth[:, 0])))))
    # Undefined predicted direction outside the near-home region is counted as 180 degrees.
    angles[~nonzero] = np.pi
    denom = np.sum((truth - truth.mean(0)) ** 2, axis=0)
    r2 = np.divide(np.sum(errors ** 2, axis=0), denom, out=np.full(2, np.nan), where=denom > 0)
    return dict(cartesian_rmse_mm=float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))),
                normalized_rmse=float(np.sqrt(error_energy / target_energy)) if target_energy else None,
                component_r2=[float(1 - v) if np.isfinite(v) else None for v in r2],
                magnitude_rmse_mm=float(np.sqrt(np.mean((np.linalg.norm(prediction, axis=1)
                                                        - np.linalg.norm(truth, axis=1)) ** 2))),
                median_angle_error_deg=float(np.rad2deg(np.median(angles[valid]))) if valid.any() else None,
                near_home_samples=int((~valid).sum()), direction_failures=int((valid & ~nonzero).sum()))


def baseline(args):
    from .dynamics import initial_coefficients, numpy_rollout
    circuit, groups, task = load(args.source)
    dt = task["dt_s"]
    legacy = np.load(Path(args.source) / "legacy_weights.npy", allow_pickle=False)

    def features(row, mode):
        controls, target = sample_protocol(row["protocol"], dt)
        pfn = encode(circuit, controls)
        if mode == "current_inputs":
            return pfn, target
        if mode == "initialized_frozen":
            rates = numpy_rollout(circuit, pfn, dt, initial_coefficients(circuit, 0),
                                  [.1] * len(circuit.population_types), [.1] * len(circuit.population_types))
            return rates, target
        drive = pfn @ legacy
        rates = np.zeros_like(drive)
        alpha = -np.expm1(-dt / .1)
        for n in range(len(rates) - 1):
            rates[n + 1] = rates[n] + alpha * (np.maximum(0, drive[n]) - rates[n])
        return rates, target

    output, record = begin(args.output, "home_vector_ridge_v1", source=str(Path(args.source).resolve()),
                           source_manifest_sha256=digest(Path(args.source) / "manifest.json"))
    try:
        report = {}
        for mode in ["current_inputs", "legacy_frozen", "initialized_frozen"]:
            # Accumulate statistics a trajectory at a time; never cache the full PFN dataset.
            first, second = [], []
            for row in groups["train"]:
                x, _ = features(row, mode)
                first.append(x.mean(0))
                second.append((x ** 2).mean(0))
            mean = np.mean(first, axis=0)
            scale = np.sqrt(np.maximum(np.mean(second, axis=0) - mean ** 2, 1e-12))

            def design(x):
                return np.column_stack(((x - mean) / scale, np.ones(len(x))))

            width = len(mean) + 1
            gram, cross = np.zeros((width, width)), np.zeros((width, 2))
            for row in groups["train"]:
                x, y = features(row, mode)
                z = design(x)
                gram += z.T @ z / len(z) / len(groups["train"])
                cross += z.T @ y / len(z) / len(groups["train"])
            candidates = []
            for strength in [1e-6, 1e-4, 1e-2, 1.]:
                penalty = np.diag([strength] * (width - 1) + [0.])
                weights = np.linalg.solve(gram + penalty, cross)
                candidates.append([0., strength, weights])
            for row in groups["validation"]:
                x, y = features(row, mode)
                for candidate in candidates:
                    candidate[0] += np.mean(np.sum((design(x) @ candidate[2] - y) ** 2, axis=1))
            _, ridge, weights = min(candidates, key=lambda v: v[0])
            scores = {}
            for split, rows in groups.items():
                predictions, targets = [], []
                for row in rows:
                    x, y = features(row, mode)
                    predictions.append(design(x) @ weights)
                    targets.append(y)
                scores[split] = metrics(predictions, targets)
            report[mode] = dict(ridge=ridge, splits=scores)
            np.savez(output / f"{mode}.npz", weights=weights, mean=mean, scale=scale)
        write_json(output / "metrics.json", report)
        finish(output, record)
    except Exception as error:
        fail(output, record, error)
        raise
    return report


def batch(circuit, records, dt, device, dtype):
    import torch
    samples = [sample_protocol(r["protocol"], dt) for r in records]
    length = max(len(x) for x, _ in samples)
    pfn = np.zeros((len(records), length, len(circuit.input_ids)))
    target, mask = np.zeros((len(records), length, 2)), np.zeros((len(records), length))
    for i, (controls, y) in enumerate(samples):
        n = len(controls)
        pfn[i, :n], target[i, :n], mask[i, :n] = encode(circuit, controls), y, 1
    return tuple(torch.as_tensor(x, device=device, dtype=dtype) for x in [pfn, target, mask])


def evaluate(network, circuit, records, dt, device, scale):
    import torch
    predictions, targets, late_p, late_y, per_trajectory = [], [], [], [], []
    states = []
    with torch.no_grad():
        for row in records:
            x, y, mask = batch(circuit, [row], dt, device, network.a.dtype)
            prediction, rates = network(x, dt)
            p, target = prediction[0].cpu().numpy(), y[0].cpu().numpy()
            if not np.isfinite(p).all() or not torch.isfinite(rates).all() or rates.max() > 1e6:
                raise ValueError("Evaluation exceeded numerical guard.")
            predictions.append(p)
            targets.append(target)
            states.append(rates[0].cpu().numpy())
            per_trajectory.append(dict(family=row["family"], kind=row["kind"],
                mse_mm2=float(np.mean(np.sum((p - target) ** 2, axis=1))),
                terminal_error_mm=float(np.linalg.norm(p[-1] - target[-1]))))
            if row["kind"] == "matched_pause" and row["protocol"][-1][0] - row["suffix_start_s"] == 5:
                count = int(round(1 / dt)) + 1
                late_p.append(p[-count:])
                late_y.append(target[-count:])
    result = metrics(predictions, targets)
    result["equal_trajectory_scaled_mse"] = float(np.mean([r["mse_mm2"] for r in per_trajectory]) / scale ** 2)
    result["late_5s_pause"] = metrics(late_p, late_y) if late_p else None
    result["per_trajectory"] = per_trajectory
    return result, predictions, targets, states


def train(args):
    import torch
    from .model import HomeVectorNetwork, objective
    from .training_batches import backward_grouped

    batching = getattr(args, "batching", "padded")
    length_ratio = getattr(args, "length_ratio", 2.)
    if batching not in ("padded", "length-groups") or not np.isfinite(length_ratio) or length_ratio < 1:
        raise ValueError("Require padded/length-groups batching and finite length ratio >=1.")

    if (args.updates < 1 or args.batch_size < 1 or args.max_seconds <= 0
            or args.validate_every < 1 or args.threads < 1 or not np.isfinite(args.max_seconds)):
        raise ValueError("Require positive updates, batch size and wall-time cap.")
    circuit, groups, task = load(args.source)
    diagnostics = None
    if args.rewire_seed is not None:
        circuit, diagnostics = rewire(circuit, args.rewire_seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    network = HomeVectorNetwork(circuit, args.seed, recurrence=not args.no_recurrence).to(args.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=.001)
    output, record = begin(args.output, "home_vector_training_v1", source=str(Path(args.source).resolve()),
        source_manifest_sha256=digest(Path(args.source) / "manifest.json"),
        seed=args.seed, rewire_seed=args.rewire_seed, recurrence=not args.no_recurrence,
        updates=args.updates, batch_size=args.batch_size, max_seconds=args.max_seconds,
        device=args.device, threads=args.threads, benchmark=args.benchmark,
        batching=batching, length_ratio=length_ratio)
    record.update(torch=torch.__version__, parameter_count=sum(p.numel() for p in network.parameters()))
    write_json(output / "manifest.json", record)
    try:
        write_json(output / "circuit.json", circuit.as_dict())
        write_json(output / "rewire_diagnostics.json", diagnostics)
        write_json(output / "training_config.json", dict(
            optimizer="Adam", learning_rate=.001, betas=[.9, .999], eps=1e-8,
            gradient_norm_clip=1., validation_every=args.validate_every,
            batching=dict(mode=batching, max_length_ratio=length_ratio,
                          sampling="Uniform original batch; group gradients weighted by trajectory count; one optimizer step"),
            loss=dict(prediction="equal-trajectory mean of squared Cartesian error / L^2",
                      activity=1e-5, edge_initialization_prior=1e-4, decoder_weights=1e-5),
            bounds=dict(edge_coefficient=[-5, 5], bias=[-2, 2], tau_s=[.05, .5]),
            initial=dict(bias=.1, tau_s=.1, decoder=0., input_edge_coefficient=1.,
                         recurrent_coefficient="seeded Normal(0,0.1), shared within each type pair"),
            state="zero at initialization only; full-trajectory BPTT; no slow state",
            population_types=list(circuit.population_types),
            normalization="counts / mean projection input across all selected receivers",
            numerical_guard_max_rate=1e6, dtype="float32", dt_s=task["dt_s"],
            length_scale_mm=task["length_scale_mm"]))
        history, best, completed = [], float("inf"), 0
        started = time.monotonic()
        for update in range(args.updates):
            # A wall cap is checked between full-trajectory updates, never by cutting history.
            if update and time.monotonic() - started >= args.max_seconds:
                break
            selected = [groups["train"][i] for i in rng.integers(0, len(groups["train"]), args.batch_size)]
            grouping = None
            if batching == "padded":
                x, target, mask = batch(circuit, selected, task["dt_s"], args.device, network.a.dtype)
                optimizer.zero_grad(set_to_none=True)
                prediction, rates = network(x, task["dt_s"])
                loss, terms = objective(prediction, target, rates, mask, task["length_scale_mm"], network)
                if not torch.isfinite(loss) or not torch.isfinite(rates).all() or rates.max() > 1e6:
                    raise ValueError("Training exceeded numerical guard.")
                loss.backward()
            else:
                optimizer.zero_grad(set_to_none=True)
                loss, terms, grouping = backward_grouped(network, circuit, selected, task["dt_s"],
                    args.device, task["length_scale_mm"], length_ratio)
            norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            network.project_parameters()
            completed = update + 1
            row = dict(update=completed, elapsed_s=time.monotonic() - started, loss=float(loss.detach()),
                       gradient_norm=float(norm), **{k: float(v.detach()) for k, v in terms.items()})
            if grouping is not None:
                row["grouping"] = grouping
            # Benchmarks time updates only; their checkpoint is explicitly unselected.
            validate = not args.benchmark and (completed % args.validate_every == 0 or completed == args.updates)
            if validate:
                result, _, _, _ = evaluate(network, circuit, groups["validation"], task["dt_s"],
                                           args.device, task["length_scale_mm"])
                row["validation_scaled_mse"] = result["equal_trajectory_scaled_mse"]
                if row["validation_scaled_mse"] < best:
                    best = row["validation_scaled_mse"]
                    torch.save(dict(model=network.state_dict(), optimizer=optimizer.state_dict(), update=completed,
                        numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state()), output / "best.pt")
            history.append(row)
            # Flush progress for interrupted runs as well as successful ones.
            write_json(output / "history.json", history)
        torch.save(dict(model=network.state_dict(), optimizer=optimizer.state_dict(), update=completed,
                       numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state()), output / "last.pt")
        benchmark = dict(completed_updates=completed, elapsed_s=time.monotonic() - started,
                         seconds_per_update=(time.monotonic() - started) / completed,
                         purpose="engineering benchmark" if args.benchmark else "pilot training",
                         stopping_reason="update_limit" if completed == args.updates else "wall_time_cap")
        write_json(output / "timing.json", benchmark)
        if not args.benchmark:
            # A time cap can occur before a scheduled validation; select the available final checkpoint.
            if not (output / "best.pt").exists():
                result, _, _, _ = evaluate(network, circuit, groups["validation"], task["dt_s"], args.device,
                                          task["length_scale_mm"])
                best = result["equal_trajectory_scaled_mse"]
                torch.save(dict(model=network.state_dict(), optimizer=optimizer.state_dict(), update=completed,
                    numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state()), output / "best.pt")
            checkpoint = torch.load(output / "best.pt", weights_only=False, map_location=args.device)
            network.load_state_dict(checkpoint["model"])
            report = {}
            for split in ["validation", "test", "challenge"]:
                report[split], predictions, targets, states = evaluate(network, circuit, groups[split],
                    task["dt_s"], args.device, task["length_scale_mm"])
                np.savez_compressed(output / f"{split}_predictions.npz", **{
                    f"{kind}_{i}": array for kind, arrays in [("prediction", predictions), ("target", targets)]
                    for i, array in enumerate(arrays)})
            # Half-step validation uses only validation records, at identical physical timestamps.
            _, _, _, coarse_r = evaluate(network, circuit, groups["validation"], task["dt_s"],
                                                args.device, task["length_scale_mm"])
            fine_result, fine_p, _, fine_r = evaluate(network, circuit, groups["validation"], task["dt_s"] / 2,
                                                     args.device, task["length_scale_mm"])
            delta = sum(np.sum((a - b[::2]) ** 2) for a, b in zip(coarse_r, fine_r))
            energy = sum(np.sum(b[::2] ** 2) for b in fine_r)
            state_error = float(np.sqrt(delta / max(energy, 1e-24)))
            coarse_error = report["validation"]["cartesian_rmse_mm"]
            task_change = abs(fine_result["cartesian_rmse_mm"] - coarse_error) / max(coarse_error, 1e-12)
            report["numerical_validation"] = dict(state_relative_rms_error=state_error,
                relative_rmse_change=task_change, passed=bool(state_error < .02 and task_change < .05))
            report["selected_update"] = checkpoint["update"]
            report["bounds"] = dict(tau_s=network.tau.detach().cpu().tolist(),
                bias=network.bias.detach().cpu().tolist(),
                max_abs_coefficient=float(network.a.detach().abs().max()),
                sign_changes=int(((network.a * network.a_initial) < 0).sum()),
                coefficient_bound_hits=int((network.a.detach().abs() >= 5).sum()))
            report["interpretation"] = "Single pilot; no anatomical-specificity or biological claim."
            if not report["numerical_validation"]["passed"]:
                report["interpretation"] = "Numerical convergence gate failed; task performance is unvalidated."
            write_json(output / "metrics.json", report)
        finish(output, record)
    except Exception as error:
        fail(output, record, error)
        raise
    return dict(output=str(output), **benchmark)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--movement", default="results/pfn-hdeltab-movement-groupnorm-pfnd6-v1.0")
    prep.add_argument("--census", default="results/hdeltab-extension-census-final-20261024")
    prep.add_argument("--inputs", default="results/vdeltae-inputs-live-20261024")
    prep.add_argument("--hdeltac", help="Verified bounded hDeltaC anatomy bundle; omitted keeps the original core")
    prep.add_argument("--seed", type=int, default=20261030)
    prep.add_argument("--dt", type=float, default=.0025)
    for split, count in [("train", 1024), ("validation", 128), ("test", 256)]:
        prep.add_argument(f"--{split}", type=int, default=count)
    prep.add_argument("--output", required=True)
    prep.set_defaults(action=prepare)
    ridge = sub.add_parser("baselines")
    ridge.add_argument("source")
    ridge.add_argument("--output", required=True)
    ridge.set_defaults(action=baseline)
    fit = sub.add_parser("train")
    fit.add_argument("source")
    fit.add_argument("--output", required=True)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--rewire-seed", type=int)
    fit.add_argument("--no-recurrence", action="store_true")
    fit.add_argument("--updates", type=int, default=2000)
    fit.add_argument("--batch-size", type=int, default=8)
    fit.add_argument("--max-seconds", type=float, default=7200)
    fit.add_argument("--validate-every", type=int, default=50)
    fit.add_argument("--threads", type=int, default=1)
    fit.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    fit.add_argument("--benchmark", action="store_true")
    fit.add_argument("--batching", choices=["padded", "length-groups"], default="padded")
    fit.add_argument("--length-ratio", type=float, default=2., help="Maximum sample-length ratio within a subgroup")
    fit.set_defaults(action=train)
    verify = sub.add_parser("verify")
    verify.add_argument("source")
    verify.set_defaults(action=lambda a: dict(status=verify_bundle(a.source)["status"]))
    args = parser.parse_args(argv)
    print(json.dumps(args.action(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
