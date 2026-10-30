"""Offline descriptive exports for the anatomically aligned stage-1 experiments."""

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from .connectivity import ValidationError
from .inputs import save_json, sha256
from .vdeltae_spatial_pilot import pilot_code_provenance, record_artifact_hashes
from .vdeltae_stage1 import SCENARIOS, _protection, verify_stage1


def offset_heatmap(responses, bodies):
    sources = bodies.loc[bodies.type.eq("vDeltaE")].sort_values(["annotation_column", "bodyId"])
    targets = bodies.loc[bodies.type.eq("hDeltaG")].sort_values(["annotation_column", "bodyId"])
    matrices = {}
    for scenario in SCENARIOS:
        for kind in ["pulse", "sustained"]:
            rows = responses.loc[responses.scenario.eq(scenario) & responses.kind.eq(kind)].copy()
            rows = rows.loc[rows.experiment.str.startswith("body_")]
            rows["source"] = rows.experiment.str.split("_").str[1].astype(int)
            matrices[scenario, kind] = rows.pivot(index="source", columns="bodyId", values="delta_at_offset").reindex(
                index=sources.bodyId, columns=targets.bodyId)
    limit = max(float(np.abs(table.to_numpy()).max()) for table in matrices.values()) or 1.
    figure, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
    for row, scenario in enumerate(SCENARIOS):
        for column, kind in enumerate(["pulse", "sustained"]):
            ax = axes[row, column]
            image = ax.imshow(matrices[scenario, kind], aspect="auto", cmap="RdBu_r",
                              norm=TwoSlopeNorm(vmin=-limit, vcenter=0., vmax=limit))
            ax.set_title(f"{scenario.replace('_', ' ')} · {kind}")
            ax.set_xticks(range(len(targets)), [f"C{v.annotation_column}\n{v.bodyId}" for v in targets.itertuples()], fontsize=8)
            ax.set_yticks(range(len(sources)), [f"C{v.annotation_column} · {v.bodyId}" for v in sources.itertuples()], fontsize=8)
    figure.supxlabel("Receiving hDeltaG body (annotation labels; no environmental angles)")
    figure.supylabel("Stimulated vDeltaE body")
    figure.suptitle("Control-relative hDeltaG rate at stimulus offset")
    figure.colorbar(image, ax=axes, label="Δ rate (arbitrary units)", shrink=.75)
    return figure, matrices


def trace_figure(directory, bodies, spec):
    directory = Path(directory)
    target = bodies.type.eq("hDeltaG").to_numpy()
    targets = bodies.loc[target]
    colors = plt.get_cmap("tab10")(np.arange(len(targets)))
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
    for column, scenario in enumerate(SCENARIOS):
        with np.load(directory / f"{scenario}__{spec['name']}.npz") as data, np.load(
                directory / f"{scenario}__tonic_control.npz") as control:
            values = [data["rates"][:, target], data["rates"][:, target] - control["rates"][:, target],
                      data["excitability"][:, target] - control["excitability"][:, target]]
            for row, array in enumerate(values):
                ax = axes[row, column]
                for i, body in enumerate(targets.itertuples()):
                    ax.plot(data["time_s"], array[:, i], color=colors[i], lw=.9,
                            label=f"C{body.annotation_column} · {body.bodyId}")
                ax.plot(data["time_s"], array.mean(axis=1), color="black", lw=1.5, label="Population mean")
                if spec["start_s"] is not None:
                    ax.axvspan(spec["start_s"], spec["stop_s"], color="grey", alpha=.15)
                ax.axhline(0., color="grey", lw=.5)
                ax.set_ylabel(["Rate", "Δ rate vs tonic", "Δ slow offset vs tonic"][row])
            axes[0, column].set_title(scenario.replace("_", " "))
            axes[-1, column].set_xlabel("Time (s)")
    axes[0, 1].legend(fontsize=7, ncol=3)
    figure.suptitle(f"{spec['name']} · arbitrary units · shaded interval: added input")
    return figure


def export_stage1_figures(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    before = sha256(run / "stage1.json")
    record = verify_stage1(run)
    source = Path(record["source"])
    _protection(source, json.loads((source / "contacts.json").read_text()), output)
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in [run, source]):
        raise ValidationError("Figure output must be separate from simulation and contact sources.")
    bodies = pd.read_csv(run / "selected_bodies.csv")
    responses = pd.read_csv(run / "response_by_body.csv")
    specs = json.loads((run / "experiments.json").read_text())
    source_ids = bodies.loc[bodies.type.eq("vDeltaE")].sort_values(["annotation_column", "bodyId"])
    # Selection depends only on annotations/body IDs, never observed response strength.
    selected_ids = source_ids.loc[source_ids.annotation_column.eq(0), "bodyId"].tolist()
    selected_ids += source_ids.loc[source_ids.annotation_column.isin([1, 5])].groupby("annotation_column").bodyId.min().tolist()
    names = {"tonic_control", "population_pulse", "population_sustained"}
    names |= {f"body_{body}_{kind}" for body in selected_ids for kind in ["pulse", "sustained"]}
    selected = [spec for spec in specs if spec["name"] in names]
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(kind="vdeltae_hdeltag_stage1_figures_v1", status="running", source=str(run),
                    source_manifest_sha256=before, selected_experiments=[spec["name"] for spec in selected],
                    selection_rule="Both C0 bodies; smallest body in C1 and C5; population and tonic controls.",
                    started_at_utc=datetime.now(timezone.utc).isoformat(), **pilot_code_provenance())
    save_json(output / "plots.json", manifest)
    try:
        figure, matrices = offset_heatmap(responses, bodies)
        for extension in ["png", "pdf"]:
            figure.savefig(output / f"offset_responses.{extension}", dpi=180)
        plt.close(figure)
        for (scenario, kind), table in matrices.items():
            table.to_csv(output / f"{scenario}_{kind}_offset_matrix.csv")
        for spec in selected:
            figure = trace_figure(run, bodies, spec)
            for extension in ["png", "pdf"]:
                figure.savefig(output / f"{spec['name']}.{extension}", dpi=150)
            plt.close(figure)
        if sha256(run / "stage1.json") != before:
            raise ValidationError("Simulation manifest changed during export.")
        verify_stage1(run)
        (output / "README.md").write_text(
            "# Stage-1 response figures\n\n"
            "Offset heatmaps include every single-body experiment, with a shared signed color scale.\n"
            "Trace panels show raw rates, control-relative rates, and control-relative slow offsets.\n"
            "Representative trace selection is fixed by annotation/body ID; every experiment remains in the source NPZ files.\n"
            "C0 bodies remain separate. Column labels are anatomical annotations, without assigned environmental angles.\n"
            "Slow tails follow the assumed slow-state equation; these plots do not demonstrate biological integration or a home vector.\n")
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["artifact_sha256"] = record_artifact_hashes(output, "plots.json")
        save_json(output / "plots.json", manifest)
    return manifest


def verify_stage1_figures(directory):
    directory = Path(directory).resolve()
    record = json.loads((directory / "plots.json").read_text())
    if record.get("kind") != "vdeltae_hdeltag_stage1_figures_v1" or record.get("status") != "complete":
        raise ValidationError("Stage-1 figures are not complete.")
    if record.get("artifact_sha256") != record_artifact_hashes(directory, "plots.json"):
        raise ValidationError("Stage-1 figure hashes or inventory changed.")
    source = Path(record["source"])
    if sha256(source / "stage1.json") != record["source_manifest_sha256"]:
        raise ValidationError("Stage-1 figure source manifest changed.")
    verify_stage1(source)
    return record
