"""Three-panel event display, following the HGTD clustering display.

Same track and jet selections, same three panels as the original:

    rz_display      tracks, jets and vertices in the r-z plane
    zt_display      track time against z0, coloured by time cluster
    time_histogram  pT-weighted stacked histogram of the track times

What changed, and why:

- Reading is from the event store rather than the ntuple, which makes an
  event a lookup rather than a scan and removes the need for a pickled
  event index.
- Clustering is the simple 100 ps windowing. The ROOT macro is not
  available here.
- The original stood a calorimeter time in with ``random.gauss(truth, 90)``,
  which is truth in a display. Where a model's predictions are given its
  answer and its predicted sigma are drawn instead; where they are not, the
  band is simply absent.

    python -m src.evaluation.event_display --config config/blocks/lar_hgtd.yaml \\
        --event 145440 --out ../displays

    # choose the events from a model's own failures
    python -m src.evaluation.event_display --config ... \\
        --model-dir ../runs/lar_hgtd/trial_000 --worst 5 --out ../displays

    # the events HGTD alone gets wrong and LAr+HGTD recovers, both drawn
    python -m src.evaluation.event_display --config ... --apply-cuts \\
        --model-dir ../runs/lar_hgtd/trial_000 --model-dir ../runs/hgtd_only/trial_000 \\
        --recovered 5 --out ../displays/recovered
"""

from __future__ import annotations

import argparse
import os
from itertools import chain
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..pipeline.event_store import EventStore
from . import plots

# Cluster colours, as in the original.
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
          "#ff7f00", "#a65628", "#f781bf", "#999999"]

TRACK_FIELDS = ("pt", "eta", "phi", "z0", "var_z0", "time", "time_res",
                "quality", "has_valid_time", "truth_vtx_idx")
JET_FIELDS = ("pt", "eta", "phi", "n_truth_hs_jets")

# VBF H->inv Run 4 event selection, on truth hard-scatter jets.
VBF_DZ_MM = 2.0
VBF_JET_PT = 30.0
VBF_DETA = 3.0
VBF_FWD_ETA = (2.38, 4.0)

IDEAL_EFF = False           # True drops the valid-time requirement on tracks
HGTD_ETA = (2.38, 4.0)
TRACK_PT = (1.0, 30.0)
JET_PT_MIN = 30.0
CLUSTER_WINDOW_PS = 100.0


# --- reading ---------------------------------------------------------------

def load_event(store: EventStore, index: int) -> Dict:
    """One event's tracks, jets and vertices, straight from the store."""
    ev: Dict = {k: store.event_column(k)[index] for k in
                ("event_number", "mu", "truth_vtx_time", "truth_vtx_z",
                 "reco_vtx_time", "reco_vtx_time_res", "reco_vtx_z")}
    ev["sample"] = store.sample
    for block, fields in (("tracks", TRACK_FIELDS),
                          ("jets_emtopo", JET_FIELDS),
                          ("reco_vertices", ("z", "time", "is_hs",
                                             "has_valid_time")),
                          ("truth_vertices", ("z", "time", "is_hs")),
                          ("truth_hs_jets", ("pt", "eta", "phi"))):
        blk = store.block(block, list(fields))
        lo, hi = blk.offsets[index], blk.offsets[index + 1]
        ev[block] = {f: blk[f][lo:hi] for f in fields}
    return ev


def event_selection(ev: Dict) -> List[str]:
    """VBF H->invisible Run 4 event selection; returns the cuts it failed.

      1. |dz(reco HS, truth HS)| < 2 mm
      2. at least two truth HS jets, leading two both above 30 GeV
      3. |d_eta| between the leading two above 3.0
      4. one of the leading two inside the HGTD forward acceptance
    """
    failed = []
    dz = abs(float(ev["reco_vtx_z"]) - float(ev["truth_vtx_z"]))
    if dz >= VBF_DZ_MM:
        failed.append(f"|dz(reco,truth)| = {dz:.2f} mm >= {VBF_DZ_MM} mm")

    pt = np.asarray(ev["truth_hs_jets"]["pt"], dtype=float)
    eta = np.asarray(ev["truth_hs_jets"]["eta"], dtype=float)
    if len(pt) < 2:
        failed.append(f"only {len(pt)} truth HS jet(s), need >= 2")
        return failed

    lead, sub = np.argsort(pt)[::-1][:2]
    if pt[lead] < VBF_JET_PT or pt[sub] < VBF_JET_PT:
        failed.append(f"leading jets pT = {pt[lead]:.1f}, {pt[sub]:.1f} GeV "
                      f"(both must exceed {VBF_JET_PT})")
    deta = abs(eta[lead] - eta[sub])
    if deta <= VBF_DETA:
        failed.append(f"|d_eta(lead,sublead)| = {deta:.2f} <= {VBF_DETA}")
    forward = [e for e in (eta[lead], eta[sub])
               if VBF_FWD_ETA[0] < abs(e) < VBF_FWD_ETA[1]]
    if not forward:
        failed.append(f"no leading jet in HGTD acceptance "
                      f"(|eta| = {abs(eta[lead]):.2f}, {abs(eta[sub]):.2f})")
    return failed


def passes_vbf(store: EventStore) -> np.ndarray:
    """The VBF selection for every event in a store at once.

    The per-event ``event_selection`` is for explaining one event; this is
    for choosing which events to look at, where the ranking has to happen
    among the survivors rather than the other way round -- the worst-predicted
    events are the misidentified ones, and the first cut removes exactly
    those.

    Vectorised: the jets are lexsorted by descending pT within each event, so
    the leading two are the first two of each group.
    """
    dz = np.abs(store.event_column("reco_vtx_z")
                - store.event_column("truth_vtx_z"))
    keep = dz < VBF_DZ_MM

    blk = store.block("truth_hs_jets", ["pt", "eta"])
    order = np.lexsort((-blk["pt"], blk.event_index()))
    pt, eta = blk["pt"][order], blk["eta"][order]
    start, counts = blk.offsets[:-1], np.diff(blk.offsets)

    two = counts >= 2
    keep &= two
    i, j = start[two], start[two] + 1
    lead_pt, sub_pt = pt[i], pt[j]
    lead_eta, sub_eta = eta[i], eta[j]

    fwd = lambda e: (np.abs(e) > VBF_FWD_ETA[0]) & (np.abs(e) < VBF_FWD_ETA[1])
    keep[two] &= ((lead_pt >= VBF_JET_PT) & (sub_pt >= VBF_JET_PT)
                  & (np.abs(lead_eta - sub_eta) > VBF_DETA)
                  & (fwd(lead_eta) | fwd(sub_eta)))
    return keep


# --- selection, as in the original -----------------------------------------

def connected_tracks(ev: Dict) -> np.ndarray:
    """Tracks in HGTD acceptance attached to the hard-scatter vertex.

    quality == 1, 1 < pT < 30 GeV, 2.38 < |eta| < 4.0, a valid time, and
    within 3 sigma in z0 of the reconstructed vertex.
    """
    t = ev["tracks"]
    nsigma = np.abs(t["z0"] - float(ev["reco_vtx_z"])) / np.sqrt(t["var_z0"])
    keep = ((t["quality"] == 1)
            & (t["pt"] > TRACK_PT[0]) & (t["pt"] < TRACK_PT[1])
            & (np.abs(t["eta"]) > HGTD_ETA[0])
            & (np.abs(t["eta"]) < HGTD_ETA[1])
            & (nsigma < 3.0))
    if not IDEAL_EFF:
        keep &= t["has_valid_time"] == 1
    return np.flatnonzero(keep)


def track_info(ev: Dict, idx: np.ndarray) -> List[Dict]:
    """Per-track drawing data, with the r-z projection of its momentum."""
    t, truth_is_hs = ev["tracks"], ev["truth_vertices"]["is_hs"]
    out = []
    for i in idx:
        eta, phi, pt = float(t["eta"][i]), float(t["phi"][i]), float(t["pt"][i])
        vtx = int(t["truth_vtx_idx"][i])
        theta = np.arctan(pt / abs(pt * np.sinh(eta)))
        out.append({
            "idx": int(i), "pt": pt, "z0": float(t["z0"][i]),
            "x": (pt / 2) * np.cos(theta) * (np.sign(eta) or 1),
            "y": (pt / 2) * np.sin(theta) * (np.sign(np.sin(phi)) or 1),
            "stat": int(truth_is_hs[vtx]) if 0 <= vtx < len(truth_is_hs) else 0,
            "time": float(t["time"][i]), "time_res": float(t["time_res"][i]),
            "var_z0": float(t["var_z0"][i]),
        })
    return out


def jet_info(ev: Dict) -> List[Dict]:
    """Jets above 30 GeV, with the same r-z projection."""
    j = ev["jets_emtopo"]
    out = []
    for i in range(len(j["pt"])):
        pt, eta, phi = float(j["pt"][i]), float(j["eta"][i]), float(j["phi"][i])
        if pt < JET_PT_MIN:
            continue
        theta = np.arctan(pt / abs(pt * np.sinh(eta)))
        out.append({
            "pt": pt, "eta": eta, "phi": phi,
            "isHS": int(j["n_truth_hs_jets"][i]),
            "x": (pt / 40) * np.cos(theta) * (np.sign(eta) or 1),
            "y": (pt / 40) * np.sin(theta) * (np.sign(np.sin(phi)) or 1),
        })
    return out


def cluster_tracks(tracks: List[Dict]) -> Dict:
    """Group tracks into 100 ps windows, weighting z by 1/var(z0)."""
    out: Dict[str, list] = {"track_clusters": [], "cluster_times": [],
                            "cluster_zs": [], "pt_weights": [],
                            "hs_times": [], "hs_zs": []}
    groups: List[List[Dict]] = []
    for t in sorted(tracks, key=lambda x: x["time"]):
        if groups and abs(t["time"] - groups[-1][0]["time"]) < CLUSTER_WINDOW_PS:
            groups[-1].append(t)
        else:
            groups.append([t])

    for g in groups:
        w = 1.0 / np.array([t["var_z0"] for t in g])
        out["track_clusters"].append([t["idx"] for t in g])
        out["cluster_times"].append(float(np.mean([t["time"] for t in g])))
        out["cluster_zs"].append(
            float(np.sum(np.array([t["z0"] for t in g]) * w) / np.sum(w)))
        out["pt_weights"].append([t["pt"] for t in g])
        for t in g:
            if t["stat"] == 1:
                out["hs_times"].append(t["time"])
                out["hs_zs"].append(t["z0"])
    return out


# --- panels ----------------------------------------------------------------

def _eta_lines(ax, z, eta_ref=2.38, length=50):
    theta = 2 * np.arctan(np.exp(-abs(eta_ref)))
    dx, dy = length * np.cos(theta), length * np.sin(theta)
    for sx in (dx, -dx):
        for sy in (dy, -dy):
            ax.plot([z, z + sx], [0, sy], linestyle="dotted", color="lightgrey")


MODEL_STYLES = (("black", "-"), ("#6e6e6e", "--"), ("#6e6e6e", ":"))


def _time_markers(ax, ev: Dict, predictions: Sequence[tuple] = ()) -> None:
    """Truth, reconstructed and predicted times; shared by both time panels.

    ``predictions`` is [(label, mu, sigma)], one per model, drawn in
    ``MODEL_STYLES`` order so the first model is the solid black one.
    """
    ax.axvline(float(ev["truth_vtx_time"]), color="blue", linestyle="--",
               linewidth=2, alpha=0.8, label="Truth HS time", zorder=4)
    if float(ev["reco_vtx_time_res"]) < 1000:
        ax.axvline(float(ev["reco_vtx_time"]), color="green", linestyle="--",
                   linewidth=2, alpha=0.8, label="HGTD reco time", zorder=4)
    for (label, mu, sigma), (colour, style) in zip(predictions, MODEL_STYLES):
        ax.axvspan(mu - sigma, mu + sigma, color=colour, alpha=0.10, zorder=1)
        ax.axvline(mu, color=colour, linestyle=style, linewidth=2, zorder=4,
                   label=rf"{label} {mu:.0f} $\pm$ {sigma:.0f} ps")


def plot_rz(ax, tracks, jets, ev):
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    z0, tz = float(ev["reco_vtx_z"]), float(ev["truth_vtx_z"])
    for t in tracks:
        ax.plot([t["z0"], t["z0"] + t["x"]], [0, t["y"]],
                color="blue" if t["stat"] == 1 else "red", alpha=0.7)

    for i, j in enumerate(jets):
        c = "green" if j["isHS"] >= 1 else "grey"
        ax.fill([z0, z0 + j["x"] - 0.15 * j["y"], z0 + j["x"] + 0.15 * j["y"]],
                [0, j["y"] + 0.15 * j["x"], j["y"] - 0.15 * j["x"]],
                color=c, alpha=0.5)
        ax.text(z0 - 6.8, -0.30 - i * 0.09,
                rf"Jet {i+1}: $p_T$={j['pt']:.0f} GeV, $\eta$={j['eta']:.1f}",
                weight="bold", fontsize=11,
                color="green" if c == "green" else "black")

    _eta_lines(ax, z0)
    rv, tv = ev["reco_vertices"], ev["truth_vertices"]
    ax.scatter(rv["z"], np.zeros(len(rv["z"])), marker="o", s=90, zorder=10,
               color=["blue" if h else "black" for h in rv["is_hs"]])
    ax.scatter(tv["z"], np.full(len(tv["z"]), -0.75), marker="|", s=90, zorder=10,
               color=["blue" if h else "black" for h in tv["is_hs"]])

    ax.set_xlim(z0 - 7.0, z0 + 7.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([])
    ax.axhline(0.0, color="black", linestyle="--", alpha=0.5)
    ax.axhline(-0.75, color="black", linestyle="--", alpha=0.5)
    ax.text(z0 + 4.2, 0.05, "Reco vertices", fontsize=11)
    ax.text(z0 + 4.2, -0.70, "Truth vertices", fontsize=11)
    ax.text(z0 - 6.8, 0.90,
            f"Reco HS  (z,t) = ({z0:.1f} mm, {float(ev['reco_vtx_time']):.1f} ps)",
            weight="bold", fontsize=11)
    ax.text(z0 - 6.8, 0.80,
            f"Truth HS (z,t) = ({tz:.1f} mm, {float(ev['truth_vtx_time']):.1f} ps)",
            weight="bold", fontsize=11)
    ax.text(z0 - 6.8, 0.70,
            f"{ev['sample']}, event {int(ev['event_number'])}",
            weight="bold", fontsize=11, color="darkred")
    ax.set_xlabel("Z [mm]")
    ax.set_title("R-Z event display")
    ax.legend([mlines.Line2D([], [], color="blue"),
               mlines.Line2D([], [], color="red"),
               mpatches.Rectangle((0, 0), 1, 1, color="green", alpha=0.5),
               mpatches.Rectangle((0, 0), 1, 1, color="grey", alpha=0.5)],
              ["Hard scatter track", "Pile-up track", "HS jet", "PU jet"],
              loc="upper left", bbox_to_anchor=(0.0, 0.66), fontsize=10)


def plot_zt(ax, tracks, ev, cl, colors, predictions=()):
    by_idx = {t["idx"]: t for t in tracks}
    for i, cluster in enumerate(cl["track_clusters"]):
        g = [by_idx[j] for j in cluster]
        ax.errorbar([t["time"] for t in g], [t["z0"] for t in g],
                    xerr=[t["time_res"] for t in g],
                    yerr=[np.sqrt(t["var_z0"]) for t in g],
                    fmt=".", color=colors[i % len(colors)], alpha=0.7, zorder=2)
    if cl["hs_times"]:
        ax.scatter(cl["hs_times"], cl["hs_zs"], marker="o", edgecolors="black",
                   s=50, color="blue", label="Hard scatter tracks", zorder=15)
    ax.axhline(float(ev["reco_vtx_z"]), color="black", linestyle="--",
               linewidth=2, label="Reco vertex Z", zorder=4)
    if cl["cluster_times"]:
        ax.scatter(cl["cluster_times"], cl["cluster_zs"], marker="*",
                   edgecolors="black", s=400, zorder=10,
                   color=[colors[i % len(colors)]
                          for i in range(len(cl["cluster_times"]))],
                   label="Cluster positions")
    _time_markers(ax, ev, predictions)
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Z [mm]")
    ax.set_title("Z-T event display")
    ax.legend(loc="upper left", fontsize=10)


def plot_time_hist(ax, tracks, ev, cl, colors, predictions=()):
    by_idx = {t["idx"]: t for t in tracks}
    per = [[by_idx[j]["time"] for j in c] for c in cl["track_clusters"]]
    if not per:
        return
    flat = list(chain.from_iterable(per)) + [float(ev["truth_vtx_time"])]
    lo, hi = min(flat), max(flat)
    pad = 0.05 * (hi - lo) or 50.0
    counts, _, _ = ax.hist(per, bins=50, stacked=True, range=(lo - pad, hi + pad),
                           weights=cl["pt_weights"], color=colors[:len(per)],
                           alpha=0.8, label="Track time")
    top = float(np.max(counts))
    for t in cl["hs_times"]:
        ax.text(t, 0, "/", ha="center", va="top", fontsize=18, color="blue")
    if cl["cluster_times"]:
        ax.scatter(cl["cluster_times"], [0.1 * top] * len(cl["cluster_times"]),
                   marker="*", edgecolors="black", s=400, zorder=4,
                   color=[colors[i % len(colors)]
                          for i in range(len(cl["cluster_times"]))],
                   label="Cluster positions")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(0, 1.1 * top)
    _time_markers(ax, ev, predictions)
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel(r"Track $p_T$ [GeV]")
    ax.set_title(r"Time histogram ($p_T$ weighted)")
    ax.legend(loc="upper left", fontsize=10)


# --- driver ----------------------------------------------------------------

def display(store: EventStore, index: int, out_base: str,
            predictions: Sequence[tuple] = (), dpi: int = 300,
            apply_cuts: bool = False) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = load_event(store, index)
    if apply_cuts:
        failed = event_selection(ev)
        if failed:
            print(f"  {ev['sample']} event {int(ev['event_number'])}: skipped")
            for reason in failed:
                print(f"      {reason}")
            return None
    tracks = track_info(ev, connected_tracks(ev))
    jets = jet_info(ev)
    cl = cluster_tracks(tracks)
    colors = COLORS * (len(cl["track_clusters"]) // len(COLORS) + 1)

    event = int(ev["event_number"])
    out = os.path.join(out_base, f"{ev['sample']}_event{event}")
    os.makedirs(out, exist_ok=True)
    plots.use_style("light")
    panels = (("rz_display", lambda a: plot_rz(a, tracks, jets, ev)),
              ("zt_display", lambda a: plot_zt(a, tracks, ev, cl, colors,
                                               predictions)),
              ("time_histogram", lambda a: plot_time_hist(a, tracks, ev, cl,
                                                          colors, predictions)))
    for name, draw in panels:
        fig, ax = plt.subplots(figsize=(13, 7.5))
        draw(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{name}_event{event}.png"), dpi=dpi)
        plt.close(fig)
    print(f"  {out}   {len(tracks)} tracks, {len(jets)} jets, "
          f"{len(cl['track_clusters'])} clusters")
    return out


def _predictions(model_dir: str) -> Dict[tuple, tuple]:
    """{(sample, event number): (mu, sigma)} of a model's test predictions."""
    z = np.load(os.path.join(model_dir, "predictions_test.npz"))
    names = np.array([str(n) for n in z["dataset_names"]], dtype=object)[z["dataset_id"]]
    return {(s, int(e)): (float(m), float(g)) for s, e, m, g
            in zip(names, z["event_number"], z["y_pred"], z["sigma"])}


def _label(model_dir: str) -> str:
    """lar_hgtd for ../runs/lar_hgtd/trial_000."""
    return os.path.basename(os.path.dirname(os.path.normpath(model_dir))) or model_dir


def _report_selection(rows: Dict[str, list], eligible, total: Dict[str, int]) -> None:
    if eligible is not None:
        for s, r in rows.items():
            print(f"  {s}: {len(r):,} of {total[s]:,} candidates pass the selection")


def pick_events(model_dir: str, how: str, n: int,
                eligible: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, list]:
    """Event numbers to open, chosen from one model's own predictions.

    ``worst`` and ``best`` rank on the residual, ``unsure`` on the predicted
    sigma. ``eligible`` restricts the ranking to given event numbers per
    sample, so a selection is applied before the ranking rather than after.
    """
    z = np.load(os.path.join(model_dir, "predictions_test.npz"))
    names = [str(s) for s in z["dataset_names"]]
    key = {"worst": -np.abs(z["errors"]), "best": np.abs(z["errors"]),
           "unsure": -z["sigma"]}[how]
    rows, total = {}, {}
    for i, s in enumerate(names):
        r = np.flatnonzero(z["dataset_id"] == i)
        total[s] = len(r)
        if eligible is not None:
            r = r[np.isin(z["event_number"][r], eligible.get(s, []))]
        rows[s] = r
    _report_selection(rows, eligible, total)
    return {s: [int(z["event_number"][i]) for i in r[np.argsort(key[r])[:n]]]
            for s, r in rows.items()}


def pick_migration(combined: str, base: str, how: str, n: int,
                   eligible: Optional[Dict[str, np.ndarray]] = None,
                   fail: float = 60.0, good: float = 30.0) -> Dict[str, list]:
    """Events ``base`` gets wrong (|Delta t0| > ``fail``), by what ``combined`` does.

    ``recovered``: ``combined`` gets them right (< ``good``), largest
    recovery first. ``unrecovered``: it is wrong too (> ``fail``), largest
    remaining error first. Both runs must share a test split, as models with
    the same event selection do.
    """
    a, b = np.load(os.path.join(combined, "predictions_test.npz")), \
        np.load(os.path.join(base, "predictions_test.npz"))
    if not (np.array_equal(a["event_number"], b["event_number"])
            and np.array_equal(a["dataset_id"], b["dataset_id"])):
        raise SystemExit(f"{combined} and {base} do not share a test split")
    ea, eb = np.abs(a["errors"]), np.abs(b["errors"])
    if how == "recovered":
        sel, key = (eb > fail) & (ea < good), ea - eb
    elif how == "unrecovered":
        sel, key = (eb > fail) & (ea > fail), -ea
    else:
        raise ValueError(f"pick_migration: unknown selection {how!r}")
    rows, total = {}, {}
    for i, s in enumerate(str(x) for x in a["dataset_names"]):
        r = np.flatnonzero(sel & (a["dataset_id"] == i))
        total[s] = len(r)
        if eligible is not None:
            r = r[np.isin(a["event_number"][r], eligible.get(s, []))]
        rows[s] = r
    _report_selection(rows, eligible, total)
    return {s: [int(a["event_number"][i]) for i in r[np.argsort(key[r])[:n]]]
            for s, r in rows.items()}


RANKINGS = ("worst", "best", "unsure")        # one model's own predictions
MIGRATIONS = ("recovered", "unrecovered")     # the second model's failures, under the first


def _eligible(stores, apply_cuts: bool, min_abs_truth: Optional[float]):
    """Event numbers per sample passing the requested selection, or None."""
    if not apply_cuts and min_abs_truth is None:
        return None
    out = {}
    for n, s in stores.items():
        m = passes_vbf(s) if apply_cuts else np.ones(s.n_events, bool)
        if min_abs_truth is not None:
            m &= np.abs(s.event_column("truth_vtx_time")) >= min_abs_truth
        out[n] = s.event_column("event_number")[m]
    return out


def main():
    import yaml

    from ..pipeline.assemble import AssemblySpec

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="block config; names the stores")
    p.add_argument("--event", type=int, action="append", default=[],
                   help="event number (repeatable)")
    p.add_argument("--model-dir", action="append", default=[],
                   help="draw this model's prediction too (repeatable; the first "
                        "is the one ranked)")
    for how in RANKINGS:
        p.add_argument(f"--{how}", type=int, metavar="N",
                       help=f"open the N {how} events of the first --model-dir")
    p.add_argument("--recovered", type=int, metavar="N",
                   help="the N events the second --model-dir gets wrong (> 60 ps) and "
                        "the first gets right (< 30 ps); both must share a test split")
    p.add_argument("--unrecovered", type=int, metavar="N",
                   help="the N events both --model-dirs get wrong (> 60 ps)")
    p.add_argument("--out", default="../displays")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--ideal-eff", action="store_true",
                   help="drop the valid-time requirement on tracks")
    p.add_argument("--apply-cuts", action="store_true",
                   help="require the VBF H->inv event selection")
    p.add_argument("--min-abs-truth", type=float, default=None, metavar="PS",
                   help="only events with |truth t0| above this. Ranking on "
                        "the residual alone favours events near zero, which "
                        "a regressor finds easy; this asks what it does when "
                        "the answer is far from the mean")
    args = p.parse_args()

    global IDEAL_EFF
    IDEAL_EFF = args.ideal_eff

    with open(args.config) as fh:
        spec = AssemblySpec.from_config(yaml.safe_load(fh)["data"])
    stores = {d.name: EventStore(d.path, sample=d.name) for d in spec.datasets}
    index = {n: {int(e): i for i, e in enumerate(s.event_column("event_number"))}
             for n, s in stores.items()}
    models = [(_label(d), _predictions(d)) for d in args.model_dir]

    how = next((h for h in RANKINGS + MIGRATIONS if getattr(args, h)), None)
    if args.min_abs_truth is not None and not how:
        p.error("--min-abs-truth selects what is ranked; give a ranking or a migration")
    # Selected before ranking; display() applies --apply-cuts to --event itself.
    eligible = _eligible(stores, args.apply_cuts, args.min_abs_truth) if how else None
    if how in RANKINGS:
        if not args.model_dir:
            p.error(f"--{how} needs --model-dir")
        wanted = pick_events(args.model_dir[0], how, getattr(args, how), eligible)
    elif how in MIGRATIONS:
        if len(args.model_dir) < 2:
            p.error(f"--{how} needs two --model-dir: the combined model, then the base")
        wanted = pick_migration(args.model_dir[0], args.model_dir[1], how,
                                getattr(args, how), eligible)
    elif args.event:
        wanted = {n: [e for e in args.event if e in index[n]] for n in stores}
    else:
        p.error("give --event, or --model-dir with a ranking or a migration")

    for name, events in wanted.items():
        for event in events:
            if event in index[name]:
                display(stores[name], index[name][event], args.out,
                        [(label, *pr[(name, event)]) for label, pr in models
                         if (name, event) in pr], args.dpi, args.apply_cuts)


if __name__ == "__main__":
    main()
