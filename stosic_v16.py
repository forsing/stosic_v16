from __future__ import annotations

"""
https://github.com/gajaka/luces-pvs-theories
"""

"""
stosic_v16.py — 7-node krug (K=7 / prilagodjenje 7/39) — Stability of maps under perturbation (7/39)

Izvor (Stosić / LUCES):
  luces-pvs-theories-main/stability_of_maps.pvs
  — Villani 5.20: μₙ→μ, νₙ→ν ⇒ πₙ→π
  — ax_stability: W₂(μ1,μ2)<δ ∧ W₂(ν1,ν2)<δ ⇒ map_distance(T11,T22)<δ
  — thm_persistence: slični dani → slične mape

Mapiranje na 7/39:
  c = ||x_i-x_j||² (ko-pojavljivanje); W₂ iz matching cost
  δ = medijan W₂ uzastopnih kola (stabilni radijus)
  referentni par: (draws[-2], draws[-1])
  za svaki (t,t+1) sa W₂(t,−2)<δ i W₂(t+1,−1)<δ: +matching u Π
  skor[j] = Σ_{i∈last} Π(i,j)
  next = top 7; bez randoma; stop ako uzastopni/AP
"""

from typing import List

import numpy as np

from stosic_v1 import EPS, MAX_NUM, load_draws
from stosic_v2 import top7_from_freq
from stosic_v8 import cooccurrence_features, cost_matrix
from stosic_v9 import optimal_matching_support
from stosic_v10 import is_degenerate
from stosic_v12 import w2_sq_draws


def predict_next(draws: np.ndarray) -> List[int]:
    if len(draws) < 3:
        raise SystemExit("CSV prekratak")
    C = cost_matrix(cooccurrence_features(draws))
    w2_consec = [
        np.sqrt(max(w2_sq_draws(draws[t], draws[t + 1], C), 0.0))
        for t in range(len(draws) - 1)
    ]
    delta = float(np.median(w2_consec))
    mu_ref, nu_ref = draws[-2], draws[-1]
    Pi = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float64)
    for t in range(len(draws) - 1):
        w_mu = np.sqrt(max(w2_sq_draws(draws[t], mu_ref, C), 0.0))
        w_nu = np.sqrt(max(w2_sq_draws(draws[t + 1], nu_ref, C), 0.0))
        if w_mu < delta and w_nu < delta:
            src = [int(n) - 1 for n in draws[t]]
            tgt = [int(n) - 1 for n in draws[t + 1]]
            for i, j in optimal_matching_support(src, tgt, C):
                Pi[i, j] += 1.0
    skor = np.zeros(MAX_NUM, dtype=np.float64)
    for n in draws[-1]:
        skor += Pi[int(n) - 1, :]
    if float(skor.sum()) <= 0:
        for d in draws:
            for n in d:
                skor[int(n) - 1] += 1.0
    combo = top7_from_freq(skor + EPS)
    if is_degenerate(combo):
        nu = np.zeros(MAX_NUM, dtype=np.float64)
        for d in draws:
            for n in d:
                nu[int(n) - 1] += 1.0
        combo = top7_from_freq(nu)
    return combo


def main():
    draws = load_draws()
    next_combo = predict_next(draws)
    if is_degenerate(next_combo):
        raise SystemExit("degenerisan next (uzastopni/AP) — zaustavljen pre ispisa")
    print(next_combo)


if __name__ == "__main__":
    main()



"""
[7, x, 14, y, 23, z, 35]
"""



"""
v16: stability_of_maps — matchingi sa parova W₂-bliзу referenci (δ=medijan).
"""



"""
21 teorija

fisher_voronoi → v1, v2
dual_observability → v3
v4 se pozivao na W₂/stabilnost — slabo / nije strogo
entropy_along_geodesic → v5
velocity_asymmetry (+ delom lie_generator_structure) → v6
brenier_uniqueness (+ delom rank_orientation) → v7

kantorovich_duality
cyclical_monotonicity
displacement_interpolation
displacement_concavity
wasserstein_metric (strogo)
transport_structure
transport_structure_v2
transport_stability
stability_of_maps
monge_kantorovich_equivalence
lie_generator_structure (pun T10)
fisher_boundary
hybrid_observability
tangent_bundle
global_optimality
"""



"""
Kratko, o repou:

21 PVS teorija — sve su prošle kroz v1–v22 (neke ranije labavo: naročito v3/v4; rank_orientation je ušao uz Brenier u v7).
Repo je o spektralnom OT / LUCES (ESP32), ne o lotou — 7/39 je naša mapa, ne Stosićev domen.
Najčistije jezgro oko Fisher–Voronoi, Brenier/CM, W₂, T10 (lie_generator_structure). global_optimality je samo aksiomi + lema (bez teorema).
Empirija u PVS-u (bootovi, κ, Monge fraction) ne prenosi se automatski na CSV — samo struktura ideja.
"""
