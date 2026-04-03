# Review Findings

Scope:
- Markdown files reviewed one by one, recording only serious issues and concrete fixes.
- Python files to review after markdown if time permits.

Status:
- Completed.

## README.md

1. Serious: the "Reproducing from scratch" command for Phase B.1 generates the wrong dataset size.
   Evidence: [README.md](/D:/home/ignat/project-third-matter/git/digital-twin/README.md#L91) names the output `oracle_membrane_50k.npz`, but [oracle_phase_b_datagen.py](/D:/home/ignat/project-third-matter/git/digital-twin/oracle/oracle_phase_b_datagen.py#L549) defaults to `--n_samples 5000`.
   Impact: anyone following the documented command will silently create a 5k dataset while believing they reproduced the 50k training corpus, which invalidates downstream training/runtime expectations and any benchmark comparison.
   Fix: change the command to include `--n_samples 50000`, or rename the expected artifact and all surrounding text to 5k if that is the intended baseline.

## hypothesis-tester/README.md

1. Serious: the usage section documents a CLI that does not match the real argument parser.
   Evidence: [hypothesis-tester/README.md](/D:/home/ignat/project-third-matter/git/digital-twin/hypothesis-tester/README.md#L18) uses flags such as `--value`, `--min`, `--max`, `--min2`, and omits `--vary`; the actual parser in [oracle_hypothesis_tester.py](/D:/home/ignat/project-third-matter/git/digital-twin/hypothesis-tester/oracle_hypothesis_tester.py#L73) accepts `--L_mack`-style overrides for `single`, `--range` for `sweep`/`threshold`, `--param1`/`--range1` plus `--param2`/`--range2` for `grid2d`, and requires `--vary` for `montecarlo`.
   Impact: a reader cannot successfully use the tool by following the README; nearly every documented example will fail at parse time.
   Fix: rewrite all examples to match the implemented CLI. Example corrections:
   `single`: `python oracle_hypothesis_tester.py --mode single --L_mack 50`
   `sweep`: `python oracle_hypothesis_tester.py --mode sweep --param L_mack --range 5,200 --n 30`
   `threshold`: `python oracle_hypothesis_tester.py --mode threshold --param L_mack --range 1,200 --target 1.0`
   `grid2d`: `python oracle_hypothesis_tester.py --mode grid2d --param1 L_mack --range1 5,200 --param2 delta_pH --range2 2,8 --n 15`
   `montecarlo`: `python oracle_hypothesis_tester.py --mode montecarlo --vary L_mack --spread 0.3 --n 100`

## oracle/README.md

No serious issues found on this pass. The documented artifacts in `oracle/data/` exist, and the only executable example shown is consistent if the commands are run from the `oracle/` directory.

## tools/README.md

1. Serious: the instance bootstrap commands still assume a flat `tools/` layout that no longer exists.
   Evidence: [tools/README.md](/D:/home/ignat/project-third-matter/git/digital-twin/tools/README.md#L75) and [tools/README.md](/D:/home/ignat/project-third-matter/git/digital-twin/tools/README.md#L108) copy `tools/*.sh` and `tools/*.py`, but the repository now stores those files under `tools/infra/`, `tools/gpaw/`, `tools/abacus/`, `tools/qe/`, and `tools/validation/`.
   Impact: a fresh Vast.ai instance created from the README will not receive `vast_monitor.sh`, `vast_launch.sh`, or the GPAW/QE/ABACUS scripts, so monitoring and job launch fail immediately.
   Fix: update both bootstrap snippets to copy from the real subdirectories, for example `tools/infra/*.sh` plus `tools/gpaw/*.py`, `tools/abacus/*.py`, and `tools/qe/*.py`.

## ABACUS_COMPUTATION_GUIDE.md

1. Serious: the environment setup block is self-contradictory and wrong if copied literally.
   Evidence: [ABACUS_COMPUTATION_GUIDE.md](/D:/home/ignat/project-third-matter/git/digital-twin/ABACUS_COMPUTATION_GUIDE.md#L431) sets `OMP_NUM_THREADS=$(nproc)` and then immediately overrides it with `OMP_NUM_THREADS=1`.
   Impact: anyone pasting the block gets `OMP_NUM_THREADS=1` in all cases, which defeats the guide's own recommendation for LCAO/CPU runs and can degrade those calculations by an order of magnitude.
   Fix: split this into two mutually exclusive snippets or use an explicit conditional: one block for LCAO/CPU, one block for PW/GPU.

2. Serious: the mackinawite barrier table swaps the pathway labels between GPAW and QE.
   Evidence: [ABACUS_COMPUTATION_GUIDE.md](/D:/home/ignat/project-third-matter/git/digital-twin/ABACUS_COMPUTATION_GUIDE.md#L370) says QE `2.479 eV` is intra-layer and GPAW `0.738 eV` is inter-layer, but [mackinawite_h_transport.json](/D:/home/ignat/project-third-matter/git/digital-twin/results/mackinawite_h_transport.json#L11) records `0.738 eV` as the GPAW intra-layer hop and [mackinawite_h_transport.json](/D:/home/ignat/project-third-matter/git/digital-twin/results/mackinawite_h_transport.json#L31) records `2.479 eV` as the QE cross-layer hop.
   Impact: this inverts the physical interpretation of the result and can lead readers to the exact opposite conclusion about which mackinawite transport pathway is favorable.
   Fix: relabel the two rows so GPAW `0.738 eV` is intra-layer and QE `2.479 eV` is cross-layer through the van der Waals gap.

## GPAW_COMPUTATION_GUIDE.md

1. Serious: the Vast.ai bootstrap snippet is outdated after the `tools/` reorganization.
   Evidence: [GPAW_COMPUTATION_GUIDE.md](/D:/home/ignat/project-third-matter/git/digital-twin/GPAW_COMPUTATION_GUIDE.md#L397) copies `tools/*.sh` and `tools/*.py`, but the actual scripts now live under `tools/infra/`, `tools/gpaw/`, `tools/abacus/`, and `tools/qe/`.
   Impact: a reader following the guide on a fresh instance will fail to deploy `vast_monitor.sh` and the GPAW workload scripts, so the advertised monitoring and launch workflow will not work.
   Fix: update the snippet to copy from the real subdirectories, or stop flattening them into `/workspace` and launch them directly from the cloned repository paths.

## JDFTX_COMPUTATION_GUIDE.md

1. Serious: the guide recommends making the active run directory immutable before launching JDFTx.
   Evidence: [JDFTX_COMPUTATION_GUIDE.md](/D:/home/ignat/project-third-matter/git/digital-twin/JDFTX_COMPUTATION_GUIDE.md#L416) applies `chattr +i /workspace/candle_production/`, and the very next run pattern writes output to that same directory at [JDFTX_COMPUTATION_GUIDE.md](/D:/home/ignat/project-third-matter/git/digital-twin/JDFTX_COMPUTATION_GUIDE.md#L437).
   Impact: an immutable directory cannot safely serve as an active work directory for a code that needs to create or rotate output/checkpoint files; readers can end up with a launch that fails or a run that cannot write new artifacts.
   Fix: protect the parent directory or specific finished artifacts instead. Do not set `+i` on the live run directory before execution.

2. Serious: the top-level checkpoint template uses syntax that the same guide later says is a stock JDFTx parse error.
   Evidence: [JDFTX_COMPUTATION_GUIDE.md](/D:/home/ignat/project-third-matter/git/digital-twin/JDFTX_COMPUTATION_GUIDE.md#L103) shows `dump-interval Electronic 5 State`, but [JDFTX_COMPUTATION_GUIDE.md](/D:/home/ignat/project-third-matter/git/digital-twin/JDFTX_COMPUTATION_GUIDE.md#L219) says this inline form is invalid in stock JDFTx 1.7.0 and must be written as `dump Electronic State` plus `dump-interval Electronic 5`.
   Impact: a reader copying the "minimal template" into a stock installation gets an input-file parse failure before the calculation even starts.
   Fix: replace the template with the portable two-line form, or label the current form as custom-build-only right where it appears.

## QE_COMPUTATION_GUIDE.md

1. Serious: the guide reverses the mackinawite pathway assignments in the benchmark and interpretation sections.
   Evidence: [QE_COMPUTATION_GUIDE.md](/D:/home/ignat/project-third-matter/git/digital-twin/QE_COMPUTATION_GUIDE.md#L395) and [QE_COMPUTATION_GUIDE.md](/D:/home/ignat/project-third-matter/git/digital-twin/QE_COMPUTATION_GUIDE.md#L409) treat QE `2.479 eV` as intra-layer and GPAW `0.738 eV` as inter-layer, but [mackinawite_h_transport.json](/D:/home/ignat/project-third-matter/git/digital-twin/results/mackinawite_h_transport.json#L11) and [mackinawite_h_transport.json](/D:/home/ignat/project-third-matter/git/digital-twin/results/mackinawite_h_transport.json#L31) show the opposite: GPAW `0.738 eV` is intra-layer, QE `2.479 eV` is cross-layer through the vdW gap.
   Impact: readers are told the wrong physical mechanism, so the cross-code comparison becomes scientifically misleading rather than informative.
   Fix: relabel the QE result as cross-layer and the GPAW result as intra-layer everywhere this discrepancy is discussed.

## Python files (targeted sweep)

1. Serious: the GPAW batch runners still hardcode the pre-reorganization `/workspace/scripts/` layout.
   Evidence: [run_all_neb.py](/D:/home/ignat/project-third-matter/git/digital-twin/tools/gpaw/run_all_neb.py#L23) and [run_overnight.py](/D:/home/ignat/project-third-matter/git/digital-twin/tools/gpaw/run_overnight.py#L28) launch children from `/workspace/scripts/{script}`, and both file headers still tell users to run `python scripts/...`; there is no `scripts/` directory in this repository anymore.
   Impact: these orchestrator scripts fail immediately on a clean deployment unless someone manually recreates the old flat layout.
   Fix: resolve child scripts relative to `Path(__file__).parent` or the cloned repo root, e.g. `Path(__file__).with_name(script)`, and update the usage text accordingly.

2. Serious: `oracle_phase_b3_pinn.py` uses repo-root-relative defaults instead of file-relative defaults.
   Evidence: [oracle_phase_b3_pinn.py](/D:/home/ignat/project-third-matter/git/digital-twin/oracle/oracle_phase_b3_pinn.py#L1088) defaults to `oracle/data/...` paths even though the script itself lives inside `oracle/`.
   Impact: launching the script from its own directory with defaults, e.g. `cd oracle && python oracle_phase_b3_pinn.py`, resolves those paths to `oracle/oracle/data/...` and fails to find the bundled artifacts.
   Fix: build defaults from `Path(__file__).resolve().parent / 'data'`, as already done more robustly in [oracle_phase_d_fno_ode.py](/D:/home/ignat/project-third-matter/git/digital-twin/oracle/oracle_phase_d_fno_ode.py#L1513).
