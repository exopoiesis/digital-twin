# Tools — DFT calculations & cloud infrastructure

Scripts for DFT NEB barrier calculations, single-point datagen, and cloud instance management.
Supports **GPAW**, **ABACUS**, and **Quantum ESPRESSO** on Vast.ai GPU/CPU instances.

## Directory structure

```
tools/
├── gpaw/           — GPAW NEB, datagen, MD, config generators (40 scripts)
├── abacus/         — ABACUS NEB + vacancy formation (4 scripts)
├── qe/             — Quantum ESPRESSO NEB (6 scripts)
├── validation/     — Dataset validation, cross-code checks (5 scripts)
└── infra/          — Vast.ai launch, monitoring, diagnostics (10 scripts)
```

## DFT NEB scripts (H⁺ diffusion barriers)

| Script | Folder | Mineral | Cell | E_a (eV) |
|--------|--------|---------|------|----------|
| `neb_pentlandite_gpu.py` | gpaw/ | Pentlandite | prim (17 at) | 1.115 |
| `neb_pentlandite_abacus.py` | abacus/ | Pentlandite | prim (17 at) | 0.900 |
| `neb_pentlandite_abacus_conv.py` | abacus/ | Pentlandite | conv (68 at) | running |
| `neb_pentlandite_qe.py` | qe/ | Pentlandite | prim (17 at) | cross-verify |
| `neb_mackinawite_gpu.py` | gpaw/ | Mackinawite | 2x2x1 (15 at) | 0.738 |
| `neb_mackinawite_qe.py` | qe/ | Mackinawite | 2x2x1 (15 at) | 2.479 (cross-layer) |
| `neb_mackinawite_grotthuss_qe.py` | qe/ | Mackinawite | 3x3x1 + water | Grotthuss |
| `neb_pyrite_gpu.py` | gpaw/ | Pyrite | 1x1x1 (12 at) | 0.181 |
| `neb_pyrite_abacus.py` | abacus/ | Pyrite | 1x1x1 (12 at) | 0.187 |
| `neb_pyrite_qe.py` | qe/ | Pyrite | 1x1x1 (12 at) | 0.190 |
| `neb_pyrite_qe_2x2x2_gpu.py` | qe/ | Pyrite | 2x2x2 (95 at) | size conv |
| `neb_troilite_qe.py` | qe/ | Troilite | 1x1x1 (24 at) | ~0.375 (running) |
| `vacancy_formation_abacus.py` | abacus/ | Multi | conv cells | E_vac |

## DFT data generation (GPAW)

```bash
# On a GPU instance with GPAW installed:
python3 -u gpaw/generate_sulfide_dft_data_v2.py \
  --output /workspace/results/sulfide_train.xyz \
  --resume --mineral pentlandite \
  --worker-id 0 --num-workers 4
```

Generates DFT (GPAW, PBE, PW mode) single-point calculations for
iron sulfide minerals. Outputs extended XYZ with energies and forces for MACE fine-tuning.

**Supported minerals:** pentlandite, mackinawite, pyrite, greigite, marcasite, millerite, troilite, cubanite, chalcopyrite, violarite, smythite, bornite

**Configurations:** bulk rattled, vacancy, surface slabs, surface rattled

**Parallelism:** `--worker-id` / `--num-workers` for splitting configs across instances.

## Cloud instance management

| Script | Where it runs | Purpose |
|--------|--------------|---------|
| `infra/vast_launch.sh` | On instance | Launch wrapper: lock file, signal traps, separate stdout/stderr, crash reports |
| `infra/vast_monitor.sh` | On instance (daemon) | System monitoring every 60s: RAM, CPU, disk, OOM detection, python3 heartbeat |
| `infra/vast_check.sh` | Local (via SSH relay) | Quick status: process, memory, heartbeat, exit code, crash info, XYZ progress |
| `infra/vast_diagnose.sh` | Local (via SSH relay) | Full post-mortem: 15 diagnostic sections including dmesg OOM, all logs |
| `infra/onstart_template.sh` | On instance (onstart) | Clones this repo, deploys all scripts, starts monitor; optionally auto-starts GPAW |

### Creating a new instance

**Option A: one-liner bootstrap (recommended)**

Scripts auto-deploy from GitHub on first boot:

```bash
vastai create instance <offer_id> \
  --image exopoiesis/infra-gpaw-gpu \
  --disk 20 --ssh \
  --label "v2-pent-w0" \
  --onstart-cmd 'export TZ=Europe/Kyiv; git clone --depth 1 https://github.com/exopoiesis/digital-twin.git /workspace/digital-twin && cp /workspace/digital-twin/tools/infra/*.sh /workspace/digital-twin/tools/gpaw/*.py /workspace/digital-twin/tools/abacus/*.py /workspace/digital-twin/tools/qe/*.py /workspace/ 2>/dev/null; chmod +x /workspace/*.sh; mkdir -p /workspace/results; nohup bash /workspace/vast_monitor.sh >/dev/null 2>&1 & disown'
```

Then SSH in and launch GPAW manually:

```bash
ssh -p <port> root@<host> "nohup bash /workspace/vast_launch.sh \
  generate_sulfide_dft_data_v2.py \
  --output /workspace/results/sulfide_train.xyz --resume \
  --mineral pentlandite --worker-id 0 --num-workers 4 \
  > /dev/null 2>&1 & disown"
```

**Option B: full auto-start (bootstrap + GPAW)**

```bash
vastai create instance <offer_id> \
  --image exopoiesis/infra-gpaw-gpu \
  --disk 20 --ssh \
  --label "v2-pent-w0" \
  --env "AUTOSTART=yes WORKER_ID=0 NUM_WORKERS=4 MINERAL=pentlandite" \
  --onstart-cmd "$(cat tools/infra/onstart_template.sh)"
```

The onstart script will: clone this repo → copy scripts → start monitor → launch GPAW.

**Option C: manual deploy (fallback)**

If onstart didn't run or you need to redeploy:

```bash
ssh -p <port> root@<host> bash -c '
  git clone --depth 1 https://github.com/exopoiesis/digital-twin.git /workspace/digital-twin
  cp /workspace/digital-twin/tools/infra/*.sh /workspace/digital-twin/tools/gpaw/*.py /workspace/digital-twin/tools/abacus/*.py /workspace/digital-twin/tools/qe/*.py /workspace/
  chmod +x /workspace/*.sh
  mkdir -p /workspace/results
  nohup bash /workspace/vast_monitor.sh >/dev/null 2>&1 & disown
'
```

**Instance requirements:** RAM > 16 GB (mandatory), GHz >= 5.0 (preferred for CPU-bound GPAW).

### Checking instance status

```bash
# Quick check (from local machine, via loki SSH relay)
bash tools/infra/vast_check.sh ssh4 19224 W5

# Full post-mortem diagnostics
bash tools/infra/vast_diagnose.sh ssh4 19224 W5
```

### Resuming a dead/replaced instance

```bash
# 1. If old instance has results — download them first!
scp -P <old_port> root@<old_host>:/workspace/results/*.xyz ./results/

# 2. Create new instance (same steps as above)

# 3. Upload old results for --resume to continue
scp -P <new_port> ./results/<old_xyz> root@<new_host>:/workspace/results/

# 4. Deploy scripts + launch (steps 2-4 from "Creating a new instance")
```

### What vast_launch.sh provides

- **Lock file** (`/workspace/run.lock`) — prevents duplicate GPAW processes
- **Signal traps** (SIGTERM, SIGINT) — graceful shutdown with exit code logging
- **Separate logs**: stdout → `run.log`, stderr → `run_stderr.log`
- **Crash report** (`/workspace/results/crash_info`) — last log lines + memory + dmesg OOM
- **Exit code** (`/workspace/results/exit_code`) — persists after process death
- **Auto-starts monitor daemon** if `vast_monitor.sh` is present

### What vast_monitor.sh tracks (every 60s)

- RAM usage and availability
- Swap usage
- Disk space on /workspace
- CPU load average
- GPU utilization and VRAM (if nvidia-smi available)
- python3 process alive/dead
- OOM killer events in dmesg
- XYZ config count (DFT progress)
- Heartbeat file for external monitoring

## Requirements

- **GPAW scripts:** GPAW 25.x, ASE, numpy (`exopoiesis/infra-gpaw-gpu` Docker image)
- **ABACUS scripts:** ABACUS v3.9, ASE (`exopoiesis/infra-abacus-gpu` Docker image)
- **QE scripts:** Quantum ESPRESSO 7.5, ASE (`exopoiesis/infra-qe-gpu:7.5` or `infra-qe:latest`)
- **Monitoring scripts:** bash, standard Linux utils (no dependencies)
- **Local check/diagnose:** Docker with SSH relay container (or direct SSH access)
- **Git** available in Docker image for auto-deploy (fallback: manual scp)
