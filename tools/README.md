# Tools — DFT data generation & cloud infrastructure

Scripts for generating DFT training data on cloud GPU instances (Vast.ai)
and monitoring long-running calculations.

## DFT data generation

```bash
# On a GPU instance with GPAW installed:
python3 -u generate_sulfide_dft_data_v2.py \
  --output /workspace/results/sulfide_train.xyz \
  --resume --mineral pentlandite \
  --worker-id 0 --num-workers 4
```

Generates DFT (GPAW, PBE, PW mode) single-point calculations for
iron sulfide minerals (pentlandite, mackinawite, pyrite). Outputs
extended XYZ with energies and forces for MACE fine-tuning.

**Supported minerals:** pentlandite, mackinawite, pyrite, greigite, marcasite, millerite

**Configurations:** bulk rattled, vacancy, surface slabs, surface rattled

**Parallelism:** `--worker-id` / `--num-workers` for splitting configs across instances.

## Cloud instance management

| Script | Where it runs | Purpose |
|--------|--------------|---------|
| `vast_launch.sh` | On instance | Launch wrapper: lock file, signal traps, separate stdout/stderr, crash reports |
| `vast_monitor.sh` | On instance (daemon) | System monitoring every 60s: RAM, CPU, disk, OOM detection, python3 heartbeat |
| `vast_check.sh` | Local (via SSH relay) | Quick status: process, memory, heartbeat, exit code, crash info, XYZ progress |
| `vast_diagnose.sh` | Local (via SSH relay) | Full post-mortem: 15 diagnostic sections including dmesg OOM, all logs |
| `onstart_template.sh` | On instance (onstart) | Clones this repo, deploys all scripts, starts monitor; optionally auto-starts GPAW |

### Creating a new instance

**Option A: one-liner bootstrap (recommended)**

Scripts auto-deploy from GitHub on first boot:

```bash
vastai create instance <offer_id> \
  --image exopoiesis/infra-gpaw-gpu \
  --disk 20 --ssh \
  --label "v2-pent-w0" \
  --onstart-cmd 'export TZ=Europe/Kyiv; git clone --depth 1 https://github.com/exopoiesis/digital-twin.git /workspace/digital-twin && cp /workspace/digital-twin/tools/*.sh /workspace/digital-twin/tools/*.py /workspace/ 2>/dev/null; chmod +x /workspace/*.sh; mkdir -p /workspace/results; nohup bash /workspace/vast_monitor.sh >/dev/null 2>&1 & disown'
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
  --onstart-cmd "$(cat tools/onstart_template.sh)"
```

The onstart script will: clone this repo → copy scripts → start monitor → launch GPAW.

**Option C: manual deploy (fallback)**

If onstart didn't run or you need to redeploy:

```bash
ssh -p <port> root@<host> bash -c '
  git clone --depth 1 https://github.com/exopoiesis/digital-twin.git /workspace/digital-twin
  cp /workspace/digital-twin/tools/*.sh /workspace/digital-twin/tools/*.py /workspace/
  chmod +x /workspace/*.sh
  mkdir -p /workspace/results
  nohup bash /workspace/vast_monitor.sh >/dev/null 2>&1 & disown
'
```

**Instance requirements:** RAM > 16 GB (mandatory), GHz >= 5.0 (preferred for CPU-bound GPAW).

### Checking instance status

```bash
# Quick check (from local machine, via loki SSH relay)
bash tools/vast_check.sh ssh4 19224 W5

# Full post-mortem diagnostics
bash tools/vast_diagnose.sh ssh4 19224 W5
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

- **DFT script:** GPAW 25.x, ASE, numpy (provided by `exopoiesis/infra-gpaw-gpu` Docker image)
- **Monitoring scripts:** bash, standard Linux utils (no dependencies)
- **Local check/diagnose:** Docker with SSH relay container (or direct SSH access)
- **Git** available in Docker image for auto-deploy (fallback: manual scp)
