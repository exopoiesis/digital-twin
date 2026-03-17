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
| `onstart_template.sh` | On instance (onstart) | Auto-deploys monitor daemon at instance boot; optionally auto-starts GPAW |

### Typical workflow

```bash
# 1. Create instance
vastai create instance <offer_id> --image exopoiesis/infra-gpaw-gpu --disk 20 --ssh

# 2. Upload scripts
scp -P <port> tools/vast_monitor.sh tools/vast_launch.sh \
  tools/generate_sulfide_dft_data_v2.py root@<host>:/workspace/

# 3. Launch with monitoring
ssh -p <port> root@<host> "nohup bash /workspace/vast_launch.sh \
  generate_sulfide_dft_data_v2.py --output /workspace/results/sulfide_train.xyz \
  --resume --mineral pentlandite --worker-id 0 --num-workers 4 &"

# 4. Check progress (from local machine)
bash tools/vast_check.sh ssh4 19224 W5

# 5. If something went wrong — full diagnostics
bash tools/vast_diagnose.sh ssh4 19224 W5
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
