#!/usr/bin/env python3
"""
Periodic checkpoint/resume for GPAW DFT scripts on Vast.ai.

Strategy: save checkpoints every N SCF iterations. On crash/kill,
vast_watchdog.sh restarts the script which resumes from last checkpoint.
No SIGTERM handling -- Vast.ai gives only 10s before SIGKILL, not enough
to write .gpw with wavefunctions. Periodic saves every ~5 min are the
real protection.

Usage in single-point loop scripts (generate_*.py):
    # No checkpoint needed -- script uses --resume flag to skip
    # completed configs. Watchdog restarts on kill.

Usage in BFGS/relaxation scripts (q075_solvation_dft.py):
    from gpaw_checkpoint import CheckpointManager

    calc = make_solvation_calc(...)
    atoms.calc = calc

    mgr = CheckpointManager(atoms, checkpoint_dir / 'checkpoint.gpw')
    mgr.attach_to_calc(calc, interval=5)

    opt = BFGS(atoms, ...)
    opt.run(fmax=0.05, steps=100)

MPI safety: periodic checkpoint callback does I/O only on rank 0
(GPAW handles this internally in calc.write()).
"""

import os
import signal
import time

# --- Lightweight SIGTERM flag for single-point loop scripts ---
# Used by generate_*.py to stop between configs (not mid-SCF).
# No checkpoint save on SIGTERM -- watchdog handles restart.

_shutdown_requested = False
_handler_registered = False


def _sigterm_handler(signum, frame):
    """Set flag only, no I/O (MPI-safe)."""
    global _shutdown_requested
    _shutdown_requested = True


def register_sigterm_handler():
    """Register SIGTERM handler. Idempotent, MPI-safe."""
    global _handler_registered
    if _handler_registered:
        return
    signal.signal(signal.SIGTERM, _sigterm_handler)
    _handler_registered = True


def is_shutdown_requested():
    """Check if SIGTERM was received. Non-blocking."""
    return _shutdown_requested


class CheckpointManager:
    """Periodic checkpoint manager for GPAW BFGS/relaxation.

    Saves .gpw every `interval` SCF iterations (with time gate to avoid
    excessive I/O). On restart, load from checkpoint to resume.

    Args:
        atoms: ASE Atoms object with GPAW calculator
        checkpoint_path: Path for .gpw checkpoint file
        interval: SCF iterations between checkpoint checks (default 5)
        min_save_interval: minimum seconds between saves (default 120)
    """

    def __init__(self, atoms, checkpoint_path, interval=5, min_save_interval=120):
        self.atoms = atoms
        self.checkpoint_path = str(checkpoint_path)
        self.interval = interval
        self.min_save_interval = min_save_interval
        self._scf_count = 0
        self._last_checkpoint_time = time.time()

    def _scf_callback(self):
        """Called by calc.attach() every `interval` SCF iterations.

        Saves checkpoint if enough time has passed since last save.
        """
        self._scf_count += self.interval

        elapsed = time.time() - self._last_checkpoint_time
        if elapsed >= self.min_save_interval:
            print(f"[checkpoint] Periodic save at SCF ~{self._scf_count} "
                  f"({elapsed:.0f}s since last)", flush=True)
            self._save()
            self._last_checkpoint_time = time.time()

    def _save(self):
        """Save GPAW checkpoint with wavefunctions."""
        calc = self.atoms.calc
        if calc is not None:
            try:
                calc.write(self.checkpoint_path, mode='all')
            except Exception as e:
                print(f"[checkpoint] WARNING: save failed: {e}", flush=True)

    def attach_to_calc(self, calc, interval=None):
        """Attach checkpoint callback to GPAW calculator.

        Must be called BEFORE any get_potential_energy() / opt.run().

        Args:
            calc: GPAW or SolvationGPAW calculator
            interval: override SCF interval (default: self.interval)
        """
        if interval is not None:
            self.interval = interval
        calc.attach(self._scf_callback, self.interval)
        print(f"[checkpoint] Attached (interval={self.interval} SCF iters, "
              f"save every >={self.min_save_interval}s, "
              f"path={self.checkpoint_path})", flush=True)

    @staticmethod
    def checkpoint_exists(checkpoint_path):
        """Check if a checkpoint file exists and is non-empty."""
        path = str(checkpoint_path)
        return os.path.exists(path) and os.path.getsize(path) > 0

    @staticmethod
    def load_checkpoint(checkpoint_path):
        """Load GPAW calculator from checkpoint.

        Returns:
            GPAW calculator with wavefunctions, or None if no checkpoint.
        """
        path = str(checkpoint_path)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return None

        try:
            from gpaw import GPAW
            calc = GPAW(path)
            print(f"[checkpoint] Loaded checkpoint from {path}", flush=True)
            return calc
        except Exception as e:
            print(f"[checkpoint] WARNING: could not load {path}: {e}", flush=True)
            return None
