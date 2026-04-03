#!/usr/bin/env python3
"""
NEB расчёт H-диффузии в троилите (FeS) через Quantum ESPRESSO + ASE.

Механизм: вакансионный прыжок H между двумя S-вакансиями.
Ячейка: конвенциональная, 24 атома (12 Fe + 12 S), P-62c (#190).
После вакансий: 23 атома (12 Fe + 10 S + 1 H).

Метод:
  - QE 7.4.1, PBE, USPP PSlibrary
  - DFT+U: U_eff=2.0 eV на Fe (Dudarev), новый синтаксис QE 7.3.1+
  - AFM: чередующиеся Fe слои ↑↓ по оси c
  - CI-NEB с IDPP интерполяцией, FIRE optimizer
  - Resume после каждого этапа

Запуск:
  export OMP_NUM_THREADS=$(nproc)
  python3 -u neb_troilite_qe.py

Ссылки:
  Xisto et al. 2025, J. Phys. Chem. C (k-mesh 2x4x2, U_eff=2.0 eV)
  Skala 2006, CIF 0004158 (структура, P-62c)
"""

import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from ase import Atom
from ase.io import write as ase_write, read as ase_read
from ase.geometry import get_distances
from ase.mep import NEB
from ase.optimize import FIRE
from ase.spacegroup import crystal
from ase.constraints import FixAtoms

warnings.filterwarnings("ignore")

# ===========================================================================
#  Конфигурация
# ===========================================================================

RESULTS = Path("/workspace/results")
WORK_DIR = Path("/workspace/neb_troilite_qe")
SCRATCH_DIR = Path("/workspace/qe_scratch")

N_IMAGES = 5            # промежуточные образы (без эндпоинтов)
FMAX_RELAX = 0.05       # eV/A, для релаксации эндпоинтов
FMAX_NEB = 0.05         # eV/A, для CI-NEB
MAX_STEPS_RELAX = 50    # шагов FIRE для эндпоинтов
MAX_STEPS_NEB = 300     # шагов FIRE для NEB

ECUTWFC = 60            # Ry (USPP PBE, Xisto 2025)
ECUTRHO = 480           # Ry (8 * ecutwfc для USPP)
KPTS = (2, 4, 2)        # Monkhorst-Pack (Xisto 2025)

# Pseudo-потенциалы PSlibrary USPP PBE
PSEUDO_DIR = os.environ.get("QE_PSEUDO_DIR", "/opt/pseudopotentials")
PSEUDOPOTENTIALS = {
    "Fe": "Fe.pbe-spn-rrkjus_psl.1.0.0.UPF",
    "S":  "S.pbe-n-rrkjus_psl.1.0.0.UPF",
    "H":  "H.pbe-rrkjus_psl.1.0.0.UPF",
}

# pw.x command: mpirun внутри калькулятора
QE_COMMAND = "mpirun --allow-run-as-root --bind-to none -np 1 pw.x"

# Файлы resume/checkpoint
RESUME_FILE = WORK_DIR / "resume.json"
CHECKPOINT_START_XYZ = WORK_DIR / "checkpoint_endA.xyz"
CHECKPOINT_END_XYZ = WORK_DIR / "checkpoint_endB.xyz"


# ===========================================================================
#  NumpyEncoder — ОБЯЗАТЕЛЕН (numpy.bool_, float64 etc не сериализуются!)
# ===========================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder, который корректно обрабатывает numpy-типы."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ===========================================================================
#  Resume / checkpoint
# ===========================================================================

def save_resume(stage: str, data: dict):
    """Сохранить состояние выполнения после завершения этапа."""
    state = {}
    if RESUME_FILE.exists():
        try:
            state = json.loads(RESUME_FILE.read_text())
        except Exception:
            pass
    state[stage] = data
    state["_updated"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    RESUME_FILE.write_text(json.dumps(state, indent=2, cls=NumpyEncoder))
    print(f"  [resume] сохранено: этап '{stage}'", flush=True)
    sys.stdout.flush()


def load_resume() -> dict:
    """Загрузить состояние resume, если файл существует."""
    if RESUME_FILE.exists():
        try:
            state = json.loads(RESUME_FILE.read_text())
            print(f"  [resume] загружен файл: {RESUME_FILE}", flush=True)
            return state
        except Exception as e:
            print(f"  [resume] WARNING: не удалось загрузить {RESUME_FILE}: {e}", flush=True)
    return {}


# ===========================================================================
#  Структура троилита
# ===========================================================================

def build_troilite() -> "ase.Atoms":
    """Построить конвенциональную ячейку троилита (P-62c, #190).

    CIF 0004158, Skala 2006:
      a = 5.965 A, c = 11.757 A, gamma = 120 deg
      Fe1 at 12i: (0.3791, 0.0549, 0.1230)
      S1  at  2a: (0, 0, 0)
      S2  at  4f: (1/3, 2/3, 0.0198)
      S3  at  6h: (0.6653, -0.0035, 1/4)
    Итого: 24 атома (12 Fe + 12 S).
    """
    atoms = crystal(
        symbols=["Fe", "S", "S", "S"],
        basis=[
            (0.3791,  0.0549,  0.1230),   # Fe1 12i
            (0.0,     0.0,     0.0),       # S1   2a
            (1/3,     2/3,     0.0198),    # S2   4f
            (0.6653, -0.0035,  0.25),      # S3   6h
        ],
        spacegroup=190,
        cellpar=[5.965, 5.965, 11.757, 90, 90, 120],
    )
    return atoms


def set_afm_species(atoms):
    """Установить AFM начальные магнитные моменты для QE.

    Стратегия: один тип Fe + starting_magnetization(Fe)=+0.5 +
    tot_magnetization=0 (constraint). QE стартует с FM, constraint
    заставляет часть Fe перевернуть спин → AFM.
    Это подход Xisto 2025 (единственная QE работа по троилиту).

    Для ASE: устанавливаем initial_magnetic_moments по слоям
    (+2.0 / -2.0 по z-медиане) для визуализации/отладки.
    QE игнорирует per-atom magmoms — использует per-species starting_mag.
    """
    syms = list(atoms.get_chemical_symbols())
    fe_indices = [i for i, s in enumerate(syms) if s == "Fe"]

    if len(fe_indices) == 0:
        return

    fe_z = atoms.positions[fe_indices, 2]
    z_med = np.median(fe_z)

    magmoms = np.zeros(len(atoms))
    for idx in fe_indices:
        if atoms.positions[idx, 2] < z_med:
            magmoms[idx] = 2.0
        else:
            magmoms[idx] = -2.0

    atoms.set_initial_magnetic_moments(magmoms)


# ===========================================================================
#  Поиск ближайшей пары S-S
# ===========================================================================

def find_nearest_ss_pair(atoms) -> tuple:
    """Найти ближайшую пару атомов S для вакансионного механизма.

    Критерий: d_SS < 4.5 A (не дальние соседи).
    Возвращает (si_idx, sj_idx, distance).
    """
    syms = np.array(atoms.get_chemical_symbols())
    s_indices = np.where(syms == "S")[0]
    s_pos = atoms.positions[s_indices]

    _, d_matrix = get_distances(s_pos, cell=atoms.cell, pbc=True)

    min_d = np.inf
    si_local, sj_local = -1, -1
    for a in range(len(s_indices)):
        for b in range(a + 1, len(s_indices)):
            d = d_matrix[a, b]
            if 1.5 < d < 4.5 and d < min_d:
                min_d = d
                si_local, sj_local = a, b

    if si_local == -1:
        raise RuntimeError("Не найдено подходящей пары S-S в диапазоне 1.5-4.5 A")

    return s_indices[si_local], s_indices[sj_local], float(min_d)


# ===========================================================================
#  Подготовка эндпоинтов
# ===========================================================================

def prepare_endpoint(pristine, si_idx: int, sj_idx: int, h_on_idx: int):
    """Создать эндпоинт NEB: удалить 2 S, поставить H на место одной вакансии.

    si_idx, sj_idx: индексы удаляемых S в pristine.
    h_on_idx: индекс (si или sj), на место которого ставится H.
    Возвращает Atoms с 23 атомами (12 Fe + 10 S + 1 H).
    """
    ep = pristine.copy()
    pos_h = ep.positions[h_on_idx].copy()

    # Удаляем с большего индекса, чтобы не сбить меньший
    to_del = sorted([si_idx, sj_idx], reverse=True)
    for idx in to_del:
        del ep[idx]

    # Добавляем H на место вакансии
    ep.append(Atom("H", position=pos_h))

    # AFM магнитные моменты обновляем (после удаления атомов индексы изменились)
    set_afm_species(ep)

    return ep


def validate_endpoint(atoms, label: str, min_h_dist: float = 1.0):
    """Проверить что H не перекрывается с другими атомами."""
    syms = np.array(atoms.get_chemical_symbols())
    h_indices = np.where(syms == "H")[0]

    if len(h_indices) == 0:
        raise ValueError(f"{label}: атом H не найден!")

    for h_idx in h_indices:
        h_pos = atoms.positions[h_idx]
        other_pos = np.delete(atoms.positions, h_idx, axis=0)
        dists = np.linalg.norm(other_pos - h_pos, axis=1)
        min_d = np.min(dists)
        if min_d < min_h_dist:
            raise ValueError(
                f"{label}: H слишком близко к другому атому "
                f"(d={min_d:.3f} A < {min_h_dist} A)"
            )
    n_fe = int(np.sum(syms == "Fe"))
    n_s = int(np.sum(syms == "S"))
    n_h = int(np.sum(syms == "H"))
    print(f"  {label}: {len(atoms)} атомов (Fe={n_fe}, S={n_s}, H={n_h}), "
          f"min_H_dist={min_d:.3f} A", flush=True)
    sys.stdout.flush()


# ===========================================================================
#  QE Espresso calculator
# ===========================================================================

def make_espresso_calc(label: str, restart_mode: str = "from_scratch"):
    """Создать ASE Espresso calculator для QE 7.4.1.

    Параметры:
      - PBE, USPP, ecutwfc=60 Ry, ecutrho=480 Ry
      - nspin=2, AFM: tot_magnetization=0
      - mixing_mode='local-TF' (аналог Kerker для неоднородных систем)
      - disk_io='high': wfc на каждом SCF шаге (checkpoint при kill)
      - Hubbard U через additional_cards (QE 7.4.1 синтаксис)
    """
    from ase.calculators.espresso import Espresso, EspressoProfile

    work_subdir = WORK_DIR / label
    work_subdir.mkdir(parents=True, exist_ok=True)

    scratch_subdir = SCRATCH_DIR / label
    scratch_subdir.mkdir(parents=True, exist_ok=True)

    profile = EspressoProfile(
        command=QE_COMMAND,
        pseudo_dir=PSEUDO_DIR,
    )

    input_data = {
        # &CONTROL
        "calculation": "scf",
        "tprnfor": True,             # ОБЯЗАТЕЛЬНО: вычислять силы в SCF mode (для ASE FIRE)
        "tstress": True,             # стрессы (для полноты)
        "restart_mode": restart_mode,
        "outdir": str(scratch_subdir),
        "prefix": "troilite",
        "disk_io": "high",           # wfc на каждом SCF шаге для checkpoint
        "max_seconds": 36000,        # 10 часов graceful stop
        # &SYSTEM
        "ecutwfc": ECUTWFC,
        "ecutrho": ECUTRHO,
        "nspin": 2,
        "occupations": "smearing",
        "smearing": "mv",            # Marzari-Vanderbilt
        "degauss": 0.01,             # Ry (малый gap у FeS-полупроводника)
        "tot_magnetization": 0.0,    # AFM: суммарный момент = 0
        # &ELECTRONS
        "mixing_beta": 0.1,
        "mixing_mode": "local-TF",   # аналог Kerker для AFM (QE mailing list)
        "mixing_ndim": 10,
        "conv_thr": 1.0e-8,
        "electron_maxstep": 300,
    }

    # Starting magnetization: Fe = +0.5 (FM initial guess)
    # tot_magnetization = 0 (constraint) → QE SCF переворачивает часть Fe → AFM
    # Подход Xisto 2025: один тип Fe, AFM через tot_mag constraint
    input_data["starting_magnetization(1)"] = 0.5    # Fe (species 1)
    input_data["starting_magnetization(2)"] = 0.0    # S
    input_data["starting_magnetization(3)"] = 0.0    # H

    # Hubbard U: legacy синтаксис (ASE 3.28 не поддерживает HUBBARD карточку)
    # QE 7.4.1 принимает старый синтаксис с deprecation warning
    input_data["lda_plus_u"] = True
    input_data["Hubbard_U(1)"] = 2.0    # Fe: U_eff = 2.0 eV (Dudarev)

    calc = Espresso(
        profile=profile,
        directory=str(work_subdir),
        input_data=input_data,
        pseudopotentials=PSEUDOPOTENTIALS,
        kpts=KPTS,
    )

    return calc


# ===========================================================================
#  Релаксация эндпоинта
# ===========================================================================

def relax_endpoint(atoms, label: str, resume_data: dict) -> tuple:
    """Релаксировать эндпоинт с FIRE + QE.

    Для эндпоинтов: фиксируем тяжёлые атомы (Fe, S),
    двигаем только H (FixAtoms на всё кроме H).

    Возвращает (relaxed_atoms, energy, nsteps, converged).
    """
    print(f"\n  Релаксация {label}...", flush=True)
    sys.stdout.flush()

    # Ограничение: двигать только H
    h_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "H"]
    heavy_indices = [i for i in range(len(atoms)) if i not in h_indices]
    atoms.set_constraint(FixAtoms(indices=heavy_indices))

    # Определить restart_mode на основе наличия checkpoint QE
    scratch_subdir = SCRATCH_DIR / label / "troilite.save"
    restart_mode = "restart" if (scratch_subdir / "data-file-schema.xml").exists() else "from_scratch"
    if restart_mode == "restart":
        print(f"    QE checkpoint найден -> restart", flush=True)

    calc = make_espresso_calc(label=label, restart_mode=restart_mode)
    atoms.calc = calc

    traj_path = WORK_DIR / f"{label}.traj"
    log_path = WORK_DIR / f"{label}.log"

    from ase.io import Trajectory
    traj = Trajectory(str(traj_path), "a", atoms)

    def save_step():
        """Callback: сохранять xyz после каждого шага FIRE."""
        xyz_path = WORK_DIR / f"checkpoint_{label}.xyz"
        ase_write(str(xyz_path), atoms)
        sys.stdout.flush()

    t0 = time.time()
    opt = FIRE(
        atoms,
        logfile=str(log_path),
        trajectory=None,   # используем ручной traj
    )
    opt.attach(traj, interval=1)
    opt.attach(save_step, interval=1)

    try:
        converged = opt.run(fmax=FMAX_RELAX, steps=MAX_STEPS_RELAX)
    except Exception as e:
        print(f"    FIRE FAILED: {e}", flush=True)
        traceback.print_exc()
        converged = False

    energy = float("nan")
    try:
        energy = float(atoms.get_potential_energy())
    except Exception:
        pass

    dt = time.time() - t0
    print(f"    {label}: E={energy:.4f} eV, шагов={opt.nsteps}, "
          f"сошлось={converged}, время={dt:.1f}s", flush=True)
    sys.stdout.flush()

    # Сохранить финальную геометрию
    xyz_path = WORK_DIR / f"checkpoint_{label}.xyz"
    ase_write(str(xyz_path), atoms)

    return atoms, energy, int(opt.nsteps), bool(converged)


# ===========================================================================
#  CI-NEB
# ===========================================================================

def run_neb(endA, endB, resume: dict) -> dict:
    """Запустить CI-NEB с IDPP интерполяцией, FIRE optimizer.

    Возвращает dict с результатами (energies, barrier, converged, ...).
    """
    print(f"\n[NEB] CI-NEB с {N_IMAGES} промежуточными образами", flush=True)
    sys.stdout.flush()

    # Фиксируем тяжёлые атомы во всех образах
    h_indices = [i for i, s in enumerate(endA.get_chemical_symbols()) if s == "H"]
    heavy_indices = [i for i in range(len(endA)) if i not in h_indices]

    # Эндпоинты: copy() теряет калькулятор, нужно назначить заново
    # NEB не оптимизирует эндпоинты, но ТРЕБУЕТ их энергии для профиля
    end_a = endA.copy()
    end_b = endB.copy()
    end_a.set_constraint(FixAtoms(indices=heavy_indices))
    end_b.set_constraint(FixAtoms(indices=heavy_indices))

    # Назначить калькуляторы эндпоинтам (restart из checkpoint — SCF за 1 iter)
    scratch_a = SCRATCH_DIR / "endA" / "troilite.save"
    rm_a = "restart" if (scratch_a / "data-file-schema.xml").exists() else "from_scratch"
    end_a.calc = make_espresso_calc(label="endA", restart_mode=rm_a)

    scratch_b = SCRATCH_DIR / "endB" / "troilite.save"
    rm_b = "restart" if (scratch_b / "data-file-schema.xml").exists() else "from_scratch"
    end_b.calc = make_espresso_calc(label="endB", restart_mode=rm_b)

    # Промежуточные образы
    images = [end_a]
    for i in range(N_IMAGES):
        img = end_a.copy()
        img.set_constraint(FixAtoms(indices=heavy_indices))
        images.append(img)
    images.append(end_b)

    # IDPP интерполяция (критично для сложных структур)
    neb = NEB(images, climb=True, allow_shared_calculator=False, k=0.1,
              method="improvedtangent")
    neb.interpolate("idpp")
    print("  IDPP интерполяция готова", flush=True)

    # Вывести H-позиции вдоль пути
    h_idx_in_img = h_indices[0]
    print("  H-позиции вдоль пути (IDPP):", flush=True)
    for k, img in enumerate(images):
        h_pos = img.positions[h_idx_in_img]
        print(f"    образ {k}: H=({h_pos[0]:.3f}, {h_pos[1]:.3f}, {h_pos[2]:.3f})",
              flush=True)
    sys.stdout.flush()

    # Назначить калькуляторы промежуточным образам (после интерполяции!)
    for i in range(1, len(images) - 1):
        lbl = f"neb_img{i:02d}"
        scratch_subdir = SCRATCH_DIR / lbl / "troilite.save"
        restart_mode = "restart" if (scratch_subdir / "data-file-schema.xml").exists() else "from_scratch"
        images[i].calc = make_espresso_calc(label=lbl, restart_mode=restart_mode)

    # FIRE для NEB
    neb_log = WORK_DIR / "neb.log"
    neb_traj = WORK_DIR / "neb.traj"

    from ase.io import Trajectory
    traj = Trajectory(str(neb_traj), "a", neb)

    # Checkpoint callback: сохраняем все образы после каждого NEB шага
    step_counter = [0]
    checkpoint_energies = []

    def neb_step_callback():
        step_counter[0] += 1
        step = step_counter[0]
        # Сохранить xyz каждые 5 шагов
        if step % 5 == 0:
            for k, img in enumerate(images):
                xyz_path = WORK_DIR / f"neb_img{k:02d}_step{step:04d}.xyz"
                try:
                    ase_write(str(xyz_path), img)
                except Exception:
                    pass
        # Логировать прогресс
        try:
            energies = []
            for img in images:
                try:
                    energies.append(float(img.get_potential_energy()))
                except Exception:
                    energies.append(float("nan"))
            e_ref = energies[0]
            rel_e = [e - e_ref for e in energies]
            barrier_est = max(e for e in rel_e if not np.isnan(e))
            checkpoint_energies.append({"step": step, "barrier_eV": barrier_est})
            print(f"  NEB шаг {step:4d}: barrier_est={barrier_est:.4f} eV", flush=True)
        except Exception:
            pass
        sys.stdout.flush()

    opt_neb = FIRE(
        neb,
        logfile=str(neb_log),
        trajectory=None,
    )
    opt_neb.attach(traj, interval=1)
    opt_neb.attach(neb_step_callback, interval=1)

    t0 = time.time()
    neb_converged = False
    try:
        neb_converged = opt_neb.run(fmax=FMAX_NEB, steps=MAX_STEPS_NEB)
    except Exception as e:
        print(f"  NEB FAILED: {e}", flush=True)
        traceback.print_exc()

    dt_neb = time.time() - t0

    # Извлечь энергии
    energies = []
    for img in images:
        try:
            energies.append(float(img.get_potential_energy()))
        except Exception:
            energies.append(float("nan"))

    e_ref = energies[0]
    rel_energies = [e - e_ref for e in energies]
    valid_rel = [e for e in rel_energies if not np.isnan(e)]
    barrier = float(max(valid_rel)) if valid_rel else float("nan")
    barrier_back = float(rel_energies[-1]) if not np.isnan(rel_energies[-1]) else float("nan")

    print(f"\n  Результаты NEB:", flush=True)
    print(f"    Сошлось: {neb_converged} ({opt_neb.nsteps} шагов, {dt_neb:.0f}s)", flush=True)
    for k, e in enumerate(rel_energies):
        marker = ""
        if not np.isnan(e) and e == barrier and 0 < k < len(rel_energies) - 1:
            marker = " <-- TS"
        print(f"    образ {k}: {e:+.4f} eV{marker}", flush=True)
    print(f"    E_a (прямой барьер): {barrier:.4f} eV", flush=True)
    if not np.isnan(barrier_back):
        print(f"    E_a (обратный, dE): {barrier - barrier_back:.4f} eV (dE={barrier_back:+.4f})", flush=True)
    sys.stdout.flush()

    result = {
        "n_images_total": len(images),
        "n_images_intermediate": N_IMAGES,
        "fmax_neb": FMAX_NEB,
        "converged": bool(neb_converged),
        "neb_steps": int(opt_neb.nsteps),
        "time_neb_s": round(dt_neb, 1),
        "energies_eV": rel_energies,
        "E_a_forward_eV": barrier,
        "dE_reaction_eV": barrier_back,
        "neb_step_log": checkpoint_energies[-10:],  # последние 10 шагов
    }
    return result


# ===========================================================================
#  Анализ диффузии
# ===========================================================================

def diffusion_analysis(E_a: float, hop_dist_A: float) -> dict:
    """Рассчитать D_H и tau по Аррениусу."""
    kB = 8.617333e-5    # eV/K
    T = 298.15           # K
    nu = 1e13            # Гц (попытки прыжка)
    hop_cm = hop_dist_A * 1e-8

    if np.isnan(E_a) or E_a <= 0:
        return {}

    D_H = nu * hop_cm**2 * np.exp(-E_a / (kB * T))
    L = 200e-7           # 200 нм в см
    tau = L**2 / (2 * D_H) if D_H > 0 else float("inf")

    print(f"\n  Анализ диффузии (T={T:.1f} K):", flush=True)
    print(f"    E_a = {E_a:.4f} eV", flush=True)
    print(f"    расстояние прыжка = {hop_dist_A:.3f} A", flush=True)
    print(f"    D_H = {D_H:.3e} cm^2/s", flush=True)
    print(f"    tau (200 нм) = {tau:.3e} s", flush=True)
    sys.stdout.flush()

    return {
        "T_K": T,
        "nu_Hz": nu,
        "hop_dist_A": hop_dist_A,
        "D_H_cm2s": D_H,
        "tau_200nm_s": tau,
    }


# ===========================================================================
#  Основной скрипт
# ===========================================================================

def main():
    t_total = time.time()

    # Создать рабочие директории
    RESULTS.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("  QE NEB: H-диффузия в троилите FeS (P-62c)", flush=True)
    print(f"  ecutwfc={ECUTWFC} Ry, ecutrho={ECUTRHO} Ry, kpts={KPTS}", flush=True)
    print(f"  U_eff=2.0 eV (Fe 3d, Dudarev, Xisto 2025)", flush=True)
    print(f"  N_images={N_IMAGES}, fmax_neb={FMAX_NEB} eV/A", flush=True)
    print("=" * 70, flush=True)
    sys.stdout.flush()

    # Инфо о процессоре
    n_omp = os.environ.get("OMP_NUM_THREADS", "?")
    print(f"  OMP_NUM_THREADS = {n_omp}", flush=True)
    sys.stdout.flush()

    results = {
        "system": "troilite_FeS_vacancy_hop",
        "method": "QE_PBE_USPP",
        "spacegroup": "P-62c #190",
        "ecutwfc_Ry": ECUTWFC,
        "ecutrho_Ry": ECUTRHO,
        "kpts": list(KPTS),
        "U_eff_Fe_eV": 2.0,
        "hubbard_method": "Dudarev (ortho-atomic)",
        "n_images": N_IMAGES,
        "fmax_relax": FMAX_RELAX,
        "fmax_neb": FMAX_NEB,
        "pseudopotentials": PSEUDOPOTENTIALS,
    }

    resume = load_resume()
    json_path = RESULTS / "neb_troilite_qe_results.json"

    # ------------------------------------------------------------------
    # [1/7] Построить ячейку троилита
    # ------------------------------------------------------------------
    print("\n[1/7] Построить конвенциональную ячейку троилита", flush=True)
    sys.stdout.flush()
    t0 = time.time()

    pristine = build_troilite()
    n_pristine = len(pristine)
    syms = pristine.get_chemical_symbols()
    n_fe = syms.count("Fe")
    n_s = syms.count("S")

    assert n_pristine == 24, f"Ожидалось 24 атома, получено {n_pristine}"
    assert n_fe == 12, f"Ожидалось 12 Fe, получено {n_fe}"
    assert n_s == 12, f"Ожидалось 12 S, получено {n_s}"

    cell_lengths = pristine.cell.lengths().tolist()
    print(f"  Формула: {pristine.get_chemical_formula()}, {n_pristine} атомов", flush=True)
    print(f"  Ячейка: a={cell_lengths[0]:.3f}, b={cell_lengths[1]:.3f}, "
          f"c={cell_lengths[2]:.3f} A", flush=True)

    # Установить AFM моменты (для отладки; QE использует starting_magnetization per species)
    set_afm_species(pristine)
    magmoms = pristine.get_initial_magnetic_moments()
    n_up = int(np.sum(magmoms > 0))
    n_dn = int(np.sum(magmoms < 0))
    print(f"  AFM моменты: {n_up} Fe↑ (+2.0), {n_dn} Fe↓ (-2.0), "
          f"sum={magmoms.sum():.2f} muB", flush=True)
    sys.stdout.flush()

    results["formula_pristine"] = pristine.get_chemical_formula()
    results["n_atoms_pristine"] = n_pristine
    results["cell_A"] = cell_lengths
    results["afm_n_up"] = n_up
    results["afm_n_down"] = n_dn
    print(f"  Время: {time.time()-t0:.1f}s", flush=True)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # [2/7] Найти ближайшую пару S-S
    # ------------------------------------------------------------------
    print("\n[2/7] Найти ближайшую пару S-S (вакансионный путь)", flush=True)
    sys.stdout.flush()
    t0 = time.time()

    si_idx, sj_idx, ss_dist = find_nearest_ss_pair(pristine)
    print(f"  Пара S: атомы {si_idx} & {sj_idx}, расстояние = {ss_dist:.3f} A", flush=True)
    print(f"  S_i позиция: {pristine.positions[si_idx]}", flush=True)
    print(f"  S_j позиция: {pristine.positions[sj_idx]}", flush=True)
    sys.stdout.flush()

    results["S_pair_indices"] = [int(si_idx), int(sj_idx)]
    results["S_pair_dist_A"] = ss_dist
    print(f"  Время: {time.time()-t0:.1f}s", flush=True)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # [3/7] Подготовить эндпоинты
    # ------------------------------------------------------------------
    print("\n[3/7] Подготовить эндпоинты NEB", flush=True)
    sys.stdout.flush()
    t0 = time.time()

    endA = prepare_endpoint(pristine, si_idx, sj_idx, h_on_idx=si_idx)
    endB = prepare_endpoint(pristine, si_idx, sj_idx, h_on_idx=sj_idx)

    validate_endpoint(endA, "endA")
    validate_endpoint(endB, "endB")

    results["formula_endpoint"] = endA.get_chemical_formula()
    results["n_atoms_endpoint"] = len(endA)
    print(f"  Время: {time.time()-t0:.1f}s", flush=True)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # [4/7] Релаксация эндпоинта A
    # ------------------------------------------------------------------
    print("\n[4/7] Релаксация эндпоинта A (H у S_i)", flush=True)
    sys.stdout.flush()

    if "relax_start" in resume and CHECKPOINT_START_XYZ.exists():
        print("  [resume] эндпоинт A уже релаксирован, загружаю...", flush=True)
        endA = ase_read(str(CHECKPOINT_START_XYZ))
        set_afm_species(endA)
        e_A = float(resume["relax_start"].get("energy_eV", float("nan")))
        steps_A = int(resume["relax_start"].get("nsteps", 0))
        converged_A = bool(resume["relax_start"].get("converged", False))
    else:
        endA, e_A, steps_A, converged_A = relax_endpoint(endA, "endA", resume)
        ase_write(str(CHECKPOINT_START_XYZ), endA)
        save_resume("relax_start", {
            "energy_eV": e_A, "nsteps": steps_A, "converged": converged_A
        })

    results["E_endA_eV"] = e_A
    results["steps_relax_A"] = steps_A
    results["converged_relax_A"] = converged_A
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # [5/7] Релаксация эндпоинта B
    # ------------------------------------------------------------------
    print("\n[5/7] Релаксация эндпоинта B (H у S_j)", flush=True)
    sys.stdout.flush()

    if "relax_end" in resume and CHECKPOINT_END_XYZ.exists():
        print("  [resume] эндпоинт B уже релаксирован, загружаю...", flush=True)
        endB = ase_read(str(CHECKPOINT_END_XYZ))
        set_afm_species(endB)
        e_B = float(resume["relax_end"].get("energy_eV", float("nan")))
        steps_B = int(resume["relax_end"].get("nsteps", 0))
        converged_B = bool(resume["relax_end"].get("converged", False))
    else:
        endB, e_B, steps_B, converged_B = relax_endpoint(endB, "endB", resume)
        ase_write(str(CHECKPOINT_END_XYZ), endB)
        save_resume("relax_end", {
            "energy_eV": e_B, "nsteps": steps_B, "converged": converged_B
        })

    results["E_endB_eV"] = e_B
    results["steps_relax_B"] = steps_B
    results["converged_relax_B"] = converged_B

    dE = abs(e_A - e_B) if (not np.isnan(e_A) and not np.isnan(e_B)) else float("nan")
    print(f"  |E_A - E_B| = {dE:.4f} eV (дисбаланс эндпоинтов)", flush=True)
    results["dE_endpoints_eV"] = dE
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # [6/7] CI-NEB
    # ------------------------------------------------------------------
    print("\n[6/7] CI-NEB с IDPP интерполяцией", flush=True)
    sys.stdout.flush()

    neb_result = {}
    barrier = float("nan")

    if "neb" in resume:
        print("  [resume] NEB уже завершён, загружаю результаты...", flush=True)
        neb_result = resume["neb"]
        barrier = float(neb_result.get("E_a_forward_eV", float("nan")))
    else:
        t0 = time.time()
        try:
            neb_result = run_neb(endA, endB, resume)
            barrier = float(neb_result.get("E_a_forward_eV", float("nan")))
            save_resume("neb", neb_result)
        except Exception as e:
            print(f"  NEB EXCEPTION: {e}", flush=True)
            traceback.print_exc()
            neb_result["error"] = str(e)
        sys.stdout.flush()

    results.update(neb_result)

    # ------------------------------------------------------------------
    # [7/7] Анализ диффузии и сохранение
    # ------------------------------------------------------------------
    print("\n[7/7] Анализ диффузии и сохранение результатов", flush=True)
    sys.stdout.flush()

    if not np.isnan(barrier):
        diff_data = diffusion_analysis(barrier, ss_dist)
        results.update(diff_data)

    total_time = time.time() - t_total
    results["total_time_s"] = round(total_time, 1)

    # Сохранить JSON с NumpyEncoder
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"  Сохранено: {json_path}", flush=True)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Итоговый отчёт
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("  ИТОГИ", flush=True)
    print("=" * 70, flush=True)
    print(f"  Система:       FeS троилит P-62c, 23 атома (H в вакансии S)", flush=True)
    print(f"  Метод:         QE PBE USPP, U_eff=2.0 eV, kpts={KPTS}", flush=True)
    print(f"  S-S расстояние: {ss_dist:.3f} A", flush=True)
    if not np.isnan(e_A) and not np.isnan(e_B):
        print(f"  E_endA:        {e_A:.4f} eV", flush=True)
        print(f"  E_endB:        {e_B:.4f} eV", flush=True)
        print(f"  |dE_endpoints|: {dE:.4f} eV", flush=True)
    if not np.isnan(barrier):
        print(f"  E_a (барьер):  {barrier:.4f} eV", flush=True)
        if "D_H_cm2s" in results:
            print(f"  D_H (298 K):   {results['D_H_cm2s']:.3e} cm^2/s", flush=True)
            print(f"  tau (200 нм):  {results['tau_200nm_s']:.3e} s", flush=True)
    else:
        print(f"  E_a:           FAILED / NaN", flush=True)
    print(f"  NEB сошлось:   {neb_result.get('converged', 'N/A')}", flush=True)
    print(f"  Общее время:   {total_time:.0f}s ({total_time/60:.1f} мин)", flush=True)
    print("=" * 70, flush=True)
    print("DONE", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
