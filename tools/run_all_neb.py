"""
Запуск всех 4 NEB расчётов последовательно.
Использование: python scripts/run_all_neb.py [mineral]
  mineral = pentlandite | greigite | violarite | pyrite | all (default)
"""
import subprocess
import sys
import time

SCRIPTS = {
    'pentlandite': 'neb_pentlandite_gpu.py',
    'greigite':    'neb_greigite_gpu.py',
    'violarite':   'neb_violarite_gpu.py',
    'pyrite':      'neb_pyrite_gpu.py',
}

def run_one(name, script):
    print(f"\n{'='*70}")
    print(f"  ЗАПУСК: {name} ({script})")
    print(f"{'='*70}\n")
    t0 = time.time()
    result = subprocess.run(
        ['python', '-u', f'/workspace/scripts/{script}'],
        cwd='/workspace'
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"ОШИБКА (код {result.returncode})"
    print(f"\n  {name}: {status}, время: {elapsed/60:.1f} мин\n")
    return result.returncode == 0

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if target == 'all':
        to_run = list(SCRIPTS.items())
    elif target in SCRIPTS:
        to_run = [(target, SCRIPTS[target])]
    else:
        print(f"Неизвестный минерал: {target}")
        print(f"Доступные: {', '.join(SCRIPTS.keys())}, all")
        sys.exit(1)

    print(f"Расчёт: {', '.join(n for n, _ in to_run)}")
    results = {}
    t_total = time.time()

    for name, script in to_run:
        results[name] = run_one(name, script)

    elapsed_total = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  ИТОГО: {elapsed_total/60:.1f} мин")
    for name, ok in results.items():
        print(f"    {name}: {'OK' if ok else 'FAIL'}")
    print(f"{'='*70}")
