"""
Ночная программа GPU расчётов для Q-071 и валидации.
Запуск: python scripts/run_overnight.py

Расчёты запускаются последовательно (один GPU):
1. Пентландит 2×2×2 суперячейка NEB (конвергенция по размеру)
2. Пентландит NN-hop NEB (альтернативный путь миграции)
3. Макинавит NEB (катализатор R1, два пути)
4. MD пентландит + H при 300K (диффузия из динамики)
"""
import subprocess
import time

SCRIPTS = [
    ('pentlandite_2x2x2', 'neb_pentlandite_2x2x2_gpu.py'),
    ('pentlandite_nn',     'neb_pentlandite_nn_gpu.py'),
    ('mackinawite',        'neb_mackinawite_gpu.py'),
    ('md_pentlandite_h',   'md_pentlandite_h_gpu.py'),
]

def run_one(name, script):
    print(f"\n{'='*70}")
    print(f"  ЗАПУСК: {name} ({script})")
    print(f"  Время: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n", flush=True)
    t0 = time.time()
    result = subprocess.run(
        ['python', '-u', f'/workspace/scripts/{script}'],
        cwd='/workspace'
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"ОШИБКА (код {result.returncode})"
    print(f"\n  {name}: {status}, время: {elapsed/60:.1f} мин")
    print(f"  Завершено: {time.strftime('%Y-%m-%d %H:%M:%S')}\n", flush=True)
    return result.returncode == 0

if __name__ == '__main__':
    print(f"Ночная программа GPU расчётов")
    print(f"Старт: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Расчётов: {len(SCRIPTS)}\n")

    results = {}
    t_total = time.time()

    for name, script in SCRIPTS:
        results[name] = run_one(name, script)

    elapsed_total = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  НОЧНАЯ ПРОГРАММА ЗАВЕРШЕНА")
    print(f"  Общее время: {elapsed_total/60:.1f} мин ({elapsed_total/3600:.1f} ч)")
    print(f"  Завершено: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for name, ok in results.items():
        print(f"    {name}: {'OK' if ok else 'FAIL'}")
    print(f"{'='*70}")
