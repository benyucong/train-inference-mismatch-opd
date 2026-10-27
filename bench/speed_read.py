"""Updates per hour of training runs, read from their Slurm logs (stdlib only; runs on a login node).

    python3 bench/speed_read.py <reference log> [<log> ...] --last N [--json out.json]

The paper's throughput rule. Steps 1 and 2 are left out (one-time warm-up). A synchronous run's wall time is the sum
of timing_s/step over steps 3..N; an asynchronous run's (its launcher banner says "[ASYNC] mode=fully_async") is the
driver's [STEP-WALL] stamp at step N minus its stamp at step 2, which leaves out the pipeline fill and the end-of-run
drain. Updates per hour = (N - 2) * 3600 / wall. Every ratio is against the first log, so put the BF16 run first and
read every run over the same N. Mean generation time is the mean of timing_s/gen over steps 3..N (the trainer's wait
for rollouts in an asynchronous run).
"""
import argparse
import json
import re

STEP = re.compile(r"\bstep:(\d+) - (.*)$")
KV = re.compile(r"([A-Za-z0-9_/@.\-]+):(np\.\w+\()?([-+]?(?:nan|inf)|[-+0-9.eE]+)\)?")
WALL = re.compile(r"\[STEP-WALL\] step=(\d+) t=([0-9]+(?:\.[0-9]*)?)")
ASYNC = "[ASYNC] mode=fully_async"


def scan(path):
    steps, walls, asynchronous = {}, {}, False
    for line in open(path, errors="replace"):
        if line.startswith(ASYNC):
            asynchronous = True
        m = WALL.search(line)
        if m and ("TaskRunner" in line or line.startswith("[STEP-WALL]")):
            walls[int(m.group(1))] = float(m.group(2))
            continue
        m = STEP.search(line)
        if m and "TaskRunner" in line:
            steps[int(m.group(1))] = {k: float(v) for k, _, v in KV.findall(m.group(2))}
    return steps, walls, asynchronous


def throughput(path, n_last):
    steps, walls, asynchronous = scan(path)
    need = range(3, n_last + 1)
    out = dict(log=path, mode="asynchronous" if asynchronous else "synchronous", steps=[3, n_last])
    if n_last < 4 or any(t not in steps for t in need):
        return dict(out, status="INCOMPLETE", why=f"steps 3..{n_last} not all logged")
    if asynchronous:
        if 2 not in walls or n_last not in walls:
            return dict(out, status="INCOMPLETE", why=f"[STEP-WALL] stamps for steps 2 and {n_last} missing")
        wall = walls[n_last] - walls[2]
    else:
        if any("timing_s/step" not in steps[t] for t in need):
            return dict(out, status="INCOMPLETE", why="timing_s/step missing")
        wall = sum(steps[t]["timing_s/step"] for t in need)
    gen = [steps[t]["timing_s/gen"] for t in need if "timing_s/gen" in steps[t]]
    return dict(out, status="READ", wall_s=wall, updates_per_hour=(n_last - 2) * 3600 / wall,
                mean_step_s=wall / (n_last - 2), mean_gen_s=sum(gen) / len(gen) if gen else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="+", help="Slurm logs; the first is the reference")
    ap.add_argument("--last", type=int, required=True, help="last step of the window (steps 3..N)")
    ap.add_argument("--json", help="also write the readings here")
    a = ap.parse_args()
    rows = [throughput(p, a.last) for p in a.logs]
    ref = rows[0].get("updates_per_hour")
    for r in rows:
        r["ratio"] = r["updates_per_hour"] / ref if ref and r.get("updates_per_hour") else None
        if r["status"] != "READ":
            print(f"{r['status']:10s} {r['log']}: {r['why']}")
            continue
        gen = f"{r['mean_gen_s']:7.1f}" if r["mean_gen_s"] is not None else "    n/a"
        ratio = f"{r['ratio']:.3f}" if r["ratio"] is not None else "n/a"
        print(f"{r['updates_per_hour']:7.2f} updates/h  x{ratio}  step {r['mean_step_s']:6.1f} s  gen {gen} s  "
              f"{r['mode']:12s} {r['log']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
