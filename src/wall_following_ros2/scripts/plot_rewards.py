#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_one(path, default_algo):
    path = Path(path).expanduser()
    if not path.exists():
        print(f"⚠️  Missing: {path}")
        return None
    df = pd.read_csv(path)

    # normalize & validate
    df.columns = [c.strip().lower() for c in df.columns]
    need = {'episode','return','algorithm'}
    missing = need - set(df.columns)
    if 'algorithm' in missing:
        df['algorithm'] = default_algo
        missing.remove('algorithm')
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")

    # numeric & de-dupe per episode
    df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
    df['return']  = pd.to_numeric(df['return'],  errors='coerce')
    df = df.dropna(subset=['episode','return'])
    df = df.sort_values('episode').drop_duplicates('episode', keep='first')
    return df[['episode','return','algorithm']]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--q',     default='~/.ros/wf_train_log.csv',        help='Q-learning log')
    ap.add_argument('--sarsa', default='~/.ros/wf_sarsa_train_log.csv',  help='SARSA log')
    ap.add_argument('--out',   default='~/.ros/reward_curve.png',        help='output figure path')
    ap.add_argument('--window', type=int, default=20,                    help='EWM smoothing span (episodes)')
    ap.add_argument('--skip_start', type=int, default=0,                 help='burn-in episodes to skip from start')
    ap.add_argument('--style', default='seaborn-v0_8-whitegrid',         help='matplotlib style')
    args = ap.parse_args()

    dfs = []
    q = load_one(args.q, 'q_learning')
    if q is not None: dfs.append(q)
    s = load_one(args.sarsa, 'sarsa')
    if s is not None: dfs.append(s)
    if not dfs:
        raise SystemExit("No valid logs found.")

    df = pd.concat(dfs, ignore_index=True).sort_values(['algorithm', 'episode']).reset_index(drop=True)

    # Rank episodes within each algorithm, apply burn-in trim, then align to common length
    df['rank'] = df.groupby('algorithm').cumcount()  # 0-based
    df = df[df['rank'] >= args.skip_start].copy()
    df['idx'] = df.groupby('algorithm').cumcount() + 1  # reindex after trimming
    # common number of episodes across algorithms (after trimming)
    common_len = int(df.groupby('algorithm')['idx'].max().min())
    if common_len <= 0:
        raise SystemExit("Not enough overlapping episodes after --skip_start trimming.")
    df = df[df['idx'] <= common_len].copy()

    # Plot
    try:
        plt.style.use(args.style)
    except Exception:
        pass

    plt.figure(figsize=(8.6, 4.8))
    colors = {'q_learning':'#1f77b4', 'sarsa':'#ff7f0e'}

    for algo, g in df.groupby('algorithm', sort=False):
        g = g.sort_values('idx')
        y = g['return'].ewm(span=max(args.window, 2), adjust=False).mean()
        label = algo.replace('_', ' ').title()
        plt.plot(g['idx'], y, label=label, linewidth=1.6, alpha=0.95, color=colors.get(algo, None))

    plt.xlabel("Episodes (aligned & trimmed)")
    plt.ylabel("Accumulated Reward (per episode)")
    plt.title("Episodic Return (smoothed) — SARSA vs Q-learning")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=True)
    plt.ylim(-2000, 3000)              # requested fixed range to avoid visual clutter
    plt.margins(x=0.02)
    plt.tight_layout()

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    print(f"✅ Saved: {out}  (episodes plotted per curve: {common_len}, skipped first {args.skip_start})")

if __name__ == '__main__':
    main()
