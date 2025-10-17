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
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # keep only the fields we need
    need = {'episode','return','algorithm'}
    missing = need - set(df.columns)
    if 'algorithm' in missing:
        df['algorithm'] = default_algo
        missing.remove('algorithm')
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    # ensure numeric
    df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
    df['return']  = pd.to_numeric(df['return'],  errors='coerce')
    df = df.dropna(subset=['episode','return'])
    # if logs were appended across sessions and restarted at episode=1,
    # sort and keep the first occurrence of each episode
    df = df.sort_values('episode').drop_duplicates('episode', keep='first')
    return df[['episode','return','algorithm']]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--q',     default='~/.ros/wf_train_log.csv', help='Q-learning log')
    ap.add_argument('--sarsa', default='~/.ros/wf_sarsa_train_log.csv', help='SARSA log')
    ap.add_argument('--out',   default='~/.ros/reward_curve.png', help='output figure path')
    ap.add_argument('--window', type=int, default=10, help='rolling mean window (episodes)')
    args = ap.parse_args()

    dfs = []
    q = load_one(args.q, 'q_learning')
    if q is not None: dfs.append(q)
    s = load_one(args.sarsa, 'sarsa')
    if s is not None: dfs.append(s)
    if not dfs:
        raise SystemExit("No valid logs found.")

    df = pd.concat(dfs, ignore_index=True)

    plt.figure(figsize=(7.2, 4.2))
    for algo, g in df.groupby('algorithm'):
        g = g.sort_values('episode')
        y = g['return'].rolling(args.window, min_periods=2).mean()
        plt.plot(g['episode'], y, label=algo.capitalize())
    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward (per episode)")
    plt.title("Episodic Return: SARSA vs Q-learning")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=250)
    print(f"✅ Saved: {out}")

if __name__ == '__main__':
    main()
