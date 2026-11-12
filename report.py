import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Sabse naya session log dhoondho ----------
files = sorted(glob.glob('logs/session_*.csv'))
if not files:
    raise SystemExit('ERROR: logs folder mein koi CSV nahi mili. Pehle app.py chalao.')
path = files[-1]
df = pd.read_csv(path)
if len(df) < 2:
    raise SystemExit('ERROR: CSV mein bahut kam data hai. App ko thoda zyada der chalao.')

emotions = [c for c in df.columns if c not in ('time_sec', 'dominant')]

COLORS = {
    'angry': '#e53935', 'disgust': '#2e7d32', 'fear': '#8e24aa',
    'happy': '#fbc02d', 'neutral': '#757575', 'sad': '#1e88e5', 'surprise': '#fb8c00',
}

# ---------- Summary ----------
duration = df['time_sec'].max()
share = df['dominant'].value_counts(normalize=True) * 100
avg = df[emotions].mean() * 100
peak_row = df.loc[df[emotions].max(axis=1).idxmax()]

print('=' * 50)
print('SESSION SUMMARY:', os.path.basename(path))
print('=' * 50)
print(f'Duration        : {duration:.1f} sec ({len(df)} readings)')
print(f'Top emotion     : {share.index[0]} ({share.iloc[0]:.1f}% of the time)')
print('Time share by dominant emotion:')
for name, val in share.items():
    print(f'   {name:<9}: {val:5.1f}%')
print(f'Strongest moment: {peak_row["dominant"]} at {peak_row["time_sec"]:.1f}s '
      f'({peak_row[emotions].max() * 100:.1f}%)')

# ---------- Graph ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [2, 1]})

for e in emotions:
    ax1.plot(df['time_sec'], df[e] * 100, label=e, color=COLORS.get(e), linewidth=2)
ax1.set_title('Emotion timeline: ' + os.path.basename(path), fontsize=14, fontweight='bold')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Confidence (%)')
ax1.set_ylim(0, 100)
ax1.grid(alpha=0.3)
ax1.legend(ncol=7, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

share_sorted = share.sort_values()
ax2.barh(share_sorted.index, share_sorted.values,
         color=[COLORS.get(n) for n in share_sorted.index])
ax2.set_title('Time share by dominant emotion', fontsize=12)
ax2.set_xlabel('% of session')
for i, v in enumerate(share_sorted.values):
    ax2.text(v + 0.5, i, f'{v:.1f}%', va='center')
ax2.set_xlim(0, max(share_sorted.values) + 10)

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
out = os.path.join('outputs', os.path.basename(path).replace('.csv', '_report.png'))
plt.savefig(out, dpi=150)
print('\nGraph saved:', out)
plt.show()