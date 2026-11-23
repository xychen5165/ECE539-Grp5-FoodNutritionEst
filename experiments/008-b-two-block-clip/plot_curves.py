"""Render the measured B6 train/dev curves from retained numeric evidence."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--results',type=Path,required=True)
parser.add_argument('--b4',type=Path,required=True)
parser.add_argument('--b5',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
with (args.results/'training_log.csv').open() as f:
    rows=list(csv.DictReader(f))
metrics=json.loads((args.results/'metrics.json').read_text())
b4=json.loads(args.b4.read_text())['summary']['normalized_nutrient_mae_mean']
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
colors=['#2563eb','#d97706','#16a34a']
for seed,color in zip((0,1,2),colors):
    series=[r for r in rows if int(r['seed'])==seed]
    assert [int(r['epoch']) for r in series]==list(range(1,31))
    epochs=[int(r['epoch']) for r in series]
    train=[float(r['train_loss']) for r in series]
    dev=[float(r['dev_normalized_nutrient_mae']) for r in series]
    axes[0].plot(epochs,train,color=color,label=f'Seed {seed}',linewidth=1.6)
    axes[1].plot(epochs,dev,color=color,label=f'Seed {seed}',linewidth=1.6)
    best=int(metrics['per_seed'][str(seed)]['best_epoch'])
    axes[1].scatter([best],[dev[best-1]],color=color,s=32,zorder=4)
axes[1].axhline(b4,color='#475569',linestyle='--',linewidth=1,label='B4 seed mean')
if args.b5:
    b5=json.loads(args.b5.read_text())['summary']['normalized_nutrient_mae_mean']
    axes[1].axhline(b5,color='#9333ea',linestyle=':',linewidth=1.2,label='B5 seed mean')
for ax in axes:
    ax.axvspan(1,3.5,color='#94a3b8',alpha=.12)
    ax.axvline(3.5,color='#64748b',linestyle=':',linewidth=.8)
    ax.set(xlabel='Epoch',xlim=(1,30))
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
axes[0].set(title='Training objective',ylabel='Mean-scaled L1 loss, five targets')
axes[1].set(title='Development model selection',ylabel='Normalized nutrient MAE, four nutrients')
axes[0].legend(frameon=False)
axes[1].legend(frameon=False,fontsize=8)
fig.suptitle('B6: two-block CLIP adaptation | Original RGB + natural crop',fontsize=12)
fig.text(.5,-.035,'Shading: three head-only warmup epochs. Dots: selected checkpoints. Development evidence; final test not evaluated.',ha='center',fontsize=8,color='#475569')
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output,dpi=170,bbox_inches='tight',facecolor='white')
plt.close(fig)
