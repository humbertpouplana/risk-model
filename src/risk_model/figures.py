"""Regenerate every figure of the paper from results/results.json.

Run:  python -m risk_model.figures
"""
import json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper'))
from tex_common import EN
OUT = ROOT / 'paper' / 'fig'
R = json.load(open(ROOT / 'results' / 'results.json'))
B,G,A,Rd,Gy,Pu='#2f6fe0','#12905a','#c47d0c','#c8372f','#6d7690','#7c5ce0'
plt.rcParams.update({'font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'grid.linewidth':.5,'figure.dpi':150,'savefig.bbox':'tight','font.family':'DejaVu Sans','legend.frameon':False})
OUT.mkdir(parents=True, exist_ok=True)
def sv(n): plt.savefig(OUT / f'{n}.pdf'); plt.close()
# ================= P4
fig,ax=plt.subplots(figsize=(6.4,2.4)); mr=R['split']['monthly_rate']; ax.bar(range(1,25),[v*100 for v in mr],color=[B]*14+[A]*4+[G]*6); ax.set_xlabel('Application month'); ax.set_ylabel('Default rate (%)'); sv('p4_split')
fig,ax=plt.subplots(figsize=(3.3,3.1)); ax.plot([a for a,_ in R['roc']['base']],[b for _,b in R['roc']['base']],color=Gy,label=f'Scorecard (AUC {R["metrics"]["base"]["auc"]:.3f})'); ax.plot([a for a,_ in R['roc']['chall']],[b for _,b in R['roc']['chall']],color=B,label=f'Challenger (AUC {R["metrics"]["chall"]["auc"]:.3f})'); ax.plot([0,1],[0,1],'k--',lw=.6); ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); ax.legend(fontsize=6.5,loc='lower right'); sv('p4_roc')
fig,ax=plt.subplots(figsize=(3.3,3.1)); a=R['approval_fine']['a']; ax.plot([x*100 for x in a],[v*100 for v in R['approval_fine']['base']],color=Gy,label='Scorecard'); ax.plot([x*100 for x in a],[v*100 for v in R['approval_fine']['chall']],color=B,label='Challenger'); ax.axhline(R['no_model_default']*100,color=A,ls='--',lw=.8,label='No model'); ax.set_xlabel('Approval rate (%)'); ax.set_ylabel('Default among approved (%)'); ax.legend(fontsize=6.5); sv('p4_app')
sg=sorted(R['segments'],key=lambda r:r['red']); fig,ax=plt.subplots(figsize=(6.4,4)); ax.barh([f'{EN(r["family"])}: {EN(r["level"])}' for r in sg],[r['red']*100 for r in sg],color=B); ax.axvline(R['headline']['mean_reduction_segments']*100,color=A,ls='--',lw=1); ax.set_xlabel('Relative default reduction at 70% approval (%)'); sv('p4_seg')
dc=R['deciles']; fig,ax=plt.subplots(figsize=(3.3,3.1)); xs=range(1,11); ax.plot(xs,[v*100 for v in dc['chall']['obs']],color=A,marker='o',ms=3,label='Observed'); ax.plot(xs,[v*100 for v in dc['chall']['pred']],color=B,marker='o',ms=3,label='Challenger'); ax.plot(xs,[v*100 for v in dc['base']['pred']],color=Gy,marker='o',ms=3,label='Scorecard'); ax.set_xlabel('Risk decile'); ax.set_ylabel('Default rate (%)'); ax.legend(fontsize=6.5); sv('p4_cal')
fig,ax=plt.subplots(figsize=(3.3,3.1)); ab=R['ablation']; ax.barh([EN(x['group']) for x in ab],[x['dauc'] for x in ab],color=Rd); ax.set_xlabel('Change in AUC when the group is removed'); sv('p4_abl')
im=R['importance'][:10][::-1]; fig,ax=plt.subplots(figsize=(6.4,3)); ax.barh([EN(x['f']) for x in im],[x['v'] for x in im],color=B); ax.set_xlabel('Drop in AUC when the feature is permuted'); sv('p4_imp')
fig,ax=plt.subplots(figsize=(3.3,2.8)); ax.bar(range(len(R['bootstrap']['hist'])),R['bootstrap']['hist'],color=B); e=R['bootstrap']['edges']; ax.set_xticks([0,10,19]); ax.set_xticklabels([f'{e[0]*100:.1f}',f'{(e[10]+e[11])/2*100:.1f}',f'{e[19]*100:.1f}']); ax.set_xlabel('Mean reduction (%)'); ax.set_ylabel('Bootstrap samples'); sv('p4_boot')
fig,ax=plt.subplots(figsize=(3.3,2.8)); lc=R['learning']; ax.plot([l['n'] for l in lc],[l['chall'] for l in lc],color=B,marker='o',ms=3,label='Challenger'); ax.plot([l['n'] for l in lc],[l['base'] for l in lc],color=Gy,marker='o',ms=3,label='Scorecard'); ax.set_xlabel('Training applications'); ax.set_ylabel('AUC'); ax.legend(fontsize=6.5); sv('p4_lc')


def main(): print('figures written to', OUT)
if __name__=='__main__': main()
