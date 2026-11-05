"""Default-risk pipeline on SYNTHETIC data: scorecard (logistic regression) vs challenger (gradient boosting).

    python -m risk_model.pipeline [intensity]      # writes results/results.json (takes a few minutes)

This module is a script: importing it runs the whole pipeline. Parameters come from data/params.json.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve, log_loss
from sklearn.inspection import permutation_importance
from risk_model.synth import generate
from risk_model import __version__

ROOT = Path(__file__).resolve().parents[2]
PAR = json.loads((ROOT / 'data' / 'params.json').read_text(encoding='utf-8'))
SEED = PAR['seed']
INTER = float(sys.argv[1]) if len(sys.argv) > 1 else PAR['intensity']
N = PAR['n_applications']
df, rng = generate(INTER, SEED, N)

NUM = ['age', 'income', 'tenure', 'temp_contract', 'amount', 'term', 'dti', 'has_bank', 'bank_inflow', 'bank_overdraft', 'inquiries', 'hist_len',
       'prev_loans', 'hour', 'form_secs', 'email_free', 'lti', 'pti', 'inflow_ratio']
CAT = ['channel', 'device']
LABEL = {'age': 'Edad', 'income': 'Ingresos', 'tenure': 'Antigüedad laboral', 'temp_contract': 'Contrato temporal', 'amount': 'Importe', 'term': 'Plazo', 'dti': 'Deuda/ingresos',
         'has_bank': 'Datos bancarios', 'bank_inflow': 'Ingresos en cuenta', 'bank_overdraft': 'Descubierto', 'inquiries': 'Consultas recientes',
         'hist_len': 'Historial crediticio', 'prev_loans': 'Préstamos previos', 'hour': 'Hora de solicitud', 'form_secs': 'Tiempo en formulario', 'email_free': 'Email gratuito',
         'lti': 'Importe/ingresos', 'pti': 'Cuota/ingresos', 'inflow_ratio': 'Flujo/ingresos', 'channel': 'Canal', 'device': 'Dispositivo'}
GROUPS = {'Perfil': ['age', 'income', 'tenure', 'temp_contract'], 'Deuda y solicitud': ['amount', 'term', 'dti', 'lti', 'pti', 'inquiries', 'hist_len', 'prev_loans'],
          'Datos bancarios': ['has_bank', 'bank_inflow', 'bank_overdraft', 'inflow_ratio'], 'Comportamiento': ['hour', 'form_secs', 'email_free', 'device'], 'Canal': ['channel']}

train = df[df.month < PAR['split']['train_end'] + 1].reset_index(drop=True)
valid = df[(df.month > PAR['split']['train_end']) & (df.month <= PAR['split']['valid_end'])].reset_index(drop=True)
test = df[df.month > PAR['split']['valid_end']].reset_index(drop=True)

def prep_linear(d, ref=None):
    X = d[NUM].copy()
    for c in ['bank_inflow', 'bank_overdraft', 'inflow_ratio']:
        X[c + '_na'] = X[c].isna().astype(int)
        X[c] = X[c].fillna(0 if ref is None else ref[c])
    X['income'] = np.log(X['income']); X['amount'] = np.log(X['amount']); X['lti'] = np.log(X['lti']); X['pti'] = np.log(X['pti'])
    X['tenure'] = np.log1p(X['tenure']); X['hist_len'] = np.log1p(X['hist_len'])
    X = pd.concat([X, pd.get_dummies(d[CAT], drop_first=True).astype(int)], axis=1)
    return X

Xtr_l = prep_linear(train); cols_l = Xtr_l.columns
sc = StandardScaler().fit(Xtr_l)
def lin(d):
    X = prep_linear(d).reindex(columns=cols_l, fill_value=0)
    return sc.transform(X)
base = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(Xtr_l), train.y)

def gbm_X(d):
    X = d[NUM].copy()
    X['channel'] = d.channel.astype('category').cat.codes
    X['device'] = (d.device == 'movil').astype(int)
    return X
cat_map = {c: i for i, c in enumerate(sorted(df.channel.unique()))}
def gbm_X(d):
    X = d[NUM].copy()
    X['channel'] = d.channel.map(cat_map)
    X['device'] = (d.device == 'movil').astype(int)
    return X
gcols = list(gbm_X(train).columns)
cat_mask = [c == 'channel' for c in gcols]
chall = HistGradientBoostingClassifier(learning_rate=.05, max_iter=400, max_leaf_nodes=24, min_samples_leaf=80, l2_regularization=1.0,
                                       categorical_features=cat_mask, early_stopping=True, validation_fraction=.15, n_iter_no_change=25, random_state=SEED)
chall.fit(gbm_X(train), train.y)

def score_base(d): return base.predict_proba(lin(d))[:, 1]
def score_chall(d): return chall.predict_proba(gbm_X(d))[:, 1]
s_b, s_c = score_base(test), score_chall(test)
s_e = 0.35 * s_b + 0.65 * s_c
yt = test.y.values

# ---------------------------------------------------------------- métricas
def ks(y, s):
    fpr, tpr, _ = roc_curve(y, s); return float(np.max(tpr - fpr))
def default_at_approval(y, s, a):
    k = int(round(len(y) * a)); idx = np.argsort(s)[:k]; return float(y[idx].mean())
def metrics(y, s):
    return dict(auc=float(roc_auc_score(y, s)), gini=float(2 * roc_auc_score(y, s) - 1), ks=ks(y, s), brier=float(brier_score_loss(y, s)), logloss=float(log_loss(y, s)))
AP = [0.5, 0.6, 0.7, 0.8, 0.9]
APR = PAR['approval_reference']
res = {'meta': {'n': N, 'default_rate': float(df.y.mean()), 'train_n': len(train), 'valid_n': len(valid), 'test_n': len(test), 'test_default': float(yt.mean()), 'inter': INTER, 'approval_ref': APR}}
res['metrics'] = {'base': metrics(yt, s_b), 'chall': metrics(yt, s_c), 'ens': metrics(yt, s_e)}
res['approval_curve'] = {'a': AP, 'base': [default_at_approval(yt, s_b, a) for a in AP], 'chall': [default_at_approval(yt, s_c, a) for a in AP], 'ens': [default_at_approval(yt, s_e, a) for a in AP]}
grid = np.linspace(0.3, 0.95, 27)
res['approval_fine'] = {'a': grid.tolist(), 'base': [default_at_approval(yt, s_b, a) for a in grid], 'chall': [default_at_approval(yt, s_c, a) for a in grid]}
res['no_model_default'] = float(yt.mean())

# ---- métrica principal: reducción relativa media del default de los aprobados a igual aprobación, promediada por segmentos
def seg_defs(d):
    inc_band = pd.qcut(d.income, 3, labels=['ingresos bajos', 'ingresos medios', 'ingresos altos'])
    amt_band = pd.cut(d.amount, [0, 700, 2000, 1e9], labels=['importe pequeño', 'importe medio', 'importe grande'])
    age_band = pd.cut(d.age, [0, 29, 45, 120], labels=['18–29', '30–45', '46+'])
    return {'Canal': d.channel, 'Ingresos': inc_band, 'Importe': amt_band, 'Edad': age_band, 'Datos bancarios': d.has_bank.map({1: 'con datos bancarios', 0: 'sin datos bancarios'}), 'Dispositivo': d.device}
SEG = seg_defs(test)
def seg_rows(sb, sc_, ap=APR):
    rows = []
    for fam, ser in SEG.items():
        for lvl in ser.unique():
            m = (ser == lvl).values
            if m.sum() < 800: continue
            yy = yt[m]
            db, dc = default_at_approval(yy, sb[m], ap), default_at_approval(yy, sc_[m], ap)
            rows.append(dict(family=fam, level=str(lvl), n=int(m.sum()), base_rate=float(yy.mean()), auc_b=float(roc_auc_score(yy, sb[m])), auc_c=float(roc_auc_score(yy, sc_[m])),
                             dr_b=db, dr_c=dc, red=float((db - dc) / db) if db > 0 else 0.0))
    return rows
rows = seg_rows(s_b, s_c); res['segments'] = rows
res['headline'] = {'mean_reduction_segments': float(np.mean([r['red'] for r in rows])), 'portfolio_reduction': float((default_at_approval(yt, s_b, APR) - default_at_approval(yt, s_c, APR)) / default_at_approval(yt, s_b, APR)),
                   'dr_base': default_at_approval(yt, s_b, APR), 'dr_chall': default_at_approval(yt, s_c, APR), 'ensemble_reduction': float((default_at_approval(yt, s_b, APR) - default_at_approval(yt, s_e, APR)) / default_at_approval(yt, s_b, APR))}
# reducción por tasa de aprobación
res['reduction_by_approval'] = [float((default_at_approval(yt, s_b, a) - default_at_approval(yt, s_c, a)) / default_at_approval(yt, s_b, a)) for a in AP]

# ---- bootstrap del titular (reducción media por segmentos) y de la AUC
rb = np.random.default_rng(SEED + 1); boots = []; dauc = []
n = len(test)
for _ in range(PAR['bootstrap_samples']):
    ix = rb.integers(0, n, n); yy = yt[ix]
    sb, scc = s_b[ix], s_c[ix]
    reds = []
    for fam, ser in SEG.items():
        sv = ser.values[ix]
        for lvl in pd.unique(sv):
            m = sv == lvl
            if m.sum() < 800: continue
            d0, d1 = default_at_approval(yy[m], sb[m], APR), default_at_approval(yy[m], scc[m], APR)
            if d0 > 0: reds.append((d0 - d1) / d0)
    boots.append(np.mean(reds)); dauc.append(roc_auc_score(yy, scc) - roc_auc_score(yy, sb))
res['bootstrap'] = {'mean': float(np.mean(boots)), 'lo': float(np.percentile(boots, 2.5)), 'hi': float(np.percentile(boots, 97.5)), 'dauc_mean': float(np.mean(dauc)), 'dauc_lo': float(np.percentile(dauc, 2.5)), 'dauc_hi': float(np.percentile(dauc, 97.5)),
                    'hist': np.histogram(boots, bins=20)[0].tolist(), 'edges': np.histogram(boots, bins=20)[1].tolist()}

# ---- curvas
fb, tb, _ = roc_curve(yt, s_b); fc, tc, _ = roc_curve(yt, s_c)
ix = np.linspace(0, len(fb) - 1, 80).astype(int); jx = np.linspace(0, len(fc) - 1, 80).astype(int)
res['roc'] = {'base': [[float(fb[i]), float(tb[i])] for i in ix], 'chall': [[float(fc[i]), float(tc[i])] for i in jx]}
def deciles(s):
    d = pd.DataFrame({'s': s, 'y': yt}); d['dec'] = pd.qcut(d.s.rank(method='first'), 10, labels=False) + 1
    g = d.groupby('dec').agg(pred=('s', 'mean'), obs=('y', 'mean'), n=('y', 'size')).reset_index()
    g['capture'] = (d.groupby('dec').y.sum() / d.y.sum()).values
    return g
db_, dc_ = deciles(s_b), deciles(s_c)
res['deciles'] = {'base': db_.to_dict('list'), 'chall': dc_.to_dict('list')}
res['calibration'] = {'base_ece': float(np.mean(np.abs(db_.pred - db_.obs))), 'chall_ece': float(np.mean(np.abs(dc_.pred - dc_.obs)))}
# ks curve del challenger
order = np.argsort(-s_c); cum_bad = np.cumsum(yt[order]) / yt.sum(); cum_good = np.cumsum(1 - yt[order]) / (1 - yt).sum(); q = np.linspace(0, len(order) - 1, 60).astype(int)
res['ks_curve'] = {'x': (q / len(order)).tolist(), 'bad': cum_bad[q].tolist(), 'good': cum_good[q].tolist()}
# gains
gb = np.cumsum(yt[np.argsort(-s_b)]) / yt.sum(); gc = np.cumsum(yt[np.argsort(-s_c)]) / yt.sum()
res['gains'] = {'x': (q / len(order)).tolist(), 'base': gb[q].tolist(), 'chall': gc[q].tolist()}

# ---- importancia por permutación (challenger) sobre muestra del test
sub = test.sample(8000, random_state=1)
pi = permutation_importance(chall, gbm_X(sub), sub.y, scoring='roc_auc', n_repeats=4, random_state=SEED)
imp = sorted([(c, float(m), float(sd)) for c, m, sd in zip(gcols, pi.importances_mean, pi.importances_std)], key=lambda x: -x[1])
res['importance'] = [{'f': LABEL.get(c, c), 'v': m, 'sd': sd} for c, m, sd in imp[:14]]
# importancia absoluta del scorecard (|coef|)
coef = pd.Series(np.abs(base.coef_[0]), index=cols_l).sort_values(ascending=False).head(10)
res['base_coef'] = [{'f': LABEL.get(c.replace('_na', ''), c) + (' (falta)' if c.endswith('_na') else ''), 'v': float(v)} for c, v in coef.items()]

# ---- ablación por grupos (reentrena el challenger sin cada grupo)
def fit_g(cols_drop):
    Xtr = gbm_X(train).drop(columns=[c for c in cols_drop if c in gcols]); cm = [c == 'channel' for c in Xtr.columns]
    m = HistGradientBoostingClassifier(learning_rate=.05, max_iter=300, max_leaf_nodes=24, min_samples_leaf=80, l2_regularization=1.0, categorical_features=cm, early_stopping=True,
                                       validation_fraction=.15, n_iter_no_change=20, random_state=SEED).fit(Xtr, train.y)
    s = m.predict_proba(gbm_X(test).drop(columns=[c for c in cols_drop if c in gcols]))[:, 1]
    return float(roc_auc_score(yt, s)), default_at_approval(yt, s, APR)
full_auc = res['metrics']['chall']['auc']
abl = []
for g, cs_ in GROUPS.items():
    a, dr = fit_g(cs_); abl.append({'group': g, 'auc': a, 'dauc': a - full_auc, 'dr': dr})
res['ablation'] = abl

# ---- curva de aprendizaje
lc = []
for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
    sub_tr = train.sample(frac=frac, random_state=3)
    Xl = prep_linear(sub_tr); sc2 = StandardScaler().fit(Xl); lb = LogisticRegression(C=.5, max_iter=2000).fit(sc2.transform(Xl), sub_tr.y)
    ab = roc_auc_score(yt, lb.predict_proba(sc2.transform(prep_linear(test).reindex(columns=Xl.columns, fill_value=0)))[:, 1])
    mm = HistGradientBoostingClassifier(learning_rate=.05, max_iter=300, max_leaf_nodes=24, min_samples_leaf=80, l2_regularization=1.0, categorical_features=cat_mask,
                                        early_stopping=True, validation_fraction=.15, n_iter_no_change=20, random_state=SEED).fit(gbm_X(sub_tr), sub_tr.y)
    ac = roc_auc_score(yt, mm.predict_proba(gbm_X(test))[:, 1])
    lc.append({'n': len(sub_tr), 'base': float(ab), 'chall': float(ac)})
res['learning'] = lc

# ---- estabilidad temporal: AUC por mes en test + PSI de score
mon = []
for m in sorted(test.month.unique()):
    mk_ = (test.month == m).values
    mon.append({'m': int(m), 'base': float(roc_auc_score(yt[mk_], s_b[mk_])), 'chall': float(roc_auc_score(yt[mk_], s_c[mk_])), 'rate': float(yt[mk_].mean())})
res['monthly'] = mon
def psi(a, b, bins=10):
    qs = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1))); qs[0], qs[-1] = -np.inf, np.inf
    pa = np.histogram(a, qs)[0] / len(a) + 1e-6; pb = np.histogram(b, qs)[0] / len(b) + 1e-6
    return float(np.sum((pa - pb) * np.log(pa / pb)))
tr_c = score_chall(train)
psi_rows = [{'f': 'Score challenger', 'psi': psi(tr_c, s_c)}, {'f': 'Score scorecard', 'psi': psi(score_base(train), s_b)}]
for c in ['dti', 'income', 'amount', 'inquiries', 'age', 'tenure', 'pti', 'form_secs']:
    psi_rows.append({'f': LABEL[c], 'psi': psi(train[c].values.astype(float), test[c].values.astype(float))})
res['psi'] = psi_rows

# ---- auditoría de fuga: añadir una variable posterior al resultado dispara la AUC
leak = test.copy(); leak['post_collections_calls'] = (leak.y * rng.poisson(3, len(leak)) + rng.poisson(.3, len(leak)))
lt = train.copy(); lt['post_collections_calls'] = (lt.y * rng.poisson(3, len(lt)) + rng.poisson(.3, len(lt)))
mleak = HistGradientBoostingClassifier(max_iter=150, random_state=SEED).fit(pd.concat([gbm_X(lt), lt[['post_collections_calls']]], axis=1), lt.y)
auc_leak = float(roc_auc_score(leak.y, mleak.predict_proba(pd.concat([gbm_X(leak), leak[['post_collections_calls']]], axis=1))[:, 1]))
res['leak'] = {'auc_with_leak': auc_leak, 'auc_clean': full_auc}

# ---- datos faltantes estructurales
miss = []
for nm, m in [('sin datos bancarios', test.has_bank == 0), ('con datos bancarios', test.has_bank == 1)]:
    m = m.values; miss.append({'seg': nm, 'n': int(m.sum()), 'rate': float(yt[m].mean()), 'auc_b': float(roc_auc_score(yt[m], s_b[m])), 'auc_c': float(roc_auc_score(yt[m], s_c[m]))})
res['missing'] = miss

# ---- impacto económico (por 100.000 solicitudes, aprobación 70 %)
LGD = PAR['lgd']; MARGIN = PAR['margin']                      # pérdida dada el default y margen sobre el importe
def econ(s, a):
    k = int(len(s) * a); idx = np.argsort(s)[:k]; amt = test.amount.values[idx]; yy = yt[idx]
    profit = float(np.sum(amt * (1 - yy) * MARGIN - amt * yy * LGD)); return profit / len(s) * 100000, float(yy.mean())
res['econ'] = {'lgd': LGD, 'margin': MARGIN, 'base': econ(s_b, APR)[0], 'chall': econ(s_c, APR)[0], 'curve_a': grid.tolist(),
               'curve_base': [econ(s_b, a)[0] for a in grid], 'curve_chall': [econ(s_c, a)[0] for a in grid]}
# a igual default: aprobación adicional con el challenger
target = default_at_approval(yt, s_b, APR)
extra = [a for a in np.linspace(0.6, 0.95, 71) if default_at_approval(yt, s_c, a) <= target]
res['same_default_approval'] = float(max(extra)) if extra else APR

# ---- partición de datos para la diapositiva de diseño
res['split'] = {'train': [0, 13], 'valid': [14, 17], 'test': [18, 23], 'monthly_rate': [float(df[df.month == m].y.mean()) for m in range(24)]}
# distribución del score
bins = np.linspace(0, max(s_c.max(), s_b.max()), 31)
res['score_hist'] = {'edges': bins.tolist(), 'good_c': np.histogram(s_c[yt == 0], bins)[0].tolist(), 'bad_c': np.histogram(s_c[yt == 1], bins)[0].tolist(),
                     'good_b': np.histogram(s_b[yt == 0], bins)[0].tolist(), 'bad_b': np.histogram(s_b[yt == 1], bins)[0].tolist()}
# dependencia parcial de 3 variables (challenger)
pdp = {}
for c, lo, hi in [('dti', 0, 1.0), ('age', 19, 70), ('inquiries', 0, 8)]:
    xs = np.linspace(lo, hi, 18) if c != 'inquiries' else np.arange(0, 9)
    ys = []
    S0 = test.sample(4000, random_state=2)
    for v in xs:
        tmp = S0.copy(); tmp[c] = v
        if c == 'dti': tmp['pti'] = tmp.pti
        ys.append(float(score_chall(tmp).mean()))
    pdp[c] = {'x': [float(v) for v in xs], 'y': ys}
res['pdp'] = pdp
# resumen de datos
res['data_summary'] = [{'f': LABEL[c], 'mean': float(df[c].mean()) if c in df else 0, 'miss': float(df[c].isna().mean())} for c in ['age', 'income', 'tenure', 'amount', 'term', 'dti', 'inquiries', 'bank_inflow', 'bank_overdraft', 'hist_len', 'form_secs']]
res['channel_rate'] = [{'c': k, 'rate': float(v)} for k, v in test.groupby('channel').y.mean().items()]
res['model_version'] = __version__; res['params'] = PAR
(ROOT / 'results').mkdir(exist_ok=True)
json.dump(res, open(ROOT / 'results' / 'results.json', 'w'), ensure_ascii=False)
print(json.dumps({'headline': res['headline'], 'metrics': res['metrics'], 'boot': {k: res['bootstrap'][k] for k in ['mean', 'lo', 'hi', 'dauc_mean']}, 'reduction_by_approval': res['reduction_by_approval'], 'same_default_approval': res['same_default_approval'],
                  'ablation': res['ablation'], 'leak': res['leak'], 'econ': {k: res['econ'][k] for k in ['base', 'chall']}, 'default': res['meta']['default_rate']}, indent=1))
