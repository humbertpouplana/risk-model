"""Synthetic loan-application generator (no real data is used anywhere in this project).

`generate(inter)` returns the application table and the numpy Generator in its current state
(the pipeline keeps drawing from it, so results are bit-for-bit reproducible).
"""
import numpy as np
import pandas as pd


def generate(inter: float = 1.0, seed: int = 20261104, n: int = 80000):
    INTER = inter
    N = n
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- generador
    month = rng.integers(0, 24, N)                       # 0..23, últimos 6 meses = test fuera de tiempo
    age = np.clip(rng.normal(37, 11, N), 18, 75)
    income = np.exp(rng.normal(7.35, 0.42, N))           # ingresos mensuales (€)
    tenure = np.clip(rng.gamma(2.0, 28, N), 0, 400)       # antigüedad laboral (meses)
    temp_contract = (rng.random(N) < np.clip(0.45 - tenure / 600, 0.05, 0.6)).astype(int)
    amount = np.clip(np.exp(rng.normal(7.1, 0.7, N)), 100, 8000)
    term = rng.choice([3, 6, 12, 24, 36, 48], N, p=[.1, .15, .3, .2, .15, .1])
    dti = np.clip(rng.beta(2.2, 5.0, N) * 1.1, 0, 1.2)    # deuda/ingresos
    has_bank = (rng.random(N) < 0.72).astype(int)         # datos bancarios enlazados
    bank_inflow = np.where(has_bank == 1, income * np.exp(rng.normal(0, .25, N)), np.nan)
    bank_overdraft = np.where(has_bank == 1, np.clip(rng.gamma(1.2, 0.8, N) * (dti + .2), 0, 8), np.nan)
    inquiries = rng.poisson(np.clip(1.4 + 2.2 * dti, .3, 5), N)
    hist_len = np.clip(rng.gamma(2.5, 35, N) * (age / 40), 0, 400)   # antigüedad histórico crediticio (meses)
    prev_loans = rng.poisson(np.clip(hist_len / 60, 0.2, 5), N)
    channel = rng.choice(['seo', 'sem_marca', 'sem_generico', 'afiliados', 'directo'], N, p=[.2, .15, .3, .2, .15])
    device = rng.choice(['movil', 'escritorio'], N, p=[.78, .22])
    hour = rng.integers(0, 24, N)
    form_secs = np.clip(rng.lognormal(5.0, .5, N), 20, 1200)         # tiempo en rellenar el formulario
    email_free = (rng.random(N) < .8).astype(int)

    ch_eff = pd.Series(channel).map({'seo': -.10, 'sem_marca': -.2, 'sem_generico': .05, 'afiliados': .30, 'directo': -.05}).values
    lti = amount / income                                  # cuota-importe sobre ingresos
    pti = (amount / term) / income                         # peso de la cuota mensual
    z = (-2.45
         + 1.35 * dti
         + 0.35 * np.log1p(inquiries)
         - 0.20 * np.log1p(tenure) / 2
         + 0.25 * temp_contract
         - 0.00 * age
         + ch_eff
         + 0.15 * (device == 'movil')
         + 0.10 * email_free)
    # no linealidades e interacciones (intensidad INTER)
    nl = (0.9 * (pti > 0.12) * (dti > 0.35)                        # cuota alta + deuda alta
          + 0.9 * ((age < 26) & (inquiries >= 3))                   # joven y muy consultado
          + 0.8 * (temp_contract * (tenure < 6))                    # contrato temporal reciente
          + 0.7 * ((hour >= 1) & (hour <= 5))                       # solicitud de madrugada
          + 0.6 * ((has_bank == 1) & (np.nan_to_num(bank_overdraft) > 3.2))   # descubierto fuerte
          - 0.5 * ((has_bank == 1) & (np.nan_to_num(bank_inflow) > income * 1.05))  # flujo bancario sano
          + 0.6 * ((hist_len < 12) & (amount > 2500))               # historial corto, importe alto
          + 0.6 * ((form_secs < 55) | (form_secs > 600))            # formulario sospechoso
          - 0.5 * ((prev_loans >= 2) & (dti < .25)))                # recurrente solvente
    z = z + INTER * nl
    z = z + 0.15 * (has_bank == 0)                                # sin datos bancarios, algo más de riesgo
    z = z + 0.12 * np.where(month >= 18, 1, 0) * (dti > .3)       # deriva macro en el tramo final
    z = z + rng.normal(0, 0.55, N)                                # riesgo no observable
    p = 1 / (1 + np.exp(-(z - 0.55)))
    y = (rng.random(N) < p).astype(int)

    df = pd.DataFrame(dict(month=month, age=age, income=income, tenure=tenure, temp_contract=temp_contract, amount=amount, term=term, dti=dti,
                           has_bank=has_bank, bank_inflow=bank_inflow, bank_overdraft=bank_overdraft, inquiries=inquiries, hist_len=hist_len,
                           prev_loans=prev_loans, channel=channel, device=device, hour=hour, form_secs=form_secs, email_free=email_free, y=y))
    df['lti'] = df.amount / df.income
    df['pti'] = (df.amount / df.term) / df.income
    df['inflow_ratio'] = df.bank_inflow / df.income
    return df, rng


def main(argv=None):
    """python -m risk_model.synth [--full]   ->  data/sample_applications.csv (2,000 rows) or data/applications.csv (all rows, git-ignored)"""
    import json, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    par = json.loads((root / "data" / "params.json").read_text(encoding="utf-8"))
    df, _ = generate(par["intensity"], par["seed"], par["n_applications"])
    full = "--full" in (argv or sys.argv[1:])
    out = root / "data" / ("applications.csv" if full else "sample_applications.csv")
    (df if full else df.sample(2000, random_state=1).sort_values("month")).to_csv(out, index=False)
    print("written", out, "rows:", len(df) if full else 2000)


if __name__ == "__main__":
    main()
