"""Simulations: ^NDX vs TQQQ (DCA, mixed strategies, and TQQQ simulated from ^NDX).

Produces PNGs and a CSV of results. Plots are saved but not shown interactively.

Usage: python sim.py
"""
from datetime import datetime
import sys

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except Exception:
    print("Missing required packages. Install with: pip install yfinance pandas matplotlib numpy pandas")
    raise


# When False, plt.show() is skipped so only files are saved
SHOW_PLOTS = False


def fetch_close(ticker: str) -> pd.Series:
    data = yf.download(ticker, period="max", progress=False, auto_adjust=True)
    if data is None or data.empty:
        raise RuntimeError(f"No data for {ticker}")
    if isinstance(data, pd.Series):
        s = data.dropna()
    elif "Close" in data.columns:
        s = data["Close"].dropna()
    elif "Adj Close" in data.columns:
        s = data["Adj Close"].dropna()
    else:
        # take first numeric column
        s = data.select_dtypes(include=["number"]).iloc[:, 0].dropna()
    s.name = ticker
    return s


def anchor_investment(price_series: pd.Series, investment: float = 1000.0) -> pd.Series:
    if isinstance(price_series, pd.DataFrame):
        price_series = price_series.iloc[:, 0]
    price_series = price_series.dropna()
    first_price = float(price_series.iloc[0])
    shares = investment / first_price
    return shares * price_series


def simulate_dca(price_series: pd.Series, contribution: float = 1000.0):
    if isinstance(price_series, pd.DataFrame):
        price_series = price_series.iloc[:, 0]
    idx = price_series.index
    month_starts = pd.date_range(start=idx[0].to_period("M").to_timestamp(), end=idx[-1], freq="MS")
    contrib_dates = []
    for ms in month_starts:
        cand = idx[idx >= ms]
        if not cand.empty:
            contrib_dates.append(cand[0])
    contrib_set = set(contrib_dates)

    shares = 0.0
    invested = 0.0
    portfolio = []
    invested_t = []
    for dt, p in price_series.items():
        if dt in contrib_set:
            shares += contribution / float(p)
            invested += contribution
        portfolio.append(shares * float(p))
        invested_t.append(invested)

    return pd.Series(portfolio, index=idx, name=price_series.name), pd.Series(invested_t, index=idx, name="Invested")


def simulate_mixed_dca(price_series_map: dict, contribution: float = 1000.0, target_weights: dict = None,
                       lower_thresh: float = 0.25, upper_thresh: float = 0.375):
    tickers = list(price_series_map.keys())
    if target_weights is None:
        target_weights = {t: 1.0 / len(tickers) for t in tickers}

    # Intersection of indices
    idx = price_series_map[tickers[0]].index
    for t in tickers[1:]:
        idx = idx.intersection(price_series_map[t].index)
    if idx.empty:
        raise RuntimeError("No common trading dates")

    aligned = {t: price_series_map[t].reindex(idx).ffill() for t in tickers}
    month_starts = pd.date_range(start=idx[0].to_period("M").to_timestamp(), end=idx[-1], freq="MS")
    contrib_dates = []
    for ms in month_starts:
        cand = idx[idx >= ms]
        if not cand.empty:
            contrib_dates.append(cand[0])
    contrib_set = set(contrib_dates)

    shares = {t: 0.0 for t in tickers}
    invested = 0.0
    portfolio_vals = []
    invested_vals = []

    def get_price(s, i):
        v = s.iloc[i]
        if isinstance(v, (pd.Series, pd.DataFrame)):
            v = v.iloc[0]
        return float(v)

    for i, dt in enumerate(idx):
        prices = {t: get_price(aligned[t], i) for t in tickers}
        if dt in contrib_set:
            for t in tickers:
                amt = contribution * float(target_weights.get(t, 0))
                if prices[t] <= 0:
                    continue
                shares[t] += amt / prices[t]
            invested += contribution

        vals = {t: shares[t] * prices[t] for t in tickers}
        total = sum(vals.values())
        portfolio_vals.append(total)
        invested_vals.append(invested)

        if total > 0:
            first_t = tickers[0]
            first_weight = vals[first_t] / total
            if (first_weight < lower_thresh) or (first_weight > upper_thresh):
                for t in tickers:
                    target_val = total * float(target_weights.get(t, 0))
                    shares[t] = target_val / prices[t] if prices[t] > 0 else 0.0

    return pd.Series(portfolio_vals, index=idx, name="Mixed"), pd.Series(invested_vals, index=idx, name="Invested")


def simulate_all_allocations_heatmap(price_series_map: dict, contribution: float = 1000.0):
    """Run allocations from 1% to 99% TQQQ (rest ^NDX) with monthly DCA and monthly rebalance.

    Returns a DataFrame with columns: pct_TQQQ, final_value, invested, return_pct
    """
    tickers = list(price_series_map.keys())
    if set(tickers) != set(["^NDX", "TQQQ"]):
        raise RuntimeError("price_series_map must contain '^NDX' and 'TQQQ'")

    # align indices
    idx = price_series_map["^NDX"].index
    idx = idx.intersection(price_series_map["TQQQ"].index)
    aligned = {t: price_series_map[t].reindex(idx).ffill() for t in ["^NDX", "TQQQ"]}

    # monthly contribution dates
    month_starts = pd.date_range(start=idx[0].to_period("M").to_timestamp(), end=idx[-1], freq="MS")
    contrib_dates = []
    for ms in month_starts:
        cand = idx[idx >= ms]
        if not cand.empty:
            contrib_dates.append(cand[0])
    contrib_set = set(contrib_dates)

    results = []
    for w in range(1, 100):
        tqqq_w = w / 100.0
        ndx_w = 1.0 - tqqq_w
        target = {"TQQQ": tqqq_w, "^NDX": ndx_w}

        shares = {"TQQQ": 0.0, "^NDX": 0.0}
        invested = 0.0
        portfolio_vals = []

        def get_price(s, i):
            v = s.iloc[i]
            if isinstance(v, (pd.Series, pd.DataFrame)):
                v = v.iloc[0]
            return float(v)

        for i, dt in enumerate(idx):
            prices = {t: get_price(aligned[t], i) for t in ["^NDX", "TQQQ"]}
            if dt in contrib_set:
                # add contributions proportionally
                for t in ["^NDX", "TQQQ"]:
                    amt = contribution * float(target[t])
                    if prices[t] <= 0:
                        continue
                    shares[t] += amt / prices[t]
                invested += contribution
                # rebalance to target weights after contribution
                vals = {t: shares[t] * prices[t] for t in ["^NDX", "TQQQ"]}
                total = sum(vals.values())
                if total > 0:
                    for t in ["^NDX", "TQQQ"]:
                        shares[t] = (total * float(target[t])) / prices[t] if prices[t] > 0 else 0.0

            vals_now = {t: shares[t] * prices[t] for t in ["^NDX", "TQQQ"]}
            total_now = sum(vals_now.values())
            portfolio_vals.append(total_now)

        final_value = portfolio_vals[-1]
        return_pct = (final_value / invested - 1.0) if invested > 0 else np.nan
        results.append({"pct_TQQQ": w, "final_value": final_value, "invested": invested, "return_pct": return_pct})

    df = pd.DataFrame(results).set_index("pct_TQQQ")
    # save CSV
    df.to_csv("allocation_returns.csv")

    # heatmap (1 x N) show percent returns
    arr = df["return_pct"].values.reshape(1, -1)
    plt.figure(figsize=(12, 2.5))
    im = plt.imshow(arr, aspect='auto', cmap='viridis', extent=[1, 99, 0, 1])
    plt.colorbar(im, orientation='vertical', label='Return (decimal)')
    plt.yticks([])
    plt.xlabel('TQQQ allocation (%)')
    plt.title('Heatmap of cumulative return vs TQQQ allocation (monthly rebalance)')
    plt.tight_layout()
    plt.savefig('allocation_heatmap.png', dpi=150)

    return df


def simulate_monthly_rebalance(price_series_map: dict, target_weights: dict, contribution: float = 1000.0):
    """Monthly DCA with monthly rebalance to target_weights.

    target_weights: dict with keys '^NDX' and 'TQQQ' summing to 1.0
    """
    tickers = ["^NDX", "TQQQ"]
    if set(price_series_map.keys()) >= set(tickers) is False:
        # allow extra keys but require these two
        pass

    idx = price_series_map[tickers[0]].index
    idx = idx.intersection(price_series_map[tickers[1]].index)
    aligned = {t: price_series_map[t].reindex(idx).ffill() for t in tickers}

    month_starts = pd.date_range(start=idx[0].to_period("M").to_timestamp(), end=idx[-1], freq="MS")
    contrib_dates = []
    for ms in month_starts:
        cand = idx[idx >= ms]
        if not cand.empty:
            contrib_dates.append(cand[0])
    contrib_set = set(contrib_dates)

    shares = {t: 0.0 for t in tickers}
    invested = 0.0
    portfolio_vals = []

    def get_price(s, i):
        v = s.iloc[i]
        if isinstance(v, (pd.Series, pd.DataFrame)):
            v = v.iloc[0]
        return float(v)

    for i, dt in enumerate(idx):
        prices = {t: get_price(aligned[t], i) for t in tickers}
        if dt in contrib_set:
            for t in tickers:
                amt = contribution * float(target_weights.get(t, 0))
                if prices[t] <= 0:
                    continue
                shares[t] += amt / prices[t]
            invested += contribution
            # monthly rebalance to exact target weights
            vals = {t: shares[t] * prices[t] for t in tickers}
            total = sum(vals.values())
            if total > 0:
                for t in tickers:
                    shares[t] = (total * float(target_weights.get(t, 0))) / prices[t] if prices[t] > 0 else 0.0

        vals_now = {t: shares[t] * prices[t] for t in tickers}
        total_now = sum(vals_now.values())
        portfolio_vals.append(total_now)

    return pd.Series(portfolio_vals, index=idx, name="MonthlyRebal"), pd.Series([invested] * len(idx), index=idx, name="Invested")


def compute_metrics(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"total_return": np.nan, "CAGR": np.nan, "annual_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    returns = s.pct_change().dropna()
    days = len(s)
    years = days / 252.0 if days > 0 else np.nan
    total_return = float(s.iloc[-1] / s.iloc[0] - 1.0)
    CAGR = float((s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    annual_return = float(returns.mean() * 252.0) if not returns.empty else np.nan
    annual_vol = float(returns.std() * np.sqrt(252.0)) if not returns.empty else np.nan
    sharpe = float(annual_return / annual_vol) if (annual_vol and annual_vol > 0) else np.nan
    dd = (s / s.cummax() - 1.0)
    max_dd = float(dd.min())
    return {"total_return": total_return, "CAGR": CAGR, "annual_return": annual_return,
            "annual_vol": annual_vol, "sharpe": sharpe, "max_drawdown": max_dd}


def simulate_leveraged_from_underlying(underlying: pd.Series, leverage: float = 3.0, start_price: float = None,
                                      annual_fee: float = 0.0095) -> pd.Series:
    if isinstance(underlying, pd.DataFrame):
        s = underlying.iloc[:, 0].copy()
    else:
        s = underlying.copy()
    returns = s.pct_change().fillna(0)
    if start_price is None:
        sim = [float(s.iloc[0])]
    else:
        sim = [float(start_price)]
    daily_fee = float(annual_fee) / 252.0 if annual_fee is not None else 0.0
    for r in returns.iloc[1:]:
        prev = sim[-1]
        new = prev * (1.0 + leverage * float(r) - daily_fee)
        sim.append(new)
    return pd.Series(sim, index=s.index, name=f"SimLeverage{leverage}")


def calibrate_annual_fee(underlying: pd.Series, real_leveraged: pd.Series, leverage: float = 3.0,
                         search_min: float = 0.0, search_max: float = 0.05, steps: int = 101):
    common = underlying.index.intersection(real_leveraged.index)
    u = underlying.reindex(common).ffill()
    r = real_leveraged.reindex(common).ffill()
    if isinstance(u, pd.DataFrame):
        u = u.iloc[:, 0]
    if isinstance(r, pd.DataFrame):
        r = r.iloc[:, 0]

    fees = np.linspace(search_min, search_max, steps)
    best_fee = fees[0]
    best_err = float('inf')
    start_price = float(r.iloc[0])
    for f in fees:
        sim = simulate_leveraged_from_underlying(u, leverage=leverage, start_price=start_price, annual_fee=f)
        sim = sim.reindex(common).ffill()
        mask = (sim > 0) & (r > 0)
        if not mask.any():
            continue
        err = np.mean((np.log(sim[mask]) - np.log(r[mask])) ** 2)
        if err < best_err:
            best_err = err
            best_fee = f
    return best_fee


def main():
    tickers = ["^NDX", "TQQQ"]
    investment = 1000.0

    print("Fetching data for:", ", ".join(tickers))
    raw = {t: fetch_close(t) for t in tickers}
    for t in tickers:
        print(f"{t}: {raw[t].index[0].date()} -> {raw[t].index[-1].date()} ({len(raw[t])} rows)")

    print("Calibrating annual fee to match real TQQQ...")
    best_fee = calibrate_annual_fee(raw["^NDX"], raw["TQQQ"], leverage=3.0, search_min=0.0, search_max=0.05, steps=201)
    print(f"Calibrated annual fee: {best_fee:.6f} ({best_fee*100:.3f}% annual)")

    # simulate full-range TQQQ from ^NDX and scale to match real at TQQQ inception
    sim_full = simulate_leveraged_from_underlying(raw["^NDX"], leverage=3.0, start_price=1.0, annual_fee=best_fee)
    t0 = raw["TQQQ"].index[0]
    if t0 in sim_full.index:
        sim_at_t0 = float(sim_full.loc[t0])
    else:
        pos = sim_full.index.get_indexer([t0], method='nearest')[0]
        sim_at_t0 = float(sim_full.iloc[pos])
    real_at_t0 = float(raw["TQQQ"].loc[t0].squeeze())
    scale = real_at_t0 / sim_at_t0 if sim_at_t0 != 0 else 1.0
    sim_full_scaled = sim_full * scale

    # combined: simulated before t0, real from t0 on
    combined_tqqq = sim_full_scaled.copy()
    real_part = raw["TQQQ"].reindex(combined_tqqq.index).ffill()
    if isinstance(real_part, pd.DataFrame):
        real_part = real_part.iloc[:, 0]
    combined_tqqq.loc[t0:] = real_part.loc[t0:]

    # full price map (use ^NDX instead of QQQ)
    full_map = {"^NDX": raw["^NDX"], "TQQQ": combined_tqqq}

    # DCA over full history
    dca = {}
    invested_map = {}
    for t in tickers:
        p, inv = simulate_dca(full_map[t], contribution=investment)
        dca[t] = p
        invested_map[t] = inv
    # use invested series (same across tickers) for plotting
    invested = invested_map[tickers[0]]

    # non-rebalancing allocated strategy (1/3 - 2/3)
    no_rebal_w = {"^NDX": 0.3333333, "TQQQ": 0.6666667}
    no_rebal, _ = simulate_mixed_dca(full_map, contribution=investment, target_weights=no_rebal_w, lower_thresh=0.0, upper_thresh=1.0)

    # monthly rebalance to 33.3/66.7
    monthly_rebal_target = {"^NDX": 0.3333333, "TQQQ": 0.6666667}
    monthly_rebal, monthly_invested = simulate_monthly_rebalance(full_map, target_weights=monthly_rebal_target, contribution=investment)

    # Save TQQQ sim vs real plots
    plt.figure(figsize=(12, 6))
    plt.plot(raw["TQQQ"].index, raw["TQQQ"].values, label="TQQQ (real)")
    plt.plot(sim_full_scaled.index, sim_full_scaled.values, label="TQQQ (sim from ^NDX, 3x)")
    plt.title("Real TQQQ vs Simulated TQQQ (3x ^NDX returns)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("tqqq_sim_vs_real.png", dpi=150)
    print("Saved tqqq_sim_vs_real.png")

    plt.figure(figsize=(12, 6))
    plt.plot(raw["TQQQ"].index, raw["TQQQ"].values, label="TQQQ (real)")
    plt.plot(sim_full_scaled.index, sim_full_scaled.values, label="TQQQ (sim from ^NDX, 3x)")
    plt.yscale("log")
    plt.title("Real TQQQ vs Simulated TQQQ (log)")
    plt.xlabel("Date")
    plt.ylabel("Price (log)")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("tqqq_sim_vs_real_logy.png", dpi=150)
    print("Saved tqqq_sim_vs_real_logy.png")

    # Investment comparison plots (linear and log)
    plt.figure(figsize=(12, 6))
    for t in tickers:
        plt.plot(dca[t].index, dca[t].values, label=t)
    plt.plot(no_rebal.index, no_rebal.values, label="NoRebal (TQQQ66/^NDX33)")
    plt.plot(monthly_rebal.index, monthly_rebal.values, label="MonthlyRebal (33.3/66.7)", color="#ff69b4")
    plt.plot(invested.index, invested.values, label="Invested", color="k", linestyle="--")
    plt.title(f"Monthly ${int(investment)} DCA since ^NDX start")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("investment_comparison_dca.png", dpi=150)
    print("Saved investment_comparison_dca.png")

    plt.figure(figsize=(12, 6))
    for t in tickers:
        plt.plot(dca[t].index, dca[t].values, label=t)
    plt.plot(no_rebal.index, no_rebal.values, label="NoRebal (TQQQ66/^NDX33)")
    plt.plot(monthly_rebal.index, monthly_rebal.values, label="MonthlyRebal (33.3/66.7)", color="#ff69b4")
    plt.plot(invested.index, invested.values, label="Invested", color="k", linestyle="--")
    plt.yscale("log")
    plt.title(f"Monthly ${int(investment)} DCA since ^NDX start (log)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value (USD, log)")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("investment_comparison_dca_logy.png", dpi=150)
    print("Saved investment_comparison_dca_logy.png")

    # Anchored comparison ($1000 initial) since ^NDX start
    anchored_ndx = anchor_investment(full_map["^NDX"], investment)
    anchored_tqqq = anchor_investment(full_map["TQQQ"], investment)

    plt.figure(figsize=(12, 6))
    plt.plot(anchored_ndx.index, anchored_ndx.values, label="^NDX (anchored $1000)")
    plt.plot(anchored_tqqq.index, anchored_tqqq.values, label="TQQQ combined (anchored $1000)")
    plt.title(f"$ {int(investment)} anchored: ^NDX vs TQQQ (sim+real) since ^NDX start")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("qqq_vs_tqqq_anchored.png", dpi=150)
    print("Saved qqq_vs_tqqq_anchored.png")

    plt.figure(figsize=(12, 6))
    plt.plot(anchored_ndx.index, anchored_ndx.values, label="^NDX (anchored $1000)")
    plt.plot(anchored_tqqq.index, anchored_tqqq.values, label="TQQQ combined (anchored $1000)")
    plt.yscale("log")
    plt.title(f"$ {int(investment)} anchored: ^NDX vs TQQQ (sim+real) since ^NDX start - log y")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value (USD, log)")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("qqq_vs_tqqq_anchored_logy.png", dpi=150)
    print("Saved qqq_vs_tqqq_anchored_logy.png")

    # Export results to CSV
    df_index = full_map["^NDX"].index
    df = pd.DataFrame(index=df_index)
    for t in tickers:
        df[f"{t}_DCA"] = dca[t].reindex(df_index)
    df["Invested"] = invested.reindex(df_index)
    df["NoRebalance"] = no_rebal.reindex(df_index)
    df["MonthlyRebal"] = monthly_rebal.reindex(df_index)
    df["^NDX"] = raw["^NDX"].reindex(df_index)
    df["TQQQ_real"] = raw["TQQQ"].reindex(df_index)
    df["TQQQ_sim"] = sim_full_scaled.reindex(df_index)
    df["TQQQ_combined"] = combined_tqqq.reindex(df_index)
    df["^NDX_anchored"] = anchored_ndx.reindex(df_index)
    df["TQQQ_combined_anchored"] = anchored_tqqq.reindex(df_index)

    df.to_csv("simulation_results.csv", index_label="Date")
    print("Saved simulation_results.csv")

    # compute industry-standard metrics for comparison
    metrics = {}
    metrics["^NDX"] = compute_metrics(dca["^NDX"]) if "^NDX" in dca else {}
    metrics["TQQQ"] = compute_metrics(dca["TQQQ"]) if "TQQQ" in dca else {}
    metrics["RebalanceMix"] = compute_metrics(monthly_rebal)
    metrics["NoRebalanceMix"] = compute_metrics(no_rebal)
    met_df = pd.DataFrame(metrics).T
    met_df.to_csv("strategy_metrics.csv")
    print("Saved strategy_metrics.csv")

    # run allocations heatmap and save
    try:
        print("Computing allocation heatmap (1%-99% TQQQ)...")
        alloc_df = simulate_all_allocations_heatmap(full_map, contribution=investment)
        print("Saved allocation_returns.csv and allocation_heatmap.png")
    except Exception as e:
        print("Allocation heatmap failed:", e)


if __name__ == "__main__":
    main()
