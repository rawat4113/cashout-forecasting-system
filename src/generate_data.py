"""
Synthetic NCRP-style data generator.

Real complaint data is not public, so this module simulates the process that
creates it. The simulator is deliberately *not* a trivial lookup table:

* ~145 ATM clusters ("cells") spread over Indian cities of three tiers.
* Each cell has a latent two-state "mule hot / not hot" process that
  - switches on with some probability (higher if neighbours in the same city are hot),
  - lasts ~10 days on average,
  - migrates to another cell in the same city after police pressure.
* Hot cells produce ~10x more cash-outs. Weekly + month-end seasonality and a growth trend are added.
* A cash-out is only *visible* to analysts when its complaint reaches the portal
  (exponential delay, mean ~20 h) and ~15% are never traced at all.

The label we later predict is "will a cash-out happen in cell c on day d".
The features only use complaints that had ALREADY arrived before day d starts.

Run:  python -m src.generate_data
"""
import numpy as np
import pandas as pd

from . import config as C

CITIES = [
    # name, state, lat, lon, tier
    ("Mumbai", "Maharashtra", 19.076, 72.878, 1),
    ("Delhi", "Delhi", 28.614, 77.209, 1),
    ("Bengaluru", "Karnataka", 12.972, 77.594, 1),
    ("Hyderabad", "Telangana", 17.385, 78.487, 1),
    ("Kolkata", "West Bengal", 22.573, 88.364, 1),
    ("Chennai", "Tamil Nadu", 13.083, 80.270, 1),
    ("Pune", "Maharashtra", 18.520, 73.857, 1),
    ("Ahmedabad", "Gujarat", 23.023, 72.572, 1),
    ("Jaipur", "Rajasthan", 26.912, 75.787, 2),
    ("Lucknow", "Uttar Pradesh", 26.847, 80.947, 2),
    ("Surat", "Gujarat", 21.170, 72.831, 2),
    ("Nagpur", "Maharashtra", 21.146, 79.088, 2),
    ("Patna", "Bihar", 25.594, 85.138, 2),
    ("Bhopal", "Madhya Pradesh", 23.259, 77.413, 2),
    ("Indore", "Madhya Pradesh", 22.720, 75.858, 2),
    ("Vadodara", "Gujarat", 22.307, 73.181, 2),
    ("Gurugram", "Haryana", 28.459, 77.027, 2),
    ("Noida", "Uttar Pradesh", 28.535, 77.391, 2),
    ("Kochi", "Kerala", 9.931, 76.267, 2),
    ("Ranchi", "Jharkhand", 23.344, 85.310, 2),
    ("Nashik", "Maharashtra", 19.998, 73.790, 3),
    ("Rajkot", "Gujarat", 22.304, 70.802, 3),
    ("Guwahati", "Assam", 26.144, 91.736, 3),
    ("Bhubaneswar", "Odisha", 20.296, 85.825, 3),
    ("Coimbatore", "Tamil Nadu", 11.017, 76.956, 3),
    ("Visakhapatnam", "Andhra Pradesh", 17.686, 83.218, 3),
    ("Chandigarh", "Chandigarh", 30.733, 76.779, 3),
]
CLUSTERS_PER_TIER = {1: 8, 2: 5, 3: 3}
ATMS_PER_TIER = {1: 60, 2: 35, 3: 18}

CATEGORIES = [
    # name, probability, amount multiplier
    ("UPI / Card Fraud", 0.30, 0.6),
    ("KYC / OTP Fraud", 0.20, 0.5),
    ("Investment / Trading Scam", 0.18, 3.0),
    ("Job / Task Fraud", 0.14, 1.0),
    ("Sextortion", 0.10, 0.8),
    ("Digital Arrest Scam", 0.08, 6.0),
]


def make_cells(rng: np.random.Generator) -> pd.DataFrame:
    rows, cid = [], 0
    for name, state, lat, lon, tier in CITIES:
        for _ in range(CLUSTERS_PER_TIER[tier]):
            jitter = rng.normal(0, 0.045, 2)  # roughly 5 km spread inside a city
            rows.append(dict(
                cell_id=cid, city=name, state=state,
                lat=round(lat + jitter[0], 5), lon=round(lon + jitter[1], 5),
                tier=tier,
                atm_count=int(rng.poisson(ATMS_PER_TIER[tier])) + 5,
            ))
            cid += 1
    return pd.DataFrame(rows)


def simulate_hot_state(cells: pd.DataFrame, rng: np.random.Generator, n_days: int):
    """Latent hot/not-hot Markov process per cell with in-city spill-over and migration."""
    n_cells = len(cells)
    city_codes = cells["city"].astype("category").cat.codes.to_numpy()
    n_city = city_codes.max() + 1
    city_size = np.bincount(city_codes, minlength=n_city)

    p_on, p_off, spill, p_migrate = 0.008, 0.10, 0.03, 0.40
    state = rng.random(n_cells) < 0.06
    hot = np.zeros((n_cells, n_days), dtype=bool)

    for d in range(n_days):
        frac_hot = np.bincount(city_codes, weights=state, minlength=n_city)[city_codes] / city_size[city_codes]
        turn_on = rng.random(n_cells) < (p_on + spill * frac_hot)
        turn_off = rng.random(n_cells) < p_off
        new_state = np.where(state, ~turn_off, turn_on)

        ended = state & ~new_state
        for c in np.flatnonzero(ended):          # mules move after police pressure
            if rng.random() < p_migrate:
                same_city = np.flatnonzero((city_codes == city_codes[c]) & (np.arange(n_cells) != c))
                if len(same_city):
                    new_state[rng.choice(same_city)] = True
        state = new_state
        hot[:, d] = state
    return hot


def generate(seed: int = C.SEED):
    rng = np.random.default_rng(seed)
    cells = make_cells(rng)
    n_cells, n_days = len(cells), C.N_DAYS
    start = pd.Timestamp(C.START_DATE)

    hot = simulate_hot_state(cells, rng, n_days)

    base = cells["atm_count"].to_numpy() * rng.lognormal(0, 0.6, n_cells)
    base = base / base.mean()
    hot_mult = rng.lognormal(np.log(10), 0.3, n_cells)

    dates = pd.date_range(start, periods=n_days)
    dow_factor = np.where(dates.dayofweek >= 4, 1.2, 1.0)                       # Fri-Sun busier
    month_end = ((dates.day >= 28) | (dates.day <= 3)).astype(float)
    season = dow_factor * (1 + 0.3 * month_end)
    trend = 1 + 0.0012 * np.arange(n_days)                                       # rising cybercrime

    lam = base[:, None] * np.where(hot, hot_mult[:, None], 1.0) * season[None, :] * trend[None, :]
    lam *= C.TARGET_EVENTS_PER_DAY / lam.sum(axis=0).mean()

    counts = rng.poisson(lam)
    c_idx, d_idx = np.nonzero(counts)
    reps = counts[c_idx, d_idx]
    cell_arr = np.repeat(c_idx, reps)
    day_arr = np.repeat(d_idx, reps)
    n_ev = len(cell_arr)

    hour_w = np.array([1.6] * 6 + [0.7] * 4 + [1.0] * 8 + [1.3] * 6)            # night-heavy cash-outs
    hour = rng.choice(24, size=n_ev, p=hour_w / hour_w.sum())
    offset = hour * 3600 + rng.integers(0, 3600, n_ev)
    w_time = start + pd.to_timedelta(day_arr, unit="D") + pd.to_timedelta(offset, unit="s")

    cat_names = [c[0] for c in CATEGORIES]
    cat_p = np.array([c[1] for c in CATEGORIES])
    cat_mult = np.array([c[2] for c in CATEGORIES])
    cat_i = rng.choice(len(CATEGORIES), size=n_ev, p=cat_p / cat_p.sum())
    amount = np.clip(rng.lognormal(np.log(18000), 0.9, n_ev) * cat_mult[cat_i], 500, 1_000_000).round(0)

    delay_h = 1 + rng.exponential(C.MEAN_COMPLAINT_DELAY_H - 1, n_ev)
    c_time = w_time + pd.to_timedelta(delay_h, unit="h")
    traced = rng.random(n_ev) > C.UNTRACED_FRACTION

    events = pd.DataFrame({
        "withdrawal_time": w_time,
        "complaint_time": c_time,
        "cell_id": cell_arr,
        "category": np.array(cat_names)[cat_i],
        "amount": amount,
        "traced": traced,
    }).sort_values("withdrawal_time").reset_index(drop=True)
    events.insert(0, "event_id", np.arange(len(events)))
    return events, cells


def main():
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    events, cells = generate()
    events.to_csv(C.DATA_DIR / "events.csv", index=False)
    cells.to_csv(C.DATA_DIR / "cells.csv", index=False)
    print(f"cells: {len(cells)} | events: {len(events):,} | "
          f"events/day: {len(events) / C.N_DAYS:.1f} | traced: {events['traced'].mean():.0%}")
    print(f"saved -> {C.DATA_DIR}")


if __name__ == "__main__":
    main()
