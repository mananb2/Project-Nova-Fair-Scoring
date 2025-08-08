import numpy as np
import pandas as pd
from pathlib import Path
from faker import Faker

def generate(n=2500, seed=42):
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)

    names = [fake.name() for _ in range(n)]

    # Protected attributes
    gender = rng.choice(['M','F'], size=n, p=[0.7, 0.3])
    age_group = rng.choice(['18-25','26-35','36-45','46-60'], size=n, p=[0.25,0.4,0.25,0.10])
    
    # --- NEW: Generate a numerical age consistent with the age_group ---
    age = []
    for group in age_group:
        if group == '18-25':
            age.append(rng.integers(18, 26))
        elif group == '26-35':
            age.append(rng.integers(26, 36))
        elif group == '36-45':
            age.append(rng.integers(36, 46))
        else: # 46-60
            age.append(rng.integers(46, 61))
    
    city_tier = rng.choice(['A','B','C'], size=n, p=[0.4,0.4,0.2])

    # Partner type
    partner_type = rng.choice(['driver','merchant'], size=n, p=[0.7, 0.3])

    # Activity/quality
    trips_weekly = np.clip(rng.poisson(60, size=n) + (city_tier=='A')*8 - (city_tier=='C')*8, 5, None)
    hours_online_per_week = np.clip(rng.normal(42, 10, size=n), 5, 90)
    avg_rating = np.clip(rng.normal(4.75, 0.15, size=n) - 0.0004*(60 - trips_weekly), 3.0, 5.0)
    on_time_pct = np.clip(rng.normal(0.92, 0.05, size=n) + 0.0006*(trips_weekly - 60), 0.55, 0.995)
    cancel_rate = np.clip(rng.normal(0.07, 0.03, size=n) - 0.0005*(trips_weekly - 60), 0.0, 0.35)
    safety_incidents = rng.poisson(0.02, size=n)

    # Earnings & stability
    weekly_earnings = np.clip(
        rng.normal(360, 85, size=n) + 2.2*trips_weekly + 60*(city_tier=='A') - 50*(city_tier=='C'),
        120, None
    )
    earnings_volatility = np.clip(np.abs(rng.normal(0.18, 0.08, size=n)), 0.02, 0.65)
    cashless_share = np.clip(
        rng.normal(0.72, 0.14, size=n) + 0.06*(city_tier=='A') - 0.06*(city_tier=='C'),
        0.0, 1.0
    )

    # New trend features
    earnings_growth_4w = np.clip(rng.normal(0.05, 0.1, size=n), -0.5, 0.5)
    trips_growth_4w = np.clip(rng.normal(0.03, 0.08, size=n), -0.5, 0.5)

    # Loyalty
    loyalty_years = np.round(np.clip(rng.normal(3, 2, size=n), 0.2, 12), 1)

    # Latent "true" creditworthiness
    latent = (
        0.30 * (weekly_earnings / weekly_earnings.max()) +
        0.20 * on_time_pct +
        0.15 * (avg_rating / 5.0) -
        0.08 * cancel_rate -
        0.07 * earnings_volatility +
        0.05 * earnings_growth_4w +
        0.05 * trips_growth_4w +
        0.05 * (loyalty_years / 12)
    )

    # Bias injection
    unfair_bias = -0.03 * (city_tier == 'C')

    # Final Nova score (300–900) with noise
    nova_score = 300 + 600*np.clip(latent + unfair_bias + rng.normal(0, 0.03, size=n), 0, 1)
    nova_score = np.rint(nova_score).astype(int)

    df = pd.DataFrame({
        'partner_id': np.arange(1, n+1),
        'partner_name': names,
        'gender': gender,
        'age': age,  # Add the new 'age' column here
        'age_group': age_group,
        'city_tier': city_tier,
        'partner_type': partner_type,
        'trips_weekly': trips_weekly,
        'hours_online_per_week': hours_online_per_week,
        'avg_rating': avg_rating,
        'on_time_pct': on_time_pct,
        'cancel_rate': cancel_rate,
        'safety_incidents': safety_incidents,
        'weekly_earnings': weekly_earnings,
        'earnings_volatility': earnings_volatility,
        'cashless_share': cashless_share,
        'earnings_growth_4w': earnings_growth_4w,
        'trips_growth_4w': trips_growth_4w,
        'loyalty_years': loyalty_years,
        'nova_score': nova_score
    })
    return df

if __name__ == "__main__":
    out = Path("data")
    out.mkdir(exist_ok=True)
    df = generate(n=2500, seed=42)
    df.to_csv(out/"partners_synthetic_upgraded.csv", index=False)
    print(f"Saved {out/'partners_synthetic_upgraded.csv'} with shape {df.shape}")