"""
Easy Transport – Synthetic Survey Dataset Generator
Generates 400 realistic, internally consistent student responses.
Run:  python generate_dataset.py
Output: easy_transport_survey.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 400

# ── Helper ────────────────────────────────────────────────────────────────────
def choice(options, p=None, n=N):
    return np.random.choice(options, size=n, p=p)


# ── SECTION A: Demographics ───────────────────────────────────────────────────
age   = np.random.randint(18, 31, N)
gender = choice(["Male", "Female", "Prefer not to say"], p=[0.48, 0.50, 0.02])
nationality = choice(
    ["Arab_GCC", "Arab_NonGCC", "South_Asian", "Southeast_Asian", "African", "European_Western", "Other"],
    p=[0.12, 0.15, 0.38, 0.10, 0.08, 0.10, 0.07]
)
level_of_study = choice(["Undergraduate", "Postgraduate", "Foundation"], p=[0.65, 0.28, 0.07])
university = choice(
    ["Murdoch_Dubai", "UOWD", "RIT_Dubai", "MAHE_Dubai", "Other_DIAC"],
    p=[0.18, 0.22, 0.20, 0.25, 0.15]
)
accommodation = choice(
    ["On_Campus", "Al_Barsha", "Deira_BurDubai", "Sharjah", "Ajman_Northern",
     "Downtown_JBR_JLT", "Other"],
    p=[0.08, 0.15, 0.18, 0.22, 0.12, 0.15, 0.10]
)

# ── SECTION B: Transport Patterns ────────────────────────────────────────────
# Base distance depends on accommodation
dist_map = {
    "On_Campus":       (1, 3),
    "Al_Barsha":       (12, 22),
    "Deira_BurDubai":  (25, 40),
    "Sharjah":         (30, 50),
    "Ajman_Northern":  (40, 60),
    "Downtown_JBR_JLT": (18, 30),
    "Other":           (10, 35),
}
distance_km = np.array([
    round(np.random.uniform(*dist_map[a]), 1) for a in accommodation
])

# Commute time correlated with distance (traffic factor 3-5 min/km)
traffic_factor = np.random.uniform(2.5, 5.0, N)
commute_time = np.clip((distance_km * traffic_factor).astype(int), 5, 120)

# Mode probabilities depend on distance
transport_mode = []
for d in distance_km:
    if d < 5:
        mode = choice(["Walking_Cycling", "University_Shuttle", "Own_Car"], p=[0.5, 0.35, 0.15], n=1)[0]
    elif d < 20:
        mode = choice(
            ["University_Shuttle", "RTA_Bus", "Metro", "Own_Car", "Taxi_RideHail", "Carpool", "Sub_Van"],
            p=[0.20, 0.18, 0.15, 0.18, 0.12, 0.10, 0.07], n=1)[0]
    else:
        mode = choice(
            ["University_Shuttle", "RTA_Bus", "Metro", "Own_Car", "Taxi_RideHail", "Carpool", "Sub_Van"],
            p=[0.18, 0.20, 0.10, 0.15, 0.15, 0.12, 0.10], n=1)[0]
    transport_mode.append(mode)
transport_mode = np.array(transport_mode)

days_per_week = choice(["1_2", "3_4", "5", "6plus"], p=[0.10, 0.35, 0.45, 0.10])

# Monthly spend: correlated with distance and mode
base_spend = distance_km * 4.5
mode_mult = {"University_Shuttle": 0.6, "RTA_Bus": 0.7, "Metro": 0.75,
             "Own_Car": 1.4, "Taxi_RideHail": 2.0, "Carpool": 0.8,
             "Sub_Van": 0.9, "Walking_Cycling": 0.1}
monthly_spend = np.array([
    round(base_spend[i] * mode_mult.get(transport_mode[i], 1.0) * np.random.uniform(0.8, 1.3), 0)
    for i in range(N)
])
monthly_spend = np.clip(monthly_spend, 50, 1500)

# ── SECTION C: Pain Points (binary) ──────────────────────────────────────────
# Correlated with commute time, distance, and mode
pain_cost          = (monthly_spend > 400).astype(int) | (np.random.rand(N) < 0.25).astype(int)
pain_long_time     = (commute_time > 40).astype(int) | (np.random.rand(N) < 0.20).astype(int)
pain_unreliable    = np.isin(transport_mode, ["RTA_Bus", "University_Shuttle"]).astype(int) & \
                     (np.random.rand(N) < 0.60).astype(int)
pain_safety        = (gender == "Female").astype(int) * (np.random.rand(N) < 0.45).astype(int) + \
                     (np.random.rand(N) < 0.15).astype(int)
pain_overcrowded   = np.isin(transport_mode, ["RTA_Bus", "Metro", "University_Shuttle"]).astype(int) & \
                     (np.random.rand(N) < 0.55).astype(int)
pain_no_late_night = (np.random.rand(N) < 0.35).astype(int)
pain_no_direct     = (distance_km > 25).astype(int) & (np.random.rand(N) < 0.50).astype(int)

# Clip all pain flags to binary 0/1
for arr in [pain_cost, pain_long_time, pain_unreliable, pain_safety,
            pain_overcrowded, pain_no_late_night, pain_no_direct]:
    arr[:] = np.clip(arr, 0, 1)

# Severity Likert (1–5); higher pain binary → higher severity
def pain_severity(pain_flag, noise=0.3):
    base = np.where(pain_flag == 1,
                    np.random.randint(3, 6, N),
                    np.random.randint(1, 4, N))
    return np.clip(base, 1, 5)

sev_cost        = pain_severity(pain_cost)
sev_time        = pain_severity(pain_long_time)
sev_unreliable  = pain_severity(pain_unreliable)
sev_safety      = pain_severity(pain_safety)
sev_overcrowded = pain_severity(pain_overcrowded)
sev_latenight   = pain_severity(pain_no_late_night)

# ── SECTION D: Preferences ────────────────────────────────────────────────────
max_commute_pref = choice(
    ["Less_20min", "20_30min", "31_45min", "46_60min", "More_60min"],
    p=[0.20, 0.35, 0.25, 0.12, 0.08]
)
# Budget correlated with current spend + willingness
max_budget_aed = np.clip(monthly_spend * np.random.uniform(0.7, 1.3, N), 100, 2000).round(0)

# Preferred mode (multi-select → 7 binary cols)
pref_shuttle   = (np.random.rand(N) < 0.55).astype(int)
pref_ondemand  = (np.random.rand(N) < 0.60).astype(int)
pref_subvan    = (np.random.rand(N) < 0.40).astype(int)
pref_carpool   = (np.random.rand(N) < 0.50).astype(int)
pref_ridehail  = (np.random.rand(N) < 0.45).astype(int)
pref_public    = (np.random.rand(N) < 0.35).astype(int)

# Feature interest (multi-select → 10 binary cols) - used for ARM
feat_live_tracking     = (np.random.rand(N) < 0.78).astype(int)
feat_carpool_match     = (np.random.rand(N) < 0.62).astype(int)
feat_subscription_pass = (np.random.rand(N) < 0.58).astype(int)
feat_female_only       = ((gender == "Female") & (np.random.rand(N) < 0.55)).astype(int)
feat_pickup_home       = (np.random.rand(N) < 0.70).astype(int)
feat_inapp_support     = (np.random.rand(N) < 0.50).astype(int)
feat_cashless          = (np.random.rand(N) < 0.75).astype(int)
feat_realtime_alerts   = (np.random.rand(N) < 0.65).astype(int)
feat_rating_system     = (np.random.rand(N) < 0.60).astype(int)
feat_timetable_integ   = (np.random.rand(N) < 0.55).astype(int)

# Importance Likert (1–5)
imp_cost        = np.random.randint(2, 6, N)
imp_time        = np.random.randint(2, 6, N)
imp_safety      = np.random.randint(3, 6, N)
imp_comfort     = np.random.randint(1, 6, N)
imp_reliability = np.random.randint(2, 6, N)
imp_flexibility = np.random.randint(1, 6, N)

# ── SECTION E: Digital Behaviour ─────────────────────────────────────────────
app_googlemaps  = (np.random.rand(N) < 0.82).astype(int)
app_rta         = (np.random.rand(N) < 0.45).astype(int)
app_uber        = (np.random.rand(N) < 0.58).astype(int)
app_careem      = (np.random.rand(N) < 0.52).astype(int)
app_uni_portal  = (np.random.rand(N) < 0.48).astype(int)
app_whatsapp    = (np.random.rand(N) < 0.65).astype(int)
app_none        = ((app_googlemaps + app_rta + app_uber + app_careem +
                    app_uni_portal + app_whatsapp) == 0).astype(int)

app_comfort     = np.random.randint(2, 6, N)
ridehail_freq   = choice(
    ["Daily", "3_5_weekly", "1_2_weekly", "Rarely", "Never"],
    p=[0.08, 0.18, 0.28, 0.30, 0.16]
)

# ── SECTION F: Satisfaction & Target ─────────────────────────────────────────
transport_satisfaction = np.clip(
    5 - (pain_cost + pain_long_time + pain_unreliable) + np.random.randint(-1, 2, N), 1, 5
)
nps_likelihood  = np.clip(transport_satisfaction + np.random.randint(-1, 2, N), 1, 5)
willingness_to_pay = np.where(
    monthly_spend > 300,
    np.random.uniform(50, 250, N).round(0),
    np.random.uniform(0, 150, N).round(0)
)

# Target label: interest in Easy Transport App
# Higher pain + lower satisfaction → more likely "Yes"
pain_total = (pain_cost + pain_long_time + pain_unreliable +
              pain_safety + pain_overcrowded + pain_no_late_night + pain_no_direct)
score = pain_total - transport_satisfaction + app_comfort
score_norm = (score - score.min()) / (score.max() - score.min())

def assign_interest(s):
    if s > 0.60:
        return "Yes"
    elif s > 0.30:
        return "Maybe"
    else:
        return "No"

interest_label = np.array([assign_interest(s + np.random.uniform(-0.1, 0.1)) for s in score_norm])

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    # Demographics
    "Age":                          age,
    "Gender":                       gender,
    "Nationality_Region":           nationality,
    "Level_Of_Study":               level_of_study,
    "University":                   university,
    "Accommodation_Area":           accommodation,
    # Transport patterns
    "Primary_Transport_Mode":       transport_mode,
    "Commute_Time_OneWay_Min":      commute_time,
    "Distance_To_Campus_KM":        distance_km,
    "Days_Per_Week":                days_per_week,
    "Monthly_Spend_AED":            monthly_spend,
    # Pain points (binary)
    "Pain_HighCost":                pain_cost,
    "Pain_LongTime":                pain_long_time,
    "Pain_Unreliable":              pain_unreliable,
    "Pain_Safety":                  pain_safety,
    "Pain_Overcrowded":             pain_overcrowded,
    "Pain_NoLateNight":             pain_no_late_night,
    "Pain_NoDirectRoute":           pain_no_direct,
    # Severity Likert
    "Severity_Cost":                sev_cost,
    "Severity_Time":                sev_time,
    "Severity_Unreliable":          sev_unreliable,
    "Severity_Safety":              sev_safety,
    "Severity_Overcrowded":         sev_overcrowded,
    "Severity_LateNight":           sev_latenight,
    # Preferences
    "Max_Commute_Preference":       max_commute_pref,
    "Max_Monthly_Budget_AED":       max_budget_aed,
    "PrefMode_FixedShuttle":        pref_shuttle,
    "PrefMode_OnDemandShuttle":     pref_ondemand,
    "PrefMode_SubVan":              pref_subvan,
    "PrefMode_Carpool":             pref_carpool,
    "PrefMode_RideHail":            pref_ridehail,
    "PrefMode_PublicTransport":     pref_public,
    # Feature interest
    "Feature_LiveTracking":         feat_live_tracking,
    "Feature_CarpoolMatch":         feat_carpool_match,
    "Feature_SubscriptionPass":     feat_subscription_pass,
    "Feature_FemaleOnly":           feat_female_only,
    "Feature_PickupHome":           feat_pickup_home,
    "Feature_InAppSupport":         feat_inapp_support,
    "Feature_CashlessPayment":      feat_cashless,
    "Feature_RealTimeAlerts":       feat_realtime_alerts,
    "Feature_RatingSystem":         feat_rating_system,
    "Feature_TimetableInteg":       feat_timetable_integ,
    # Importance Likert
    "Importance_Cost":              imp_cost,
    "Importance_Time":              imp_time,
    "Importance_Safety":            imp_safety,
    "Importance_Comfort":           imp_comfort,
    "Importance_Reliability":       imp_reliability,
    "Importance_Flexibility":       imp_flexibility,
    # Digital behaviour
    "App_GoogleMaps":               app_googlemaps,
    "App_RTA":                      app_rta,
    "App_Uber":                     app_uber,
    "App_Careem":                   app_careem,
    "App_UniPortal":                app_uni_portal,
    "App_WhatsApp":                 app_whatsapp,
    "App_None":                     app_none,
    "App_Comfort_Score":            app_comfort,
    "RideHail_Frequency":           ridehail_freq,
    # Satisfaction
    "Transport_Satisfaction":       transport_satisfaction,
    "NPS_Likelihood":               nps_likelihood,
    "Willingness_To_Pay_AED":       willingness_to_pay,
    # Target
    "Interested_In_EasyTransport_App": interest_label,
})

df.to_csv("easy_transport_survey.csv", index=False)

print(f"Dataset saved: easy_transport_survey.csv  ({N} rows × {len(df.columns)} columns)")
print("\nTarget distribution:")
print(df["Interested_In_EasyTransport_App"].value_counts())
print("\nSample rows:")
print(df.head(5).to_string())
