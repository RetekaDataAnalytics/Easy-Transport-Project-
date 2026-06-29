# 🚌 Easy Transport — Dubai Academic City Dashboard

A data analytics dashboard for the **Easy Transport** student commuting platform concept,  
built for a Project-Based Learning (PBL) individual data analytics assignment.

## 📊 Live Dashboard

👉 **[View Dashboard](https://YOUR-USERNAME.github.io/easy-transport-dashboard/)**  
*(Replace `YOUR-USERNAME` with your GitHub username after deploying)*

---

## 📁 Repository Structure

```
easy-transport-dashboard/
│
├── index.html                    ← Self-contained dashboard (open this in any browser)
├── dashboard_data.json           ← Pre-aggregated chart data (400 respondents, 20 metrics)
├── easy_transport_survey.csv     ← Full synthetic dataset (400 rows × 61 columns)
│
├── generate_dataset.py           ← Python script that generated the synthetic dataset
├── easy_transport_classification.py  ← ML classification code (Decision Tree, RF, GBT)
│
└── README.md                     ← This file
```

---

## 📋 Dashboard Pages

| Page | What You'll See |
|---|---|
| **Overview** | KPI cards, app interest split, interest by university, satisfaction & spend distribution |
| **Demographics** | Gender, level of study, university, accommodation area, nationality |
| **Transport Patterns** | Current modes, avg spend per mode, days on campus, ride-hail frequency |
| **Pain Points** | Ranked pain point frequency, average severity radar chart |
| **Features & Preferences** | Feature interest bar chart, preferred modes, importance radar, budget willingness |
| **Digital Behaviour** | App usage frequency, app comfort score distribution |

---

## 🗂️ Dataset Overview

**File:** `easy_transport_survey.csv`  
**Rows:** 400 synthetic student responses  
**Columns:** 61 variables

| Category | Variables |
|---|---|
| Demographics | Age, Gender, Nationality_Region, Level_Of_Study, University, Accommodation_Area |
| Transport Pattern | Primary_Transport_Mode, Commute_Time_OneWay_Min, Distance_To_Campus_KM, Monthly_Spend_AED |
| Pain Points (binary) | Pain_HighCost, Pain_LongTime, Pain_Unreliable, Pain_Safety, Pain_Overcrowded, Pain_NoLateNight, Pain_NoDirectRoute |
| Severity (Likert 1–5) | Severity_Cost, Severity_Time, Severity_Unreliable, Severity_Safety … |
| Feature Interest (binary) | Feature_LiveTracking, Feature_CarpoolMatch, Feature_SubscriptionPass … |
| Importance (Likert 1–5) | Importance_Cost, Importance_Time, Importance_Safety … |
| Digital Behaviour | App_GoogleMaps, App_Uber, App_Careem, RideHail_Frequency … |
| **Target (Classification)** | **Interested_In_EasyTransport_App** (Yes / No / Maybe) |

---

## 🛠️ How to Run the Python Code

```bash
# 1. Install dependencies
pip install pandas scikit-learn matplotlib seaborn

# 2. (Optional) Regenerate the synthetic dataset
python generate_dataset.py

# 3. Run classification models
python easy_transport_classification.py
```

The classification script trains **Decision Tree**, **Random Forest**, and **Gradient Boosting** models  
and outputs accuracy tables, confusion matrices, and feature importance plots.

---

## 🚀 Deploy to GitHub Pages (Step-by-Step)

1. Create a new GitHub repository named `easy-transport-dashboard`
2. Upload all files from this folder to the repository root
3. Go to **Settings → Pages**
4. Under **Source**, select `main` branch and `/ (root)` folder
5. Click **Save** — your dashboard will be live in ~1 minute at:  
   `https://YOUR-USERNAME.github.io/easy-transport-dashboard/`

---

## 📌 Project Context

- **Business Idea:** Easy Transport — a mobile/web platform for students in Dubai Academic City  
  to find affordable, safe, and convenient commuting options  
- **Survey Target:** Students at MAHE Dubai, UOWD, RIT Dubai, Murdoch Dubai, and other DIAC institutions  
- **ML Tasks:**
  - 🔵 **Classification** — Predict interest in the app (Yes / No / Maybe)
  - 🟡 **Regression** — Predict monthly transport spend
  - 🟢 **Clustering** — Identify student commuter personas
  - 🔴 **Association Rule Mining** — Pain points and feature preferences co-occurrence

---

*Synthetic dataset generated for academic/demo purposes only.*
