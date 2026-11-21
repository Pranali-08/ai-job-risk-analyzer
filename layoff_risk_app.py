import os
import matplotlib
matplotlib.use("Agg")  # save charts as files instead of opening windows
os.makedirs("charts", exist_ok=True)
_n = 0
def save_chart(name=None):
    global _n
    _n += 1
    import matplotlib.pyplot as _plt
    _plt.savefig(f"charts/{name or f'eda_{_n}'}.png", bbox_inches="tight")
    _plt.close()


import pandas as pd

df = pd.read_csv("ai-impact-jobs-layoff-risk-dataset.csv")
df.head()

df.shape

df.groupby("Layoff_Risk")["Routine_Task_Percentage"].mean()

df.groupby("Layoff_Risk")["Creativity_Requirement"].mean()

df.groupby("Layoff_Risk")["AI_Adoption_Level"].value_counts()

import matplotlib.pyplot as plt
avg_routine = df.groupby("Layoff_Risk")["Routine_Task_Percentage"].mean()

avg_routine.plot(kind="bar", color=["green", "orange", "red"])
plt.title("Average Routine Task Percentage by Layoff Risk")
plt.xlabel("Layoff Risk")
plt.ylabel("Average Routine Task Percentage")
save_chart()

risk_order = ["Low", "Medium", "High"]
avg_routine = avg_routine.reindex(risk_order)

avg_routine.plot(kind="bar", color=["green", "orange", "red"])
plt.title("Average Routine Task % by Layoff Risk")
plt.xlabel("Layoff Risk")
plt.ylabel("Routine Task %")
save_chart()

avg_creativity = df.groupby("Layoff_Risk")["Creativity_Requirement"].mean()
risk_order = ["Low", "Medium", "High"]
avg_creativity = avg_creativity.reindex(risk_order)

avg_creativity.plot(kind="bar", color=["green", "orange", "red"])
plt.title("Average Creativity Requirement % by Layoff Risk")
plt.xlabel("Layoff Risk")
plt.ylabel("Creativity Requirement %")
save_chart()

import seaborn as sns

sns.boxplot(data=df, x="Layoff_Risk", y="Years_of_Experience", order=risk_order)
plt.title("Years of Experience by Layoff Risk")
plt.xlabel("Layoff Risk")
plt.ylabel("Years of Experience")
save_chart()


df = pd.read_csv("ai-impact-jobs-layoff-risk-dataset.csv")

df["Education_Level"].unique()

df["Job_Level"].unique()

education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
df["Education_Level"] = df["Education_Level"].map(education_map)

df['Education_Level'].unique()

size_map = {"Small": 0, "Medium": 1, "Large": 2}
df["Company_Size"] = df["Company_Size"].map(size_map)

df["Company_Size"].unique()

level_map = {"Entry": 0, "Mid":1, "Senior":2}
df["Job_Level"] = df["Job_Level"].map(level_map)

df["Job_Level"].unique()

ai_map = {"Low": 0, "Medium": 1, "High": 2}
df["AI_Adoption_Level"] = df["AI_Adoption_Level"].map(ai_map)
df["AI_Adoption_Level"].unique()

df = pd.get_dummies(df, columns=["Industry", "Job_Role"])

df.head()

df.shape

ai_map = {"Low": 0, "Medium": 1, "High": 2}
df["Layoff_Risk"] = df["Layoff_Risk"].map(ai_map)
df["Layoff_Risk"].unique()

df.head()

X = df.drop("Layoff_Risk", axis=1)
y = df["Layoff_Risk"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

import joblib

joblib.dump(model, "layoff_risk_model.pkl")

# Confusion matrix: where does the model get confused?
from sklearn.metrics import ConfusionMatrixDisplay

y_pred = model.predict(X_test)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Low", "Medium", "High"], cmap="Blues")
plt.title("Confusion Matrix (test set)")
save_chart("confusion_matrix")

# Feature importance: which inputs drive the prediction most?
importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
importances.tail(10).plot(kind="barh", color="teal")
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
save_chart("feature_importance")

print(importances.tail(10).sort_values(ascending=False).round(3))

# Compare with a simple baseline using 5-fold cross-validation
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest (300 trees)": model,
}
for name, m in models.items():
    scores = cross_val_score(m, X, y, cv=5)
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

X_train.columns.tolist()

education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
size_map = {"Small": 0, "Medium": 1, "Large": 2}
level_map = {"Entry": 0, "Mid": 1, "Senior": 2}
ai_map = {"Low": 0, "Medium": 1, "High": 2}

industries = ["Education", "Finance", "Healthcare", "IT", "Logistics", "Manufacturing", "Retail", "Telecom"]
job_roles = ["Academic Coordinator", "Accountant", "Auditor", "Data Analyst", "Dispatcher",
             "Financial Analyst", "Health Analyst", "Inventory Analyst", "ML Engineer",
             "Medical Assistant", "Network Engineer", "Nurse", "Operations Analyst", "Operator",
             "Production Supervisor", "Quality Engineer", "Research Assistant", "Sales Associate",
             "Software Engineer", "Store Manager", "Supply Chain Analyst", "Support Specialist",
             "Teacher", "Warehouse Manager"]

defaults = X_train.mean().to_dict()
print(defaults)

hidden_defaults = {
    "Years_of_Experience": 7.27,
    "Company_Size": 0.95,
    "Job_Level": 0.70,
    "Human_Interaction_Level": 60.10,
    "Number_of_AI_Tools_Used": 2.46,
    "AI_Usage_Hours_Per_Week": 6.78,
    "Tasks_Automated_Percentage": 37.18,
    "AI_Training_Hours": 12.61,
}

def predict_risk(age, education, industry, job_role, routine_task, creativity, ai_adoption):

    row = {col: 0 for col in X_train.columns}

    # Fill in hidden fields with realistic averages
    for col, val in hidden_defaults.items():
        row[col] = val

    # Fill in the 7 visible, user-controlled fields
    row["Age"] = age
    row["Education_Level"] = education_map[education]
    row["Routine_Task_Percentage"] = routine_task
    row["Creativity_Requirement"] = creativity
    row["AI_Adoption_Level"] = ai_map[ai_adoption]

    row[f"Industry_{industry}"] = True
    row[f"Job_Role_{job_role}"] = True

    input_df = pd.DataFrame([row])[X_train.columns]
    prediction = model.predict(input_df)[0]

    risk_labels = {0: "Low", 1: "Medium", 2: "High"}
    return f"Predicted Layoff Risk: {risk_labels[prediction]}"

import gradio as gr

import gradio as gr

# Proper font loading — Gradio's own GoogleFont class injects it correctly,
# instead of an @import that gets stripped/overridden
theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("Space Grotesk"), gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    # ...keep everything else exactly the same as before...
    body_background_fill="#0a0e17", body_background_fill_dark="#0a0e17",
    background_fill_primary="#111827", background_fill_secondary="#0a0e17",
    border_color_primary="#1f2937", block_background_fill="#111827",
    block_border_color="#1f2937", block_label_text_color="#94a3b8",
    block_title_text_color="#f1f5f9", body_text_color="#e2e8f0",
    input_background_fill="#0a0e17", input_border_color="#1f2937",
    slider_color="#22d3ee",
    button_primary_background_fill="#22d3ee",
    button_primary_background_fill_hover="#06b6d4",
    button_primary_text_color="#0a0e17",
)

custom_css = """
#header h1 { font-weight: 700 !important; letter-spacing: -0.5px; }
#header p { color: #64748b !important; }
#scan-btn { font-weight: 700 !important; letter-spacing: 0.5px; }
#result-panel { padding: 28px !important; }

/* Reset Gradio's own span styling from leaking into our custom HTML */
#result-panel span {
    background: none !important;
    border: none !important;
    padding: 0 !important;
    border-radius: 0 !important;
    text-decoration: none !important;
    display: inline !important;
}
"""

def predict_risk(age, education, industry, job_role, routine_task, creativity, ai_adoption):
    row = {col: 0 for col in X_train.columns}
    for col, val in hidden_defaults.items():
        row[col] = val
    row["Age"] = age
    row["Education_Level"] = education_map[education]
    row["Routine_Task_Percentage"] = routine_task
    row["Creativity_Requirement"] = creativity
    row["AI_Adoption_Level"] = ai_map[ai_adoption]
    row[f"Industry_{industry}"] = True
    row[f"Job_Role_{job_role}"] = True
    input_df = pd.DataFrame([row])[X_train.columns]
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]  # NEW: confidence per class

    risk_config = {
        0: {"label": "LOW",    "color": "#34d399", "pct": 20, "msg": "This role shows strong resistance to AI automation — it leans on skills AI still struggles to replicate."},
        1: {"label": "MEDIUM", "color": "#fbbf24", "pct": 55, "msg": "This role sits in the middle ground — part of the work could plausibly be automated over time."},
        2: {"label": "HIGH",   "color": "#f87171", "pct": 88, "msg": "This role has heavy overlap with tasks AI already automates well today."},
    }
    r = risk_config[prediction]

    html = f"""
    <div style="font-family:'Space Grotesk',sans-serif;">
      <div style="font-family:'JetBrains Mono',monospace; color:#64748b; font-size:0.78rem; letter-spacing:1px; margin-bottom:8px;">
        SCAN RESULT
      </div>
      <div style="font-size:2.6rem; font-weight:700; color:{r['color']}; text-shadow:0 0 24px {r['color']}66; margin-bottom:20px;">
        {r['label']} RISK
      </div>
      <div style="background:#0a0e17; border-radius:8px; height:10px; width:100%; overflow:hidden; margin-bottom:8px;">
        <div style="background:{r['color']}; box-shadow:0 0 12px {r['color']}; height:100%; width:{r['pct']}%;"></div>
      </div>
      <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#475569; margin-bottom:24px;">
        <span>LOW</span><span>MEDIUM</span><span>HIGH</span>
      </div>
      <div style="font-family:'Inter',sans-serif; color:#94a3b8; font-size:0.9rem; line-height:1.55; margin-bottom:26px;">
        {r['msg']}
      </div>
      <div style="border-top:1px solid #1f2937; padding-top:18px;">
        <div style="font-family:'JetBrains Mono',monospace; color:#475569; font-size:0.7rem; letter-spacing:1px; margin-bottom:10px;">
          MODEL CONFIDENCE
        </div>
        {"".join([f'''
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
          <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#64748b; width:60px;">{lbl}</span>
          <div style="flex:1; background:#0a0e17; border-radius:6px; height:6px; overflow:hidden;">
            <div style="background:{col}; height:100%; width:{p*100:.0f}%;"></div>
          </div>
          <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#64748b; width:34px; text-align:right;">{p*100:.0f}%</span>
        </div>
        ''' for lbl, col, p in zip(["LOW","MEDIUM","HIGH"], ["#34d399","#fbbf24","#f87171"], probs)])}
      </div>
    </div>
    """
    return html

with gr.Blocks(css=custom_css, theme=theme) as demo:
    with gr.Column(elem_id="header"):
        gr.Markdown("# AI Job Risk Analyzer")
        gr.Markdown("Enter a job profile below to scan its exposure to AI-driven automation.")

    with gr.Row():
        with gr.Column(scale=1):
            age = gr.Number(value=30, label="Age", precision=0)
            education = gr.Dropdown(["High School", "Bachelor's", "Master's", "PhD"], value="Bachelor's", label="Education Level")
            industry = gr.Dropdown(industries, value="IT", label="Industry")
            job_role = gr.Dropdown(job_roles, value="Software Engineer", label="Job Role")
            routine_task = gr.Slider(0, 100, value=50, step=1, label="Routine Task %")
            creativity = gr.Slider(0, 100, value=50, step=1, label="Creativity Requirement %")
            ai_adoption = gr.Dropdown(["Low", "Medium", "High"], value="Medium", label="AI Adoption Level")
            btn = gr.Button("Run Scan", elem_id="scan-btn", variant="primary")

        with gr.Column(scale=1, elem_id="result-panel"):
            output = gr.HTML("<div style='color:#475569; font-family:Inter; padding:40px 0;'>Fill in the form and hit <b>Run Scan</b> to see the result.</div>")

    btn.click(fn=predict_risk,
              inputs=[age, education, industry, job_role, routine_task, creativity, ai_adoption],
              outputs=output)

demo.launch()
