import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veriler
data = {
    "Model": [
        "LightGBM", "CatBoost", "XGBoost", "Extra Trees",
        "Gradient Boosting", "Random Forest",
        "KNN", "SVM", "Logistic Regression",
        "Decision Tree", "AdaBoost"
    ],
    "Time (s)": [
        17.366250, 530.143966, 15.406534, 2.692939,
        752.650735, 13.941680, 0.383556,
        24.344774, 1.732285, 2.751479, 10.199561
    ],
    "Accuracy": [
        0.979277, 0.976225, 0.975100, 0.967871,
        0.965141, 0.962088, 0.927550,
        0.929157, 0.906506, 0.879197, 0.445622
    ],
    "Precision": [
        0.981171, 0.978743, 0.975854, 0.973314,
        0.965902, 0.966203, 0.928665,
        0.928981, 0.905294, 0.872755, 0.332651
    ],
    "Recall": [
        0.971738, 0.972870, 0.967734, 0.962370,
        0.953368, 0.950584, 0.921673,
        0.914673, 0.903185, 0.866226, 0.375901
    ],
    "F1 Score": [
        0.976166, 0.975673, 0.971583, 0.967468,
        0.959218, 0.957577, 0.924772,
        0.920672, 0.904119, 0.868868, 0.314044
    ]
}

df = pd.DataFrame(data).set_index("Model")

metrics_df = df.drop(columns=["Time (s)"])

plt.figure(figsize=(10, 6))
sns.heatmap(
    metrics_df,
    annot=True,
    fmt=".4f",
    cmap="RdYlGn",
    linewidths=0.6,
    linecolor="white",
    cbar=True
)

plt.title("Model Performans Metrikleri", fontsize=13)
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.show()