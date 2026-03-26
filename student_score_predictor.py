# I built this project to explore a simple question I've had for a while:
# does the number of hours you study actually predict how well you do in exams?
# Turns out, yes — and you can prove it with about 50 lines of Python.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 1. Dataset ───────────────────────────────────────────────
# I'm generating fake student data here since I don't have access to real records.
# The score formula is based on what I've read about what actually affects performance —
# study hours matter most, but sleep and attendance play a role too.

np.random.seed(42)
N = 200

study  = np.random.uniform(1, 10, N)   # hours studied per day
sleep  = np.random.uniform(4, 10, N)   # hours of sleep
attend = np.random.uniform(50, 100, N) # attendance percentage
prev   = np.random.uniform(40, 100, N) # previous exam score

# Weighted formula + some random noise to make it realistic
score = (3.5 * study + 1.5 * sleep + 0.3 * attend + 0.2 * prev
         + np.random.normal(0, 5, N)).clip(0, 100)

df = pd.DataFrame({
    "study_hours": study,
    "sleep_hours": sleep,
    "attendance":  attend,
    "prev_score":  prev,
    "exam_score":  score
})

print("Dataset shape:", df.shape)
print(df.head())
print("\nBasic stats:")
print(df.describe().round(2))


# ── 2. Quick look at the data ────────────────────────────────
# Before jumping into modelling I wanted to see if there's even
# a visible pattern between study hours and scores. There is.

plt.figure(figsize=(6, 4))
plt.scatter(df["study_hours"], df["exam_score"], alpha=0.5, color="steelblue")
plt.xlabel("Study Hours per Day")
plt.ylabel("Exam Score")
plt.title("Do more study hours mean better scores?")
plt.tight_layout()
plt.savefig("study_vs_score.png")
plt.show()
print("Plot saved: study_vs_score.png")


# ── 3. Linear Regression from scratch ───────────────────────
# I decided not to use sklearn here — I wanted to actually understand
# what's happening under the hood. The least squares formula is
# just a few lines of numpy and it was satisfying to see it work.

X = df["study_hours"].values
y = df["exam_score"].values

x_mean, y_mean = X.mean(), y.mean()
slope     = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean) ** 2)
intercept = y_mean - slope * x_mean

print(f"\nModel: score = {slope:.2f} × study_hours + {intercept:.2f}")


# ── 4. How well does it do? ──────────────────────────────────
y_pred = slope * X + intercept

mae  = np.mean(np.abs(y - y_pred))
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
r2   = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2)

print(f"MAE  : {mae:.2f}  (off by this many marks on average)")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}  (how much of the score variation is explained)")


# ── 5. Plot the regression line ──────────────────────────────
x_line = np.linspace(X.min(), X.max(), 100)
y_line = slope * x_line + intercept

plt.figure(figsize=(6, 4))
plt.scatter(X, y, alpha=0.4, color="steelblue", label="Actual scores")
plt.plot(x_line, y_line, color="red", linewidth=2, label="Model prediction")
plt.xlabel("Study Hours per Day")
plt.ylabel("Exam Score")
plt.title("Linear Regression Fit")
plt.legend()
plt.tight_layout()
plt.savefig("regression_fit.png")
plt.show()
print("Plot saved: regression_fit.png")


# ── 6. Try it on a new student ───────────────────────────────
hours = 7
predicted = slope * hours + intercept
print(f"\nIf a student studies {hours} hrs/day → predicted score: {predicted:.1f} / 100")
