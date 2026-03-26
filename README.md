# Student Score Predictor

I built this as my BYOP project for the Fundamentals of AI and ML course. The idea came from a pretty simple question I've always had — does studying more actually lead to better exam scores, and by how much?

Turns out you can answer that with a dataset and a regression line.

---

## What this does

- Generates a synthetic dataset of 200 student records
- Plots study hours against exam scores to visually check the relationship
- Implements linear regression **from scratch** using just NumPy (no sklearn)
- Evaluates the model using MAE, RMSE, and R²
- Predicts the score for any given number of study hours

---

## Project Structure

```
student-score-predictor/
├── student_score_predictor.py   # the whole project, ~60 lines
├── requirements.txt
├── study_vs_score.png           # generated when you run it
├── regression_fit.png           # generated when you run it
└── README.md
```

---

## How to run it

```bash
pip install -r requirements.txt
python student_score_predictor.py
```

Two plots will be saved in the same folder, and the terminal will print the model equation, metrics, and a sample prediction.

---

## Sample output

```
Model: score = 3.47 × study_hours + 28.31
MAE  : 6.12  (off by this many marks on average)
RMSE : 7.84
R²   : 0.7231  (how much of the score variation is explained)

If a student studies 7 hrs/day → predicted score: 72.6 / 100
```

---

## Dependencies

```
numpy
pandas
matplotlib
```

---

## Why I kept it simple

I intentionally avoided sklearn and complex models. I wanted to actually understand what linear regression is doing mathematically, not just call `.fit()` and hope for the best. The least squares formula is only a couple of lines in NumPy, and seeing it produce a working model was genuinely satisfying.
