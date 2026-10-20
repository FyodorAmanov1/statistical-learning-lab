# Primary references and boundaries

- Hoerl and Kennard (1970), Ridge Regression: Biased Estimation for Nonorthogonal Problems: https://doi.org/10.1080/00401706.1970.10488634
- Hastie, Tibshirani and Friedman, The Elements of Statistical Learning: https://hastie.su.domains/ElemStatLearn/

The initial synthetic experiment uses a prescribed AR(1) coefficient change and training-only preprocessing. It provides a controlled illustration rather than market forecasting evidence.

## ML01 preprocessing references

Verified through web search and direct retrieval on 2026-10-20:

- scikit-learn maintainers, [Common pitfalls: data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
  Primary implementation guidance for splitting before fitting preprocessing,
  fitting transforms only on training data, and reusing them on held-out data.
- scikit-learn maintainers, [StandardScaler API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
  Defines training-fitted mean/scale, population variance (`ddof=0`), and a scale
  of one for zero-variance features. These conventions inform our NumPy scaler;
  scikit-learn is not an added dependency or an asserted drop-in equivalent.

Our scaler requires finite dense 2D inputs and optionally preserves explicitly
excluded columns, including the experiment's constant. It has no sparse,
sample-weight, missing-value, or incremental-fit support. The mutation audit is
a software isolation check, not a statistical generalization guarantee. The
leaky control's coefficient difference is measured in each scaler's own feature
coordinates; it is not by itself an accuracy or raw-coordinate effect size.
