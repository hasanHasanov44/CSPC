# CSPC Practical Work 4

## Session A: Concrete Strength Prediction

### A1. Variables and Units
| Variable | Role | Type | Units | Description |
|---|---|---|---|---|
| `cement` | Input | Quantitative | $\text{kg/m}^3$ | Cement content |
| `slag` | Input | Quantitative | $\text{kg/m}^3$ | Blast furnace slag |
| `ash` | Input | Quantitative | $\text{kg/m}^3$ | Fly ash |
| `water` | Input | Quantitative | $\text{kg/m}^3$ | Water content |
| `superplastic` | Input | Quantitative | $\text{kg/m}^3$ | Superplasticizer |
| `coarse_agg` | Input | Quantitative | $\text{kg/m}^3$ | Coarse aggregate |
| `fine_agg` | Input | Quantitative | $\text{kg/m}^3$ | Fine aggregate |
| `age` | Input | Quantitative | Days | Curing time |
| `strength` | Target | Quantitative | MPa | Compressive strength |

### A5. Polynomial Regression Analysis
- **Optimal Degree Choice**: Degree 2 (or Degree 3 depending on test $R^2$) is chosen as it provides the best balance between underfitting and overfitting.
- **Degree 4 Overfitting**: At degree 4, the number of engineered features ($p = 495$) approaches the number of training samples ($n \approx 800$). This leads to extreme overfitting: the model memorizes the training set ($\text{Train } R^2 \approx 1$), but fails to generalize to new data, causing the $\text{Test } R^2$ to drop drastically (or become negative).
- **Adjusted $R^2$ Limitation**: Adjusted $R^2$ penalizes additional features based on the training data alone, but it cannot fully protect against overfitting on unseen test data when $p$ becomes excessively large relative to $n$.