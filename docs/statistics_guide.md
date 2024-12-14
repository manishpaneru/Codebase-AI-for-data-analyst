# Statistics Guide for Data Analysis

## Introduction
Understanding statistics is crucial for data analysis. This guide covers essential statistical concepts, tests, and their practical applications in data analysis.

## Descriptive Statistics

### Central Tendency
- **Mean**: Average of all values
- **Median**: Middle value when ordered
- **Mode**: Most frequent value

```python
import numpy as np

# Calculate mean
mean = np.mean(data)

# Calculate median
median = np.median(data)

# Calculate mode
from scipy import stats
mode = stats.mode(data)
```

### Spread
- **Variance**: Average squared deviation from mean
- **Standard Deviation**: Square root of variance
- **Range**: Difference between max and min
- **Interquartile Range (IQR)**: Range between 25th and 75th percentiles

```python
# Calculate variance
variance = np.var(data)

# Calculate standard deviation
std_dev = np.std(data)

# Calculate range
data_range = np.max(data) - np.min(data)

# Calculate IQR
q75, q25 = np.percentile(data, [75, 25])
iqr = q75 - q25
```

## Probability Distributions

### Normal Distribution
- Bell-shaped curve
- Defined by mean and standard deviation
- Common in natural phenomena

```python
from scipy import stats

# Generate normal distribution
x = np.linspace(-4, 4, 100)
y = stats.norm.pdf(x, loc=0, scale=1)

# Test for normality
statistic, p_value = stats.normaltest(data)
```

### Other Common Distributions
- Binomial Distribution
- Poisson Distribution
- Chi-Square Distribution
- Student's t-Distribution

## Hypothesis Testing

### Steps in Hypothesis Testing
1. State null and alternative hypotheses
2. Choose significance level (α)
3. Calculate test statistic
4. Compare p-value with α
5. Make decision

### Common Tests

#### T-Test
```python
# Independent samples t-test
t_stat, p_value = stats.ttest_ind(group1, group2)

# Paired t-test
t_stat, p_value = stats.ttest_rel(before, after)

# One-sample t-test
t_stat, p_value = stats.ttest_1samp(data, population_mean)
```

#### Chi-Square Test
```python
# Chi-square test of independence
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
```

#### ANOVA
```python
# One-way ANOVA
f_stat, p_value = stats.f_oneway(group1, group2, group3)
```

## Correlation and Regression

### Correlation
- Pearson correlation (linear)
- Spearman correlation (monotonic)
- Kendall's tau (ordinal)

```python
# Pearson correlation
r, p_value = stats.pearsonr(x, y)

# Spearman correlation
rho, p_value = stats.spearmanr(x, y)
```

### Linear Regression
```python
from sklearn.linear_model import LinearRegression

# Simple linear regression
model = LinearRegression()
model.fit(X, y)

# Get coefficients
slope = model.coef_
intercept = model.intercept_

# R-squared
r_squared = model.score(X, y)
```

## Sampling

### Sample Size Calculation
```python
from statsmodels.stats.power import TTestIndPower

# Calculate required sample size
analysis = TTestIndPower()
sample_size = analysis.solve_power(
    effect_size=0.5,
    power=0.8,
    alpha=0.05
)
```

### Sampling Methods
1. Simple Random Sampling
2. Stratified Sampling
3. Cluster Sampling
4. Systematic Sampling

## Effect Size

### Common Effect Size Measures
- Cohen's d
- Pearson's r
- Odds Ratio
- Risk Ratio

```python
# Calculate Cohen's d
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_se
```

## Best Practices

1. Always check assumptions before applying tests
2. Use appropriate tests for your data type
3. Consider effect size, not just p-values
4. Report confidence intervals
5. Be aware of multiple testing problems
6. Document your analysis steps

## Common Pitfalls

1. Assuming normality without testing
2. Ignoring effect size
3. P-hacking
4. Overlooking assumptions
5. Misinterpreting correlation as causation

## Practical Applications

### A/B Testing
```python
def ab_test(control, treatment, alpha=0.05):
    t_stat, p_value = stats.ttest_ind(control, treatment)
    effect_size = cohens_d(control, treatment)
    return {
        'p_value': p_value,
        'significant': p_value < alpha,
        'effect_size': effect_size
    }
```

### Outlier Detection
```python
def detect_outliers(data, threshold=1.5):
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    upper_bound = q75 + threshold * iqr
    lower_bound = q25 - threshold * iqr
    return (data > upper_bound) | (data < lower_bound)
```

### Power Analysis
```python
def power_analysis(effect_size, n_samples, alpha=0.05):
    analysis = TTestIndPower()
    power = analysis.power(
        effect_size=effect_size,
        nobs=n_samples,
        alpha=alpha
    )
    return power
```

## Resources

- [Statistics in Python Tutorial](https://scipy-lectures.org/packages/statistics/index.html)
- [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html)
- [Statistical Tests Overview](https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/) 