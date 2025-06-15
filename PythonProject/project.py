# === IMPORT SECTION ===
# import the needed libraries for the code to function
import pandas as pd
import math
import numpy as np
import scipy.stats
import os
import matplotlib.pyplot as plt
import seaborn as sns

# GET THE WORKING DIRECTORY
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

# === DATA LOADING AND CLEANING ===
# Load salary data from CSV, only keep 'annual_base_pay' column
df = pd.read_csv("salaries_clean.csv", usecols=['annual_base_pay'])

# Mkake sure all values are numeric and drop missing or non-numeric entries
df['annual_base_pay'] = pd.to_numeric(df['annual_base_pay'], errors='coerce')
df = df.dropna()

# Filter out extreme outliers above $500,000
df = df[df['annual_base_pay'] <= 500_000]
print(df)

# === DESCRIPTIVE STATISTICS ===
# Compute mean (average)
n = len(df)
total_sum = df['annual_base_pay'].sum()
mean = total_sum / n
print("Mean / Average is: " + str(mean))

# Prepare sorted list of salaries for median and other statistics
salary_list = sorted(df['annual_base_pay'].tolist())
n = len(salary_list)

# Compute median (middle value)
if n % 2 == 0:
    median1 = salary_list[n // 2]
    median2 = salary_list[n // 2 - 1]
    median = (median1 + median2) / 2
else:
    median = salary_list[n // 2]
print("Median is:", median)


# Compute variance
def get_variance(data):
    return sum((x - mean) ** 2 for x in data) / len(data)


variance = get_variance(salary_list)
print("Variance is:", variance)

# Compute standard deviation using pandas built-in method
std_dev = df['annual_base_pay'].std()
print("Standard Deviation is:", std_dev)

# Compute standard error with std_dev
std_error = std_dev / math.sqrt(n)
print("Standard Error is:", std_error)


# === MEAN CONFIDENCE INTERVAL FUNCTION ===
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)  # sem = standard error of the mean
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)  # margin of error
    print(m, m - h, m + h)  # print mean, lower, upper

print("Mean confidence interval is:")
mean_confidence_interval(salary_list, 0.95)

# === VARIANCE CONFIDENCE INTERVAL FUNCTION ===
confidence_level = 0.95


def variance_confidence_interval(data, confidence=0.95):
    var = get_variance(data)

    var_err = math.sqrt((2 * var) / (n - 1))

    marg_err = 1.96 * var_err

    ci_lower_bound = var - marg_err
    ci_upper_bound = var + marg_err

    print(f"{ci_lower_bound=}, {ci_upper_bound}")

print("Variance confidence interval is:")
variance_confidence_interval(salary_list, 0.95)


# === SAMPLE SIZE ESTIMATION FUNCTION ===
def sampleSize(
        population_size,
        margin_error=0.01,
        confidence_level=0.95,
        sigma=1 / 2
):
    # Estimate required sample size for a given population size, margin of error,
    # confidence level, and assumed population standard deviation (sigma).

    alpha = 1 - (confidence_level)
    zdict = {
        0.90: 1.645,
        0.91: 1.695,
        0.99: 2.576,
        0.97: 2.17,
        0.94: 1.881,
        0.93: 1.812,
        0.95: 1.96,
        0.98: 2.326,
        0.96: 2.054,
        0.92: 1.751
    }

    # Look up z-score for confidence level
    if confidence_level in zdict:
        z = zdict[confidence_level]
    else:
        from scipy.stats import norm
        z = norm.ppf(1 - (alpha / 2))

    N = population_size
    M = margin_error
    numerator = z ** 2 * sigma ** 2 * (N / (N - 1))
    denom = M ** 2 + ((z ** 2 * sigma ** 2) / (N - 1))
    return numerator / denom
print("Sample size is: ")
print(sampleSize(len(salary_list)))

# === HYPOTHESIS TESTING ===
# Define null and alternative hypotheses
H0 = "The average annual base salary is 100000 dollars."
H1 = "The average annual base salary is greater than 100000 dollars."

# Perform one-sample t-test
t_stat, p_value_two_tailed = scipy.stats.ttest_1samp(salary_list, 100000)

# Since this is a one-tailed test (greater than), adjust the p-value
if t_stat > 0:
    p_value = p_value_two_tailed / 2
else:
    p_value = 1.0  # Not in the direction of H1, so do not reject H0

# Display results
print("H0:", H0)
print("H1:", H1)
print("Test statistic:", t_stat)
print("p-value:", p_value)

# Make decision based on significance level (a = 0.05)
if p_value < 0.05:
    print("Reject the null hypothesis.")
    print("Conclusion: There is significant evidence that the average salary is greater than $100,000.")
else:
    print("Fail to reject the null hypothesis.")
    print("Conclusion: Not enough evidence to say the average salary is greater than $100,000.")
#== BOXPLOT ==
# Load filtered salary data
filt_salary = pd.read_csv("salaries_filtered.csv")

# Set style
sns.set_theme(style="whitegrid")

# Create plot
fig, ax = plt.subplots(figsize=(14, 5))  # Wider and taller
sns.boxplot(x=filt_salary['annual_base_pay'], ax=ax, color='skyblue', width=0.4, fliersize=5)

# Titles and labels
ax.set_title('Boxplot with outliers', fontsize=16)
ax.set_xlabel('Annual base pay', fontsize=12)

screen = 0

# Grid and tight layout to avoid cropping
plt.grid(True)
# Make the edges not touch the screen
plt.tight_layout(pad=2.0)
# Make sure it doesn't crop
plt.savefig("boxplot.png", dpi=300, bbox_inches='tight')
plt.show()
# == HISTOGRAM ==
# Create histogram
plt.hist(salary_list)
# Set x axis label
plt.xlabel('Salary')
# Set Y axis label
plt.ylabel('Number of occurrences')
plt.show()