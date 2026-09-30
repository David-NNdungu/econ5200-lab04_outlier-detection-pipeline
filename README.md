# Outlier Detection on California Housing

## Objective

In this lab, I built and compared three methods for finding unusual observations in the California Housing dataset.

## Methodology

- Fixed three bugs in an outlier-detection pipeline.
- Corrected the modified Z-score so it uses the median and MAD.
- Changed the Tukey fence multiplier from 1.0 to 1.5.
- Changed the Isolation Forest contamination setting from 0.50 to 0.05.
- Used an `OutlierDetector` class to run all three methods.
- Compared the overlap between the methods with a Venn diagram.
- Built an interactive explorer for changing the settings and reviewing flagged rows.

## Key Findings

The modified Z-score flagged 400 observations based on median income, while Tukey fences flagged 681. Isolation Forest used all nine numeric columns and flagged 1,032 rows. All three methods agreed on 322 rows.

The methods do not give the same results because the modified Z-score and Tukey fences only examine one variable, while Isolation Forest looks for unusual combinations across several variables. I would use Tukey fences as the main method because the results are easier to explain, then use Isolation Forest as a second check. A flagged observation should be reviewed before it is removed because it may be a real extreme value rather than a data error.
