---
layout: post
title: Inferential Statistical Analysis
date: 2021-06-06
category: Data Science
tags: 
- data-science
redirect_from:
- /datascience/2021/06/06/inferential-stat-analysis/
- /inferential-stat-analysis.html
---

### **T-Test**

* The `T-test` i.e. `Student’s T Test` compares two **means** and tells us if they are different from each other.
* It tells whether two samples have been drawn from same sample or not.
* It tell's how significant our result is, more specifically it tells whether it happened by chance or not.

### **T-Values and Degrees of Freedom**

* `T-Value` is the ratio of the difference between the mean of the two sample sets and the variation that exists within the sample sets.
* T-Value is also called `T-Score`.
* Large t-score indicates that the groups are different.
* Small t-score indicates that the groups are similar.
* `Degree Of Freedom` are the values that has a freedom to vary.
* **Formula of Degree of Freedom:** df = n<sub>x</sub> + n<sub>y</sub> - 2

### **Normal Distribution**

* It has a `bell-shaped` density curve.
* The density curve is symmetrical and centered about `mean(μ)`. It determines the **peak** of the curve.
* Data spread it determined by `standard deviation(sigma-σ)` i.e. it is a measure of **variability**. It determines how far the data falls from the mean.

The density curve is as follows: 

![normal-dis](https://user-images.githubusercontent.com/62256509/120937902-d53e0780-c72d-11eb-8e42-a67048ba670e.png)

### **Standardization (Normalization, z-Scores)**

* The process of putting different variables to a same scale is known as `Standardization`. 
* Also called `Normalization`.
* It allows us to compare scores between different types of variables.
* **Formula:**

![z-form](https://user-images.githubusercontent.com/62256509/120937924-028ab580-c72e-11eb-9b97-6e75a8b400fe.png)


* Result of this formula is known as `z-score`.
* `z-score` tells the overall data lies compared to overall population.

> <font color='green'><b> Note:</b></font> The higher (or lower) the Z-score, the more unlikely the result is to happen by chance and the more likely the result is meaningful.

### **p-Value and alpha**

##### **Significance Level-Alpha**

* It is a probability of rejecting the null hypothesis when it is true.

* Drawing a two tailed graph for alpha = 0.05 and alpha = 0.01:<sup>[1]</sup>

**Some Keypoints:**
* We need to shade 5% or 1% of graph that is furthest from null hypothesis(since we are rejecting).
* The sample mean for the given distribution is 330.6.

![two](https://user-images.githubusercontent.com/62256509/120937994-454c8d80-c72e-11eb-86f9-874567cad691.png)

![one](https://user-images.githubusercontent.com/62256509/120938005-55646d00-c72e-11eb-8cd8-95745d983781.png)

**Observation:**

* For the above two-tailed test, the critical region (the shaded part) lies equidistant from the null hypothesis value.
* Sample mean 330.6 is significant in case for 5% significance but not for the case of 1% significance level.

**Conclusion**

* We'll reject null hypothesis for 5% significance level.
* Fail to reject null hypothesis for 1% significance level.

### **P-Value**

* The probability that the results from our sample data occurred by chance is known as `p-value`.
* Lower p-values are good because they indicate that the data does not occur by chance.
* *Example:* p-value = 0.01 indicates that their is only 1% probability that the data occured by chance.
* If the observed p-value is less than alpha, then the results are statistically significant.

### **Type's Of T-Test**

#### **Student's T-Test**

* Sample's are Independent.
* Sample's are drawn from Gaussian Distribution.
* Size of each sample must be same.
* Sample's have same variance.
* Sample's have different mean.
* Values of one sample does not have any effect to values of other sample.

**Formula:**

![ttest](https://user-images.githubusercontent.com/62256509/120938151-1f73b880-c72f-11eb-9fa6-113e6ab43e1e.png)

#### **Paired Student's T-Test**

* Sample's are Dependent.
* They may be from same population.
* Used to check whether the difference of means of two samples are zero or not.
* They have unequal variance.
* Similar to Student's T-Test they are also drawn from Gaussian Distribution.
* Values of one sample effect the values of other sample.

**Formula:**

![d](https://user-images.githubusercontent.com/62256509/120938310-acb70d00-c72f-11eb-955f-05f7c2e48ad1.jpg)

### **Hypothesis**

* Hypothesis testing is used to assess the probability of a hypothesis by using sample data.
* An assumption is made looking into population and test are preformed according to it.

#### **Null Hypothesis**

* It stats/assumnes that their is no difference between population characteristics (mean, propotion).
* It is denoted by H<sub>0</sub>.

#### **Alternative Hypothesis**

* It claims that the population is contradictory to null hypothesis. Hence, reject null hypothesis.
* It is denoted by H<sub>1</sub>.

### **Procedure**

* Determine a null and alternate hypothesis.
* Collect sample data.
* Determine a confidence interval and degrees of freedom.
* Calculate the t-statistic.
* Calculate the critical t-value from the t distribution.
* Compare the critical t-values with the calculated t statistic.

### **Creating Samples**

```python
_emotion_type = input("Enter type of emotion which needs to be tested: ")
_data, _data_numpy, _data_label, _data_label_numpy = select_emotion(df_all, df_label, df, _emotion_type)
sample_1 = _data_numpy.copy()
sample_1
```

```python
_emotion_type = input("Enter type of emotion which needs to be tested: ")
_data, _data_numpy, _data_label, _data_label_numpy = select_emotion(df_all, df_label, df, _emotion_type)
sample_2 = _data_numpy.copy()
sample_2
```

### **T-Test Implementation**

Creating some user defined function for Dependent Samples

```python
"""
    To find sum squared difference and sum difference between observations.
"""

def find_diff(sample1, sample2):
  sq_diff = sum([(sample1[i] - sample2[i]) ** 2 for i in range(len(sample1))])
  diff = sum([sample1[i] - sample2[i] for i in range(len(sample1))])
  return sq_diff, diff

"""
    To find standard deviation.
"""

def find_dev(sq_diff, diff, size):
  std = np.sqrt((sq_diff - (diff ** 2 / size)) / (size - 1))
  return std
  
"""
    To calculate t-statistic.
"""

def dep_ttest(sample1, sample2, n, sample1_mean, sample2_mean, size):
  sq_diff, diff = find_diff(sample1, sample2)
  std_dev = find_dev(sq_diff, diff, size)
  std_error = std_dev / np.sqrt(size)
  t_stat = (sample1_mean - sample2_mean) / std_error
  return t_stat
  
"""
    To calculate p-value, compare with critical t-value

"""

def dep_pval(sample1, sample2, size, t_statistic):
  # Degree of freedom.
  df = size - 1
  # p-value after comparision with the t-stat
  p = 1 - sci.stats.t.cdf(t_statistic, df = df)
  pval = 2 * p
  return pval
```

Creating some user defined function for Independent Samples

```python
"""
    To calculate t-statistic for independent samples.
"""

def ind_ttest(sample1, sample2, n, sample1_mean, sample2_mean, size1, size2):

  var_1, std_1 = find_var_std(sample1, n)
  var_2, std_2 = find_var_std(sample2, n)

  print("Variance of sample1:", var_1)
  print("Variance of sample2:", var_2)
  print("Standard Deviation of sample1: ", std_1)
  print("Standard Deviation of sample2: ", std_2)

  t_stat = (sample1_mean - sample2_mean)/(np.sqrt(np.sum(np.power(std_1, 2)/size1), np.power(std_2, 2)/size2))

  return t_stat
  
"""
    To calculate p-value, compare with critical t-value
"""

def ind_pval(sample1, sample2, t_statistic, size1, size2):
  # Degree of freedom.
  df = size1 + size2 - 2
  # p-value after comparision with the t-stat
  p = 1 - sci.stats.t.cdf(t_statistic, df = df)
  pval = 2 * p
  return pval
```

### **Function for T-Test**

```python
def t_test(sample_1, sample_2, alpha, n, sample_type, sample_1_mean, sample_2_mean, size1, size2):

  try:

    n = int(n)
    alpha = float(alpha)

    if n is 1:

      if sample_type is 0:

        start = time.time()

        statistic, pvalue = sci.stats.ttest_rel(sample1, sample2)

        print("Statistics: ", statistic)
        print("P-Value: ", pvalue)
        print("Same Distributions- fails to Reject H0") if pvalue.any() > alpha else print("Different Distributions- Reject H0")

        print("Time taken to formulate: ", time.time() - start)

      if sample_type is 1:

        start = time.time()

        statistic, pvalue = sci.stats.ttest_ind(sample1, sample2)

        print("Statistics: ", statistic)
        print("P-Value: ", pvalue)
        print("Same Distributions- fails to Reject H0") if pvalue.any() > alpha else print("Different Distributions- Reject H0")

        print("Time taken to formulate: ", time.time() - start)

      if sample_type is not 0 or sample_type is not 1:
        print("Enter correct sample type")

    elif n is 0:

      if sample_type is 0:

        start = time.time()
        
        t_stat = dep_ttest(sample_1, sample_2, sample_1_mean, sample_2_mean, n, size1)
        pvalue = dep_pval(sample_1, sample_2, size1, t_stat)

        print("Statistics: ", t_stat)
        print("P-Value: ", pvalue)
        print("Same Distributions- fails to Reject H0") if pvalue.any() > alpha else print("Different Distributions- Reject H0")
        
        print("Time taken to formulate: ", time.time() - start)

      if sample_type is 1:

        start = time.time()

        t_stat = ind_ttest(sample_1, sample_2, n, sample_1_mean, sample_2_mean, size1, size2)
        pvalue = ind_pval(sample_1, sample_2, t_statistic, size1, size2)

        print("Statistics: ", t_stat)
        print("P-Value: ", pvalue)

        print("Same Distributions- fails to Reject H0") if pvalue.any() > alpha else print("Different Distributions- Reject H0")

        print("Time taken to formulate: ", time.time() - start)


    elif n is not 0 or n is not 1:
      print("Enter correct value")

  except StatisticsError as error:
    raise error
  
  except (FloatingPointError, NameError, ZeroDivisionError, ValueError, TypeError, AttributeError) as error:
    print()
    print(error)
    raise error
```

**Null Hypothesis (H<sub>0</sub>):** Sample's are independent if means of sample are same.

**Alternate Hypothesis (H<sub>1</sub>):** Samples are independent if means of sample are not same.

```python
n = input("Enter 0 (To calculate without library functions), 1 (Via Library function): ")
alpha = input("Enter alpha value: ")

size_sample_1 = sample_1.shape[0] * sample_1.shape[1]
size_sample_2 = sample_2.shape[0] * sample_2.shape[1]

sample_1_mean = find_mean(sample_1, n)
sample_2_mean = find_mean(sample_1, n)

print("Mean of Sample 1:", sample_1_mean)
print("Mean of Sample 2:", sample_2_mean)


if size_sample_1 != size_sample_2:
  sample_type = 1

if sample_1_mean != sample_2_mean:
  sample_type = 1
else:
  sample_type = 0

t_test(sample_1, sample_2, alpha, n, sample_type, sample_1_mean, sample_2_mean, size_sample_1, size_sample_2)
```

Check out my other post:

* [Descriptive Statistical Analysis](https://khushi-411.github.io/datascience/descriptive-stat-analysis/) 
* [Exploratory Data Analysis](https://khushi-411.github.io/datascience/exploratory-analysis/)

 Do visit my [GitHub](https://github.com/khushi-411/emotion-recognition/tree/main/data-science) to view complete code! 
