---
layout: post
title: Exploratory Statistical Analysis
date: 2021-06-05
category: Data Science
tags: 
- data-science
redirect_from:
- /datascience/2021/06/05/exploratory-analysis/
- /exploratory-analysis.html
---

{% highlight python %}
{% endhighlight %}

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

The process of analysing data so as to take out some important characteristics from it, is known as **exploratory data analysis**. Performing data analysis through statistical methods like `mode`, `probability`, `expected value`, `correlation` etc is known as **exploratory statistical analysis.**

### **Exploring Binary and Categorical Data**

**Binary Data:** Data which can take only two possible values, 0 or 1.

**Categorical Data:** Data which has been divided into categories or groups according to their features.

#### **Mode**

* `Mode` is the value that appear most offten in dataset.
* It is a summay statistic for **categorical data** i.e. grouped data or set of values representing a possible categories.
* It is not used for **numeric data** i.e. data expressed in number scale.

The function to calculate model:

```python
def find_mode(counts, n):

  """Finds model of sample.

  This function consists of two method to calculate:
  * without library function
  * via library function

  Parameters
  ----------
  counts : list
           number of entries each emotion type has

  n : str
      select method to calucate iqr range

  Example
  -------
  >>> for n = 0, Time taken to calculate:  0.0002598762512207031, Mode of sample:  [3, 8989]

  Returns
  -------
  mode : list
         emotion along with its number of values

"""

  try:

    n = int(n)

    print("Shape of selected numpy array: ", counts.shape)
    print("Data type of _data_numpy: ", type(counts))

    if n is 0:

      start = time.time()
      max = counts[0]
      num = 0
      for row in counts:
        num +=1
        if row > max:
          mode = num-1
          max = row

      list = [mode, max]

      print("Time taken to calculate: ", time.time() - start)
      return list

    if n is 1:
   
      start = time.time()
      mode = counts.mode(dropna=True)
      print("Time taken to calculate: ", time.time() - start)
      return mode

    if n is not 0 or n is not 1:
      print("Enter correct value")
  
  except StatisticsError as error:
    raise error
  
  except Exception as error:
    print(error)
    raise error
```

Function Call:

```python
n = input("Enter 0 (To find mode without any library function), Enter 1 (Via using Library Function): ")
mode = find_mode(counts, n)
print("Mode of sample: ", mode)
```

#### **Probability**

* `Probability` of an event will happen is how likly the event occurs again and again if tested again and again.
* **Formula:**

$$ \begin{equation} \textrm{Probability} = \frac{Number of Favourable Outcomes}{Total Number of Favourable Outcomes}  \end{equation} $$

The function to calculate probability:

```python
def find_prob(counts):
  
  """To find probability of emotion type to be predicted.

  Parameters
  ----------
  counts : list
           number of entries each emotion type has

  Example
  -------
  >>> Time taken to calculate:  2.6702880859375e-05, Probability of each emotion type:  [0.13801655195474685, 0.01524228829381113, 0.14269791289324826, 0.25048067545350683, 0.1693370858528158, 0.11151670521358709, 0.17270878033828405]
  
  Returns
  -------
  prob : list
         probability of ouccurance of each emotion.
         
  """

  try:

    print("Shape of selected numpy array: ", counts.shape)
    print("Data type of _data_numpy: ", type(counts))

    start = time.time()
    sum = 0
    for row in counts:
      sum += row
    probability = []
    for row in counts:
      p = row/sum
      probability.append(p)
    print("Time taken to calculate: ", time.time() - start)
    return probability
  
  except StatisticsError as error:
    raise error
  
  except Exception as error:
    print(error)
    raise error
    
```

Function Call:

```python
prob = find_prob(counts)
print("Probability of each emotion type: ", prob)
```

#### **Expected Value**

* `Expected Value` is calculated by:

   i) Multiply each outcome by its probability of occurrence.

   ii) Sum these values.
* It is a form of `weighted mean`.
* This concept is basically used for **categorical data**.
* **Formula:**

![ev](https://user-images.githubusercontent.com/62256509/120922692-39d67380-c6e8-11eb-834d-79af7e6d9d80.png)


Here, $$x_1$$, $$x_2$$, ......, $$x_i$4 are data and f($$x_i$$) are the probability of outcomes and E[X]: Represents Expected Value

The function to calculate expected value:

```python
def find_ev(counts, prob):

  """Finds expected value.

  Parameters
  ----------
  counts : list
           number of entries each emotion has

  prob : list
         probability of each emotion

  Example 
  -------
  >>> Time taken to calculate:  3.528594970703125e-05
  >>> Expected Value of each emotion type:  [683.5959818318612, 8.337531696714688, 730.7560119263244, 2251.570791651573, 1029.0614707275615, 446.2898542647755, 1070.4490205366844]
  >>> Total Expected value:  6220.060662635495

  Returns
  -------
  ev : list
       expected value of each emotion 

  total_ev : float
             expected value of comeplete sample

  """

  try:

    print("Shape of selected numpy array: ", counts.shape)
    print("Data type of _data_numpy: ", type(counts))

    start = time.time()

    ev = []
    num = 0
    for row in counts:
      val = row * prob[num]
      num += 1
      ev.append(val)

    total_ev = 0
    for row in ev:
      total_ev += row
    
    print("Time taken to calculate: ", time.time() - start)
    return ev, total_ev

  except StatisticsError as error:
    raise error
  
  except Exception as error
    print(error)
    raise error
    
```

Function Call:

```python
ev, total_ev = find_ev(counts, prob)
print("Expected Value of each emotion type: ", ev)
print("Total Expected value: ", total_ev)
```

### **Correlation**

#### **Definition** 

`Correlation` is the measure which is used to know how similar the two variables are.

#### **Correlation Coefficient**

* It is used to measure the strength of correlation between two variables.
* **Range:** 1 to -1
* Values above 1 or less than -1 are concluded as error in calculation.
* `Negative Correlation` indicates that if one variable increases the other decreases or vice versa.
* `Positive Correlation` indicates that both are similar i.e. if one increases the other one also increases and vice versa.
* `0` correlation means that the samples shows no relation between each other.

![corr](https://user-images.githubusercontent.com/62256509/120922834-03e5bf00-c6e9-11eb-942a-dc799b5e7349.png)


#### **Correlation Matrix**

A table where variables are shown as rows and columns and the cell values are the correlations between the variables is known as `Correlation Matrix`.

The function to calculate `correlation matrix`:

```python
def corr_matrix(df_all):

  """To find correlation matrix.

  Parameters
  ----------
  df_all : dataframe
           
  Example
  -------
  >>> Time taken to calculate:  356.91664814949036

  Returns
  -------
  corr : pandas.core.frame.DataFrame
         correlation matrix
  """

  try:
    start = time.time()
    corr = df_all.corr() 
    print("Time taken to calculate: ", time.time() - start)
    return corr

  except StatisticsError as error:
    raise error
  
  except Exception as error:
    print()
    print(error)
    raise error
```

Function Call:

```python
corr = corr_matrix(df_all)
print(corr)
```

Output:

![corr-matrix](https://user-images.githubusercontent.com/62256509/120923201-eb76a400-c6ea-11eb-86c2-50360b580cdd.png)

#### **Correlation Plot**

```python
def corr_plot(corr, cmap_ar):

  """To plot correlation.

  Parameters
  ----------
  corr : dataframe
         of correlation matrix
  
  cmap_ar : str
            type of heatmap to be plotted
  
  Example
  -------
  >>> Enter the cmap style from the above arrguments: 7
  >>> Color plotted:  Blues_r   
  >>> Time taken to plot:  6.074030876159668

  Returns
  -------
  Graph

  """

  try:
    start = time.time()
    plt.figure(figsize=(16, 16))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap=cmap_ar) 
    sns.set(font_scale=2,style='white')
    plt.tight_layout()
    
    plt.title('Heatmap correlation')
    plt.show()
    print("Time taken to plot: ", time.time() - start)

  except AttributeError as error:
    print("Attribute Error Occured.")
    print("The error is ", error)
    
  except ValueError as error:
    print("Value Error Occured.")
    print("The error is ", error)
    
```

Function Call:

```python
val = ['coolwarm', sns.diverging_palette(20, 220, as_cmap=True), 'Blues', 'YlGnBu', 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']

n = input("Enter the cmap style from the above arrguments: ")
n = int(n)
cmap_ar = val[n]
print("Color plotted: ", val[n])

corr_plot(corr, cmap_ar)

```

Output:

![corr-graph](https://user-images.githubusercontent.com/62256509/120923443-3218ce00-c6ec-11eb-9677-b4694337256a.png)


#### **Pearson’s Correlation Coefficient**

Steps to calculate:

* Calculate covariance of the given two varaibles.
* Calculate standard deviation of both variables.
* Put in formula.

**Formula:**

$$ \begin{equation} r = \frac{\sum^n_{i=1}(x_i-\bar{x})(y_i-\bar{y})}{(n-1)s_xs_y} \end{equation} $$

Check out my other posts related to topic:

* [Descriptive Statistical Analysis](https://khushi-411.github.io/datascience/descriptive-stat-analysis/) 
* [Inferential Statistical Analysis](https://khushi-411.github.io/datascience/inferential-stat-analysis/)

Do visit my [GitHub](https://github.com/khushi-411/emotion-recognition/tree/main/data-science) to view complete code! 
