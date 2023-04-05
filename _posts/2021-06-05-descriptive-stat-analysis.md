---
layout: post
title: Descriptive Statistical Analysis
date: 2021-06-05
category: Data Science
tags: 
- data-science
redirect_from:
- /datascience/2021/06/05/descriptive-stat-analysis/
- /descriptive-stat-analysis.html
---

{% highlight python %}
{% endhighlight %}

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

This statistical data analysis methods is used when we want to know about the features of data i.e what all things are present in the distribution

### **Dataset Description & Basic Analysis**

I used [fer2013](https://www.kaggle.com/deadskull7/fer2013) dataset to perform data analysis. The dataset consists of 48x48 pixel grayscale images of faces. The dataset is categorized into seven groups each of different type of emotions as {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}. 

On reading the dataset:

![fer2013-data](https://user-images.githubusercontent.com/62256509/120928488-731bdd00-c702-11eb-8e5a-901cc4bd8c5d.png)

I performed some data preprocessing and converted that dataset into:

![fer2013-pre-data](https://user-images.githubusercontent.com/62256509/120928559-c130e080-c702-11eb-9e3f-aab33e72d28f.png)

#### **Data Labels**

```emotion_label = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}```

#### **Useful Variable Creation**

```python
"""
   Converting dataframe into numpy array.
"""

df_all = df.iloc[:,:2304]
df_all_numpy = df_all.to_numpy()

df_label = df["emotion"]
df_label_numpy = df_label.to_numpy()

```

#### **Emotion Selection**

```python
def select_emotion(df_all, df_label, df, _emotion):

  """Select emotion type among all.

    Parameters
    ----------
    df_all : data-frame of shape (35887, 2304) 
             columns consists of pixel values (emotion column removed)

    df_label : consists of emotion label

    df : complete data-frame of shape (35887, 2305)

    _emotion : emotion type selected by the user

    Returns
    -------
    _data : all the data consisting the given emotion

    _data_numpy : numpy array to _data

    _data_label : consist all the data labels, of provided emotion 

    _data_label_numpy : numpy array of data label
    """


  try:

    _emotion = int(_emotion)

    _data = df_all[df["emotion"] == _emotion]
    _data_numpy = _data.to_numpy()

    _data_label = df_label[df["emotion"] == _emotion]
    _data_label_numpy = _data_label.to_numpy()

    return _data, _data_numpy, _data_label, _data_label_numpy

  except Exception as error:
    raise error

```


This blog post is about Statistical Data Analysis, so I have presented few necessary topics.



### **Estimates Of Location**

* To find the value that best describes the data.
* Helps to estimate location parameter for the distribution.

#### **Mean**

* `Mean` is sum of values divided by total number of values.
* It is also known as `Average`.

**Formula:**

To compute the mean for a set of $$ n $$ values $$ x_1, x_2, ..., x_n $$ is:

$$ \begin{equation} \textrm{Mean} = \bar{x} = \frac{\sum ^n_{i=1} x_i}{n} \tag{i} \end{equation} $$

Here, $$\bar{x}$$ : Represents mean of a sample from population.

The function to calculate mean:

```python
def find_mean(_data_numpy, n):

  """Finds mean of sample.

  This method calculates mean in three different ways:
  * without using library function
  * using numpy np.sum() function
  * via direct library function

  Parameters
  ----------
  _data_numpy : ndarray 
                All the pixel data of selected emotion

  n : str
      user difined, how to calculate mean

  Example
  -------
  >>> for _emotion_type = 3 (happiness)
  >>> n = 0, Time taken to calculate:  10.261376142501831, Mean of sample:  129.08117932140826
  >>> n = 1, Time taken to calculate:  0.021773815155029297, Mean of sample:  129.08117932140826
  >>> n = 2, Time taken to calculate:  0.030038833618164062, Mean of sample:  129.08117932140826

  Results
  -------
  mean : float
        mean of sample
  
  """

  try:

    print("Shape of selected numpy array: ", _data_numpy.shape)

    n = int(n)

    if n is 0:
      start = time.time()
      sum = 0
      for i in range(_data_numpy.shape[0]):
        for j in range(_data_numpy.shape[1]):
          sum += _data_numpy[i][j]
      mean = float(sum/(_data_numpy.shape[0]*_data_numpy.shape[1]))
      print("Time taken to calculate: ", time.time() - start)
      return mean

    elif n is 1:
      start = time.time()
      sum = np.sum(_data_numpy)
      mean = float(sum/(_data_numpy.shape[0]*_data_numpy.shape[1]))
      print("Time taken to calculate: ", time.time() - start)
      return mean

    elif n is 2:
      start = time.time()
      mean = _data_numpy.mean()
      print("Time taken to calculate: ", time.time() - start)
      return mean
    
    if n is not 0 or n is not 1 or n is not 2:
      print("Enter correct value")

  except StatisticsError as error:
    raise error
    
  except Exception as error:
    print(error)
    raise error

```

Function call:

```python
n = input("Enter 0 (To find mean without any library function), Enter 1 (To find via using np.sum()), Enter 2 (Via using Library Function): ")
_emotion_type = input("Enter the emotion type whose mean of pixel values you want to find: ")
_data, _data_numpy, _data_label, _data_label_numpy = select_emotion(df_all, df_label, df, _emotion_type)

mean = find_mean(_data_numpy, n)
print("Mean of sample: ", mean)
```

#### **Trimmed Mean**

* `Trimmed Mean` is the average of all values after dropping fixed number of extreme values (sorted values).
* Also known as `Truncated Mean`.
* Widely used to avoid the influence of outliers.
* It is a trade-off between the median and the mean.

**Formula:**

Here, $$x_1 , x_2 , ..., x_n$$ represents the sorted values, where $$x_1$$ is the smallest and $$x_n$$ is the largest value.

$$ \begin{equation} \textrm{Trimmed Mean} = \bar{x} = \frac{\sum ^{n-p}_{i=p+1} x_{(i)}}{n-2p} \tag{ii} \end{equation} $$

> <font color = "green"> <b> <i> Note: </i> </b></font> `2p` represent the omitted smallest and largest values.

The function to calculate `Trimmed Mean`:

```python
def find_trimmed_mean(_data_numpy, n, p):

  """To find the trimmed mean of the sample.

  This method finds trimmed mean using two ways.
  * without using library function
  * using library function

  Parameters
  ----------
  _data_numpy : ndarray
                numpy array of selected sample

  n : str
      select type of method to calucate trimmed mean

  p : str
      fraction of sample we want to omit from both side

  Example
  -------
  >>> for _emotion_type = 3, p = 0.05
  >>> n = 0, Time taken to calculate:  0.13057208061218262, Trimmed Mean of sample:  129.47220185935402
  >>> n = 1, Time taken to calculate:  0.1196749210357666, Trimmed Mean of sample:  129.47220185935402

  Results
  -------
  trimmed_mean : float
                 trimmed mean of sample

  Note
  ----
  This function 1st sorts the numpy array, then calculate

  """

  try:

    print("Shape of selected numpy array: ", _data_numpy.shape)

    n = int(n)
    p = float(p)
    _data_numpy = np.asarray(_data_numpy)
    start_time = time.time()
    _data_numpy = np.sort(_data_numpy).ravel()
    print("Time taken to sort the array: ", time.time() - start_time)

    if n is 0:
      start = time.time()
      num = _data_numpy.shape[0]
      lower_val = int(p * num)
      upper_val = num - lower_val

      if (lower_val > upper_val):
        raise ValueError("Proportion too big.")

      atmp = np.partition(_data_numpy, (lower_val, upper_val - 1), 0)
      sl = [slice(None)] * atmp.ndim
      sl[0] = slice(lower_val, upper_val)
      trim_mean = np.mean(atmp[tuple(sl)], axis=0)

      print("Time taken to calculate: ", time.time() - start)
      return trim_mean

    elif n is 1:
      start = time.time()
      trim_mean = stats.trim_mean(_data_numpy, p)
      print("Time taken to calculate: ", time.time() - start)
      return trim_mean
    
    if n is not 0 or n is not 1:
      print("Enter correct value")

  except StatisticsError as error:
    raise error
    
  except Exception as error:
    print(error)
    raise error
```

Function call:

```python
n = input("Enter 0 (To find trimmed mean without any library function), Enter 1 (Via using Library Function): ")
_emotion_type = input("Enter the type of emotion whose trimmed mean of pixel values you want to find: ")
p = input("Enter fraction of values you want to omit: ")
_data, _data_numpy, _data_label, _data_label_numpy = select_emotion(df_all, df_label, df, _emotion_type)

trim_mean = find_trimmed_mean(_data_numpy, n, p)
print("Trimmed Mean of sample: ", trim_mean)
```

#### **Median**

*  `Median` is the middle number on a sorted list of the data.
* It depends only on values in center of the data.

**Formula:**

![median](https://user-images.githubusercontent.com/62256509/120903438-b4a97b00-c663-11eb-80ef-0d5b4c0ed6a3.png)

The function to calculate median:

```python
def find_median(_data_numpy, n):

  """Finds median of sample

  Using two methods:
  * without library function
  * using library function

  It 1st checks the number of samples, then uses the formula accordingly.

  Parameters
  ----------
  _data_numpy : ndarray
                numpy array of all the data points of selected sample

  n : str
      method to calcuate

  Results
  -------
  median : float
           median of sample

  Example
  -------
  >>> for _emotion_type = 3
  >>> n = 0, Number of terms is even, Time taken to calculate:  0.00019073486328125, Median of sample:  77.0
  >>> n = 1, Time taken to calculate:  0.1251697540283203, Median of sample:  134.0

  Note
  ----
  This function 1st sorts the numpy array, then calculate
  """

  try:

    print("Shape of selected numpy array: ", _data_numpy.shape)

    n = int(n)
    _sort_time = time.time()
    _data_numpy = np.sort(_data_numpy).ravel()                # to flaten array
    print("Time taken to sort the array: ", time.time() - _sort_time)
    print("Data type of _data_numpy: ", type(_data_numpy))
    print(_data_numpy.shape)
   
    if n is 0:
      start = time.time()

      num = _data_numpy.size#.shape[0]*_data_numpy.shape[1]
      print(num)
      
      if num%2 is 0:
        print("Number of terms is even")
        ob_num = int(num/2)
        #median = np.where(_data_numpy == )
        median = _data_numpy[ob_num]
        print("Time taken to calculate: ", time.time() - start)
        return median

      elif num%2 is 1:
        print("Number of terms is odd")
        ob_num_1 = int((num-1)/2)
        observation_1 = _data_numpy[ob_num_1]
        ob_num_2 = int((num+1)/2)
        observation_2 = _data_numpy[ob_num_2]
        median = (observation_1 + observation_2)/2

        print("Time taken to calculate: ", time.time() - start)
        return median

    elif n is 1:
      start = time.time()
      median = np.median(_data_numpy)
      print("Time taken to calculate: ", time.time() - start)
      return median
    
    if n is not 0 or n is not 1:
      print("Enter correct value")

  except StatisticsError as error:
    raise error
    
  except Exception as error:
    print(error)
    raise error
```

Function call:

```python
n = input("Enter 0 (To find median without any library function), Enter 1 (Via using Library Function): ")
_emotion_type = input("Enter the type of emotion whose median of pixel values you want to find: ")
_data, _data_numpy, _data_label, _data_label_numpy = select_emotion(df_all, df_label, df, _emotion_type)

median = find_median(_data_numpy, n)
print("Median of sample: ", median)
```

### **Estimates Of Variability**

* `Variability` between samples refers to `range` of values differs between samples.
* Also refered as `Dispersion`.
* It measures whether data values are `tighly clustered` or `spread`.

#### **Mean Absolute Deviation**

* `Deviations` tell us how dispersed the data is around the central value.
* `MAD` is the mean of absolute deviation of data points from mean.

**Formula**

$$ \begin{equation} \textrm{Mean absolute deviation} = \frac{\sum^n_{i=1} |x_i-\bar{x}|}{n} \tag{iv}\\ \end{equation} $$

where, $$\bar{x}$$ is the sample mean, n is total number of observations and $$x_1$$, $$x_2$$, .......$$x_n$$ are data points.

The function to call `MAD`:

```python
def find_mad(_data):
  """Finds mean absolute deviation of sample.

  Parameters
  ----------
  _data : dataframe of given sample or population

  Results
  -------
  mad_row : Series
            Computes MAD value accross each row in provided dataframe
  """

  try:

    print("Shape of selected numpy array: ", _data.shape)
    print("Data type of _data: ", type(_data))
   
    start = time.time()
    #mad_col = _data_numpy.mad()
    mad_row = _data.mad(axis=1, skipna=False)
    print("Time taken to calculate: ", time.time() - start)
    return mad_row
  
  except StatisticsError as error:
    raise error

  except Exception as error:
    print(error)
    raise error
```

To calculate mean absolute deviation for complete data frmae:

```python
"""
    Printing Mean Absolute Deviation along each row of data-frame
"""

mad_row = find_mad(df_all)
print(mad_row)
```

To calculate mean absolute deviation for particular emotion:

```python
_emotion_type = input("Enter the type of emotion whose MAD of pixel values you want to find: ")
_data, _data_numpy, _data_label, _data_label_numpy = select_emotion(df_all, df_label, df, _emotion_type)

mad_row = find_mad(_data)
print("Mean Absolute Deviation of sample (row): \n", mad_row)
```

#### **Variance and Standard Deviation**


**Variance:**

* `Variance` is defined as the average of squared deviations.
* Variance of population is denoted by σ<sup>2</sup>
* Variance of sample is denoted by s<sup>2</sup>
* **Formula:**

$$ \begin{equation} \textrm{Variance} = s^2 = \frac{\sum^n_{i=1} (x_i-\bar{x})^2}{n-1} \tag{v}\\ \end{equation}$$

Here, $$x_1$$, $$x_2$$, $$x_3$$, ......, $$x_n$$ are values in dataset, $$\bar{x}$$ : Represents mean of a sample and n is total number of observations of sample.

**Standard Deviation**

* `Standard deviation` is the square root of the variance.
* It measures spread around the mean.
* Standard deviation is on same scale as original data, therefore it is easier to interpret.
* **Formula:**

$$ \begin{equation} \textrm{Standard Deviation} = s = \sqrt{Variance}\tag{vi}\\ \end{equation} $$

> <font color = "green"> <b> <i> Note: </i> </b></font> `Variance` and `Standard Deviation` are sensitive to **outliers** because they are based on the squared deviations.

Function to find `Variance` and `Standard Deviation`:

```python
def find_var_std(_data_numpy, n):

  """To find variance and standard deviation.

  This function consists of two method to calculate:
  * without library function
  * via library function

  Parameters
  ----------
  _data_numpy : darray
                numpy array of selected sample

  n : str
      select method to calucate variance and stanadard deviation

  Example
  -------
  >>> for _emotion_type = 3
  >>> n = 0, Time taken to calculate:  0.7255523204803467, Variance of sample:  [4645.25424363878, 3525.1757720193746, 3197.1766355538075... 1207.6763326385874, 2137.4093603440274], Standard Deviation of sample:  [68.15610203 59.37319068 56.54358174 ... 68.27522534 34.75163784 46.23212477]
  >>> n = 1, Time taken to calculate:  0.2692246437072754, Variance of sample:  [4645.25424364 3525.17577202 3197.17663555 ... 4661.50639513 1207.67633264 2137.40936034], Standard Deviation of sample:  [68.15610203 59.37319068 56.54358174 ... 68.27522534 34.75163784 46.23212477]

  Returns
  -------
  var : numpy.ndarray
        Returns variance along each row

  std : numpy.ndarray
        Returns standard deviation along each row

  """

  try:

    n = int(n)
    print("Shape of selected numpy array: ", _data_numpy.shape)
    print("Data type of _data_numpy: ", type(_data_numpy))

    if n is 0:

      start = time.time()
      means = [row.mean() for row in _data_numpy]
      squared_error = [(row - mean)**2 for row, mean in zip(_data_numpy, means)]
      var = [row.mean() for row in squared_error]
      std = np.sqrt(var)
      print("Time taken to calculate: ", time.time() - start)

      return var, std

    if n is 1:

      start = time.time()
      var = _data_numpy.var(axis=1)
      std = _data_numpy.std(axis=1)
      print("Time taken to calculate: ", time.time() - start)
      return var, std

    if n is not 1 or n is not 0:
      print("Enter correct value")
  
  except StatisticsError as error:
    raise error
  
  except Exception as error:
    print(error)
    raise error
```
  
Function Call:

```python
n = input("Enter 0 (To find variance & std without any library function), Enter 1 (Via using Library Function): ")
_emotion_type = input("Enter the type of emotion whose variance and standard deviation of pixel values you want to find: ")
_data, _data_numpy, _data_label, _data_label_numpy = select_emotion(df_all, df_label, df, _emotion_type)

var, std = find_var_std(_data_numpy, n)

print()
print("Variance of sample: ", var)
print("Standard Deviation of sample: ", std)
```

### **Estimates Based on Percentiles**

#### **Range**

**Basic Theory**

* Statistics based on sorted data.
* `Range` is the difference between the largest and the sortest value.
* It is extremely sensitive to `outliers`.
* Not usefull as a general measure of data.

**Formula:**

$$ \begin{equation} \textrm{Range} = Maximum Value - Minimum Value \tag{vii} \end{equation} $$

There is no need to calculate range of pixel values as we already know it ranges from 0-255 where, 0 is taken as white and 255 is taken as black.

#### **Percentile**

* `Percentile` is the estimate of propotions of data that should fall above or bellow.
* The dataset should be sorted.

> <font color = "green"> <b> <i> Note: </i> </b></font> Median is the same thing as the 50<sup>th</sup> percentile.

> <font color = "green"> <b> <i> Note: </i> </b></font> Percentile is same as quantile.

The function to calculate `Percentile`:

```python
def find_percentile(_data_numpy, _percent, ax, n):

  """To find percentile.

  This function consists of two method to calculate:
  * without library function
  * via library function

  Parameters
  ----------
  _data_numpy : ndarray
                numpy array of selected sample

  _percent : str
             propotion of data that should fall above or bellow

  ax : str
       axis along which we have to estimate

  n : str
      select method to calucate variance and stanadard deviation

  Example
  -------
  >>> for _emotion_type = 3, _percent = 5, along row
  >>> n = 0, Time taken to calculate:  0.02375483512878418, Percentile of sample:  [  1.8   3.    3.  ... 234.8 238.4 239. ]
  >>> n = 1, Time taken to calculate:  1.5285449028015137, Percentile of sample:  [  0.   0.   0. ... 185. 188. 191.]

  Returns
  -------
  p : list
      percentile

  Note
  ----
  List must be sorted

  """

  try:

    n = int(n)
    ax = int(ax)
    _percent = int(_percent)
    _data_numpy = np.sort(_data_numpy)
    print("Shape of selected numpy array: ", _data_numpy.shape)

    if n is 0:

      start = time.time()

      if not _data_numpy.any():
        return None

      k = (_data_numpy.shape[ax]-1) * (_percent/100)
      floor_op = math.floor(k)
      ceil_op = math.ceil(k)

      if floor_op is ceil_op:
        return _data_numpy[int(k)]

      d0 = _data_numpy[int(floor_op)] * (ceil_op - k)
      d1 = _data_numpy[int(ceil_op)] * (k - floor_op)
      val = d0 + d1

      print("Time taken to calculate: ", time.time() - start)

      return val

    if n is 1:
    
      start = time.time()
      p = np.percentile(_data_numpy, _percent, axis=ax)
      print("Time taken to calculate: ", time.time() - start)
      return p

    if n is not 1 or n is not 0:
      print("Enter correct value")
  
  except StatisticsError as error:
    raise error
  
  except Exception as error:
    print(error)
    raise error
```

Function Call:

```python
n = input("Enter 0 (To find percentile without any library function), Enter 1 (Via using Library Function): ")
_emotion_type = input("Enter the type of emotion whose percentile of pixel values you want to find: ")
_percent = input("Enter percent you want to find: ")
ax = input("Enter 0 (To find along row), 1 (To find along column): ")

_data, _data_numpy, _data_label, _data_label_numpy = select_emotion(df_all, df_label, df, _emotion_type)

percentile = find_percentile(_data_numpy, _percent, ax, n)
print("Percentile of sample: ", percentile)
```

#### **Interquartile Range (IQR)**

* `Interquartile Range` is the difference between 25<sup>th</sup> percentile and 75<sup>th</sup> percentile of data.

The Figure represent the simple demostration of `IQR`:

![iqr](https://user-images.githubusercontent.com/62256509/120905283-bdec1500-c66e-11eb-91d0-03dfd76242a2.png)

**Formula:**

$$ \begin{equation} \textrm{Interquartile Range (IQR)} = Q_3 - Q_1 \tag{viii} \end{equation} $$

Here, given 2n (even) or 2n+1 (odd) number of values.

* first quartile `Q1` = median of the n smallest values
* third quartile `Q3` = median of the n largest values
* second quartile `Q2` = same as the median

The function for `IQR Range`:

```python
def find_iqr(_data, _data_numpy, ax, n):

  try:

    n = int(n)
    ax = int(ax)

    if n is 0:

      _data_numpy = np.sort(_data_numpy)
      print("Shape of selected numpy array: ", _data_numpy.shape)

      start = time.time()

      if not _data_numpy.any():
        return None
        
      q3 = find_percentile(_data_numpy, 75, ax, n)
      q1 = find_percentile(_data_numpy, 25, ax, n)
      
      iqr = q3 - q1
      print("Time taken to calculate: ", time.time() - start)
      
      return iqr

    if n is 1:

      start = time.time()
      iqr = _data.quantile(axis=ax)
      print("Time taken to calculate: ", time.time() - start)

      return iqr

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
n = input("Enter 0 (To find iqr without any library function & in numpy array), Enter 1 (Via using Library Function & in dataframe): ")
_emotion_type = input("Enter the type of emotion whose percentile of pixel values you want to find: ")
ax = input("Enter 0 (To find along row), 1 (To find along column): ")

_data, _data_numpy, _data_label, _data_label_numpy = select_emotion(df_all, df_label, df, _emotion_type)

iqr = find_iqr(_data, _data_numpy, ax, n)
print("IQR of sample: ", iqr)
```

Check out my other posts related to topic:

* [Exploratory Data Analysis](https://khushi-411.github.io/datascience/exploratory-analysis/)
* [Inferential Statistical Analysis](https://khushi-411.github.io/datascience/inferential-stat-analysis/)

Do visit my [GitHub](https://github.com/khushi-411/emotion-recognition/tree/main/data-science) to view complete code! 
