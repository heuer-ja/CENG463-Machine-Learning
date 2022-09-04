# Conclusion

## 1. Datasets
The task is to perform a binary classification task. For this purpose, the German dataset "german.data" is used, which contains 7 numeric and 13 qualitative attributes. In an additional dataset "german.data.numeric" all attributes are in numeric form. 

Classification analysis is performed for both datasets:
1. german.data (7 numerical attributes)
2. german.data-numeric (24 numerical attributes)

Each dataset is split to 67% training and 33% test.


## 2. Approach
### Calculate best 2 attributes
Since we are dealing with a binary classification problem, the function **routine_gradient_descent(df, columns, iterations)** is used to brute-force the 2-attribute combination that provides the highest average accuracy.
The function takes the following attributes:
- df: dataset to be examined
- columns: 2 attributes of the dataset to be examined
- iterations: number of iterations the Gradient Descent algorithm runs through to calculate a statistically relevant result for mu and Sigma (here iterations=500)

This function calls the gradient_descent(df, iterations) function.


### Calculate gaussian parameters for 2 attributes
The function **gradient_descent(df, iterations)** is used to train the model iteratively, i.e. the parameters mu and Sigma are determined based on two fixed attributes.
The function takes the following attributes:
- df: already known
- iterations: already known (here iterations=100)


## 3. Results: Normal data set
The used data set for the analysis is *german.data* which contains 7 numerical attributes.

### Accuracy
The highest accuracy (avg. acc ~ 65%) could be achieved with the numerical attributes
1. *Age in years*
2. *Duration in month*

### Risk Analysis
The risk analysis is abstracted by finding the average and maximum risk for each 2-combination of attributes.
For the combination *Age in years* & *Duration in month* the average risk is -9422.05. 
If you calculate the specific risk of an unknown value x (here: one row in the dataset), you should use the smallest of the three risk values: 
1. R(x belongs to class 1)
1. R(x belongs to class 2)
1. R(classification rejected)

### Distribution (2-D Gaussian)
There is no intersection between the 2D Gaussian distributions for class=Good and class=Bad.
The following table shows the approximate 2D Gaussian parameters (mu, Sigma) for both classes (Good, Bad) and attributes *Age in years* and *Duration in month*.

|              | Class=Good        |                   | Class=Bad         |                   |
|--------------|-------------------|-------------------|-------------------|-------------------|
|              | Age in years      | Duration in month | Age in years      | Duration in month |
| **mu**    | 36.53             | 19.30             | 32.91             | 24.69             |
| **Sigma** | sigma11=141.55 | sigma12=-5.69  | sigma11=110.91 | sigma12=6.26   |
|              | sigma21=-5.69  | sigma22=127.28 | sigma21=6.26   | sigma22=181.14 |


## 4. Results: Numerical data set
The used data set for the analysis is *german.data* which contains 24 numerical attributes. The attributes are named alphabetically (a to x) because no headings are provided in the dataset or documentation.

### Accuracy
The highest accuracy (avg. acc ~70%) could be achieved with the numerical attributes:
1. *u*
2. *h*

### Risk Analysis
The risk analysis is abstracted by finding the average and maximum risk for each 2-combination of attributes.
For the combination *u* & *h* the average risk is 72.83. 

### Distribution (2-D Gaussian)
There is a intersection between the 2D Gaussian distributions for class=Good and class=Bad.
The following table shows the approximate 2D Gaussian parameters (mu, Sigma) for both classes (Good, Bad) and attributes *u* and *h*.

|          | Class=Good       | Class=Good        | Class=Bad        | Class=Bad         |
|----------|------------------|-------------------|------------------|-------------------|
|          | Age in years     | Duration in month | Age in years     | Duration in month |
| mu    | 0.91             | 0.03              | 0.91             | 0.06              |
| Sigma | sigma11=0.08  | sigma12=-0.03  | sigma11=0.09  | sigma12=-0.06  |
|          | sigma21=-0.03 | sigma22=0.03   | sigma21=-0.06 | sigma22=0.06   |