UNCC ECGR 4105 Homework 0
Due: 2/5/23 @ 11:59pm

In this homework, we will use a dataset provided: “D3.csv” 
Let the first three columns of the data set be separate explanatory variables x1, x2, x3. 
Let the fourth column be the dependent variable Y. 
(Note: You cannot use the built-in function from ML libraries for gradient descent, you have to implement it yourself.)

 

Problem 1 (50 points)

Develop a code that run linear regression with gradient decent algorithm for each of the explanatory variables in isolation.
In this case, you assume that in each iteration, only one explanatory variable (either X1, or X2, or X3) is explaining the output.
Basically, you need to do three different training, one per each explanatory variable.
For the learning rate, explore different values between 0.1 and 0.01 (your choice).
Initialize your parameters to zero (theta to zero).

Report the linear model you found for each explanatory variable.
Plot the final regression model and loss over the iteration per each explanatory variable.
Which explanatory variable has the lower loss (cost) for explaining the output (Y)?
Based on your training observations, describe the impact of the different learning rates on the final loss and number of training iteration.
 

Problem 2 (50 points)

This time, run linear regression with gradient descent algorithm using all three explanatory variables.
For the learning rate, explore different values between 0.1 and 0.01 (your choice). 
Initialize your parameters (theta to zero).

Report the final linear model you found the best. 
Plot loss over the iteration.
Based on your training observations, describe the impact of the different learning rates on the final loss and number of training iteration.
Predict the value of y for new (X1, X2, X3) values (1, 1, 1), for (2, 0, 4), and for (3, 2, 1)
 

Homework instructions:

You will submit:
(1) a report (pdf file) that explains your answers for each problem. 
The report MUST contain the link to your GitHub repository to access the source code you have developed for this homework.
Your report should contain your name, student ID, and homework number. 

(2) a second pdf file which contains both source codes, results, and plots you have developed and plotted for this homework. 
You can easily generate this pdf in Jupyter Lab or Jupyter Notebook (sample pdf provided on Canvas.). 
The pdf file name should contain your name, student ID, and homework number.
In your report, provide separate and clear responses for each problem. 
Make reasonable assumptions where necessary and clearly state them! 
Be sure to show all the work involved in deriving your answers! 
If you just give a final answer without explanation, you may not receive credit for that question.

You may discuss concepts with your classmates. This fosters group learning and improves the class’ progress.
However, make sure to submit your own independent and individual solutions. 
