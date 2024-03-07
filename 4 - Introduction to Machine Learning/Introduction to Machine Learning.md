## **What is Machine Learning?**

Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.

Machine learning approaches have been applied to many fields including natural language processing, computer vision, speech recognition, email filtering, agriculture, and medicine. ML is known in its application across business problems under the name predictive analytics. Although not all machine learning is statistically based, computational statistics is an important source of the field's methods.

The mathematical foundations of ML are provided by mathematical optimization (mathematical programming) methods. Data mining is a related (parallel) field of study, focusing on exploratory data analysis (EDA) through unsupervised learning. From a theoretical point of view Probably approximately correct (PAC) learning provides a framework for describing machine learning.

There are different types of machine learning we will focus on during the next sections of the course: Supervised Learning and Unsupervised Learning.

## **Supervised Learning.**

Supervised learning, also known as supervised machine learning, is a subcategory of machine learning and artificial intelligence. It is defined by its use of labeled datasets to train algorithms that to classify data or predict outcomes accurately. As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately, which occurs as part of the cross validation process. Supervised learning helps organizations solve for a variety of real-world problems at scale, such as classifying spam in a separate folder from your inbox.

For example, a segment of text could have a category label, such as:
+ Spam vs. Legitimate Email,
+ Positive vs. Negative Movie Review.

The network receives a set of inputs along with the corresponding correct outputs, and the algorithm learns by comparing its actual 
output with correct outputs to find errors. It then modifies the model accordingly. Supervised learning is commonly used in applications where historical data predicts likely future events.

Supervised Learning process:

![alt text](image.png)

1. **Data Acquisition**: This is the first step and also one of the most important steps in the Machine Learning process. Data can be collected from various sources such as databases, APIs, web scraping or even manually. The data collected may include text, images, audio, video, etc.

2. **Data Cleaning**: Collected data is often not always ready to be used immediately. There can be many problems like missing data, noisy data, inconsistent data, etc. This step ensures that the data is properly prepared so that it can be used effectively in training the model.

3. **Model Training & Building**: Once the data has been cleaned and prepared, we can start building the Machine Learning model. There are different types of models such as supervised learning, unsupervised learning, reinforcement learning, etc. The model is then trained using the cleaned data.

4. **Model testing**: Once the model has been trained, we need to test its performance. This is typically done using a separate set of test data that the model has not seen during training. This helps evaluate the model's generalizability - that is, its ability to work with new data. If the model is not correct with the test data set, we will adjust the model parameters (Step 3).

5. **Model Deployment**: If the model has been tested and shows good performance, it can be deployed for actual use. This can include integrating the model into an application or service, or using the model to generate predictions or useful information from new data.

We often divide the data set with about 30% used to test the data and the remaining 70% used to run the data model.

What we just showed is a simplified approach to supervised learning, it contains an issue! Is it fair to use our single split of the data to evaluate our models performance? After all, we were given the chance to update the model parameters again and again.

To fix this issue, data is often split into 3 sets:

1. **Training Data**: This data is used to train the model parameters. During this process, the model learns to predict the output based on the input using training data.

2. **Validation Data**: This data is used to adjust the hyperparameters of the model. Hyperparameters are parameters that are not learned from training data but are set in advance. For example, the maximum depth of a decision tree is a hyperparameter. Validation data help us choose the best values for these hyperparameters.

3. **Test Data**: After the model has been trained and tuned, we use test data to evaluate the final performance of the model. This helps us understand the model's generalization ability, i.e. its ability to correctly predict the output for new data it has never seen during training.

This last measure is the measure by which we evaluate the actual performance of the model.

## **Underfitting and Overfitting.**

When we talk about the Machine Learning model, we actually talk about how well it performs and its accuracy which is known as prediction errors. Let us consider that we are designing a machine learning model. A model is said to be a good machine learning model if it generalizes any new input data from the problem domain in a proper way. This helps us to make predictions about future data, that the data model has never seen. Now, suppose we want to check how well our machine learning model learns and generalizes to the new data. For that, we have overfitting and underfitting, which are majorly responsible for the poor performances of the machine learning algorithms.

### **1. Bias and Variance in Machine Learning.**

+ Bias: Bias refers to the error due to overly simplistic assumptions in the learning algorithm. These assumptions make the model easier to comprehend and learn but might not capture the underlying complexities of the data. It is the error due to the model’s inability to represent the true relationship between input and output accurately. When a model has poor performance both on the training and testing data means high bias because of the simple model, indicating underfitting.

+ Variance: Variance, on the other hand, is the error due to the model’s sensitivity to fluctuations in the training data. It’s the variability of the model’s predictions for different instances of training data. High variance occurs when a model learns the training data’s noise and random fluctuations rather than the underlying pattern. As a result, the model performs well on the training data but poorly on the testing data, indicating overfitting.

![alt text](image-1.png)

### **2. Underfitting in Machine Learning.**

A statistical model or a machine learning algorithm is said to have underfitting when a model is too simple to capture data complexities. It represents the inability of the model to learn the training data effectively result in poor performance both on the training and testing data. In simple terms, an underfit model’s are inaccurate, especially when applied to new, unseen examples. It mainly happens when we uses very simple model with overly simplified assumptions. To address underfitting problem of the model, we need to use more complex models, with enhanced feature representation, and less regularization.

*Note:* The underfitting model has 'high bias' and 'low variance'.

Summary:

+ Reasons for Underfitting:

        a. The model is too simple, so it may be not capable to represent the complexities in the data.

        b. The input features which is used to train the model is not the adequate representations of underlying factors influencing the target variable.

        c. The size of the training dataset used is not enough.

        d. Excessive regularization are used to prevent the overfitting, which constraint the model to capture the data well.
        
        e. Features are not minimized.

+ Techniques to Reduce Underfitting:

        a. Increase model complexity.

        b. Increase the number of features, performing feature engineering.

        c. Remove noise from the data.

        d. Increase the number of epochs or increase the duration of training to get better results.

![alt text](image-2.png)

![alt text](image-3.png)

### **3. Overfitting in Machine Learning.**

A statistical model is said to be overfitted when the model does not make accurate predictions on testing data. When a model gets trained with so much data, it starts learning from the noise and inaccurate data entries in our data set. And when testing with test data results in High variance. Then the model does not categorize the data correctly, because of too many details and noise. The causes of overfitting are the non-parametric and non-linear methods because these types of machine learning algorithms have more freedom in building the model based on the dataset and therefore they can really build unrealistic models. A solution to avoid overfitting is using a linear algorithm if we have linear data or using the parameters like the maximum depth if we are using decision trees. 

*Note:* The model fits too much to the noise from the data. This often results in low error on training sets but high error on test/validation sets.


Summary:

+ Reasons for Overfitting:

        a. High variance and low bias.

        b. The model is too complex.

        c. The size of the training data.

+ Techniques to Reduce Overfitting:

        a. Increase training data.

        b. Reduce model complexity.

        c. Early stopping during the training phase (have an eye over the loss over the training period as soon as loss begins to increase, stop training).

        d. Ridge Regularization and Lasso Regularization (You can read about them at: https://www.geeksforgeeks.org/lasso-vs-ridge-vs-elastic-net-ml/ and https://www.geeksforgeeks.org/implementation-of-lasso-regression-from-scratch-using-python/).

        e. Use dropout for neural networks to tackle overfitting.

![alt text](image-4.png)

![alt text](image-5.png)

![alt text](image-6.png)

![alt text](image-7.png)

![alt text](image-8.png)

### **4. Good Fit in a Statistical Model.**