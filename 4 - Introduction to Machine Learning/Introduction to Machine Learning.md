## **What is Machine Learning?**

Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.

Machine learning approaches have been applied to many fields including natural language processing, computer vision, speech recognition, email filtering, agriculture, and medicine. ML is known in its application across business problems under the name predictive analytics. Although not all machine learning is statistically based, computational statistics is an important source of the field's methods.

The mathematical foundations of ML are provided by mathematical optimization (mathematical programming) methods. Data mining is a related (parallel) field of study, focusing on exploratory data analysis (EDA) through unsupervised learning. From a theoretical point of view Probably approximately correct (PAC) learning provides a framework for describing machine learning.

There are different types of machine learning we will focus on during the next sections of the course: Supervised Learning and Unsupervised Learning.

## **Supervised Learning.**

Supervised learning, also known as supervised machine learning, is a subcategory of machine learning and artificial intelligence. It is defined by its use of labeled datasets to train algorithms that to classify data or predict outcomes accurately. As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately, which occurs as part of the cross validation process. Supervised learning helps organizations solve for a variety of real-world problems at scale, such as classifying spam in a separate folder from your inbox.

***Applications of Supervised learning:*** Supervised learning can be used to solve a wide variety of problems, including:

+ Spam filtering: Supervised learning algorithms can be trained to identify and classify spam emails based on their content, helping users avoid unwanted messages.

+ Image classification: Supervised learning can automatically classify images into different categories, such as animals, objects, or scenes, facilitating tasks like image search, content moderation, and image-based product recommendations.

+ Medical diagnosis: Supervised learning can assist in medical diagnosis by analyzing patient data, such as medical images, test results, and patient history, to identify patterns that suggest specific diseases or conditions.

+ Fraud detection: Supervised learning models can analyze financial transactions and identify patterns that indicate fraudulent activity, helping financial institutions prevent fraud and protect their customers.

+ Natural language processing (NLP): Supervised learning plays a crucial role in NLP tasks, including sentiment analysis, machine translation, and text summarization, enabling machines to understand and process human language effectively.

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

For example, supervised learning, as the name indicates, has the presence of a supervisor as a teacher. Supervised learning is when we teach or train the machine using data that is well-labelled. Which means some data is already tagged with the correct answer. After that, the machine is provided with a new set of examples(data) so that the supervised learning algorithm analyses the training data(set of training examples) and produces a correct outcome from labeled data. I labeled dataset of images of Elephant, Camel and Cow would have each image tagged with either “Elephant” , “Camel” or “Cow.”

![alt text](image-26.png)

***Advantages of Supervised learning:***

+ Supervised learning allows collecting data and produces data output from previous experiences.

+ Helps to optimize performance criteria with the help of experience.

+ Supervised machine learning helps to solve various types of real-world computation problems.

+ It performs classification and regression tasks.

+ It allows estimating or mapping the result to a new sample.

+ We have complete control over choosing the number of classes we want in the training data.

***Disadvantages of Supervised learning:***
+ Classifying big data can be challenging.

+ Training for supervised learning needs a lot of computation time. So, it requires a lot of time.

+ Supervised learning cannot handle all complex tasks in Machine Learning.

+ Computation time is vast for supervised learning.

+ It requires a labelled data set.

+ It requires a training process.

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

        d. Ridge Regularization and Lasso Regularization.

        e. Use dropout for neural networks to tackle overfitting.

You can read more about [Ridge Regularization](https://www.geeksforgeeks.org/lasso-vs-ridge-vs-elastic-net-ml/) and [Lasso Regularization](https://www.geeksforgeeks.org/implementation-of-lasso-regression-from-scratch-using-python/) by click on its, the reference sources about them are collected in Greeks for Greesk website.

![alt text](image-4.png)

![alt text](image-5.png)

![alt text](image-6.png)

![alt text](image-7.png)

![alt text](image-8.png)

### **4. Good Fit in a Statistical Model.**

Ideally, the case when the model makes the predictions with 0 error, is said to have a good fit on the data. This situation is achievable at a spot between overfitting and underfitting. In order to understand it, we will have to look at the performance of our model with the passage of time, while it is learning from the training dataset.

With the passage of time, our model will keep on learning, and thus the error for the model on the training and testing data will keep on decreasing. If it will learn for too long, the model will become more prone to overfitting due to the presence of noise and less useful details. Hence the performance of our model will decrease. In order to get a good fit, we will stop at a point just before where the error starts increasing. At this point, the model is said to have good skills in training datasets as well as our unseen testing dataset.

I think the above is just a theory. To make it easier to imagine, we should go into an example, specifically as follows:

+ Good model:

![alt text](image-9.png)

+ Bad model:

![alt text](image-10.png)

When thinking about overfitting and underfitting we want to keep in mind the relationship of model performance on the training set versus the test validation set.

Let’s imagine we split our data into a <span style="color: red">**training set**</span> and a <span style="color: blue">**test set**</span>.

+ We first see performance on the <span style="color: red">**training set**</span>:

![alt text](image-11.png)

+ Next we check performance on the <span style="color: blue">**test set**</span>. Ideally, the model would perform well on both, with similar behavior:

![alt text](image-12.png)

+ But what happens if we overfit on the <span style="color: red">**training data**</span>? That means we would perform poorly on new <span style="color: blue">**test data**</span>!

![alt text](image-13.png)

![alt text](image-14.png)

+ This is a good indication of training too much on the <span style="color: red">**training data**</span>, you should look for the point to cut off training time!

![alt text](image-15.png)

## **Evaluating Performance.**

### **1. Clasification.**

We just learned that after our machine learning process is complete, we will use performance metrics to evaluate how our model did. Let’s discuss classification metrics in more detail! The key classification metrics we need to understand are:

+ Accuracy,
+ Recall,
+ Precision,
+ F1-Score.

But first, we should understand the reasoning behind these metrics and how they will actually work in the real world! Typically in any classification task, your model can only achieve two results: Either your model was correct in its prediction or your model was incorrect in its prediction.

Fortunately incorrect vs correct expands to situations where you have multiple classes, such as trying to predict categories of more than two. For example, you have categories A, B, C, and D, you can either be correct in predicting the correct category or incorrect in predicting the right category.

For the purposes of explaining the metrics, let’s imagine a **binary classification** situation, where we only have two available classes and this idea is going to expand to multiple classes as well.

In my example, we will attempt to predict if an image is a dog or a cat. Since this is supervised learning, we will first fit/train a model on training data, then test the model on testing data. That means we're gonna have images that someone's already gone ahead and labeled dog or cat so we know the correct answer on these images. We're then gonna show new images that the model hasn't seen before get the model's prediction and compare the results of the model's prediction to the correct answer that we already know. Once we have the model’s predictions from the `X_test data`, we compare it to the true `y` values (the correct labels).

+ Let's imagine we've already trained our model on some training data, and now it's time to actually evaluate the model's perfomance. This is where our test dataset comes in:

![alt text](image-16.png)

+ We take a test image from what we're gonna label `X_test data`:

![alt text](image-17.png)

+ And there is a corresponding correct label from `y_test`:

![alt text](image-18.png)

+ The model is going to make some prediction and the model predicts that this is a dog:

![alt text](image-19.png)

+ We then compare the prediction to the correct label. So the dog equal dog and in this case, it was correct:

![alt text](image-20.png)

+ However, maybe it predicted that this image was a cat and in this case, this comparition to the correct label would be incorrect:

![alt text](image-21.png)

We repeat this process for all the images in our `X_test data`. At the end we will have a count of correct matches and a count of incorrect matches. The key realization we need to make, is that in the real world, not all incorrect or correct matches hold equal value!

Also in the real world, a single metric won’t tell the complete story! To understand all of this, let’s bring back the 4 metrics we mentioned and see how  they are calculated. We could organize our predicted values compared to the real values in a **confusion matrix**.

**a. Accuracy:**

Accuracy in classification problems is *the number of correct predictions made by the model **divided by** the total number of predictions*. For example,if the `X_test` set was 100 images and our model correctly predicted 80 images, then we have 80/100 (0.8 or 80% accuracy).

Accuracy is useful when target classes are well balanced. So what does 'well balanced' means? It means the actual labels themselves are roughly equally represented in the dataset. In our example, we would have roughly the same amount of cat images as we have dog images.

Accuracy is not a good choice with unbalanced classes! Imagine we had 99 images of dogs and 1 image of a cat. If our model was simply a line that 
always predicted dog we would get 99% accuracy!

**b. Recall:**

Ability of a model to find all the relevant cases within a dataset. The precise definition of recall is *the number of true positives **divided by** the number of true positives plus the number of false negatives*.

**c. Precision:**

Ability of a classification model to identify only the relevant data points. Precision is defined as t*he number of true positives **divided by** the number of true positives plus the number of false positives*.

***Recall and Precision:***

Often you have a trade-off between Recall and Precision. While recall expresses the ability to find all relevant instances in a dataset, precision expresses the proportion of the data points our model says was relevant actually were relevant.

**d. F1-score:**

In cases where we want to find an optimal blend of precision and recall we can combine the two metrics using what is called the F1 score. The F1 score is the harmonic mean of precision and recall when taking both metrics into account in the following equation:

$$F_1 = 2\cdot\frac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}$$

We use the harmonic mean instead of a simple average because it punishes extreme values. A classifier with a precision of 1.0 and a recall of 0.0 has a simple average of 0.5 but an F1 score of 0. 

***Confusion matrix:***

![alt text](image-22.png)

We will explain a little about the two parts: positive and negative predictions, but I will focus on True Positive (TP) and True Negative (TN). For example, your classification model is predicting a person to have a disease, if the person actually has the disease then it will be TP and if the person does not have the disease but the model predicted to have the disease then it will be TN, and for other cases are the same.

After that, there's a lot of other metrics you can calculate:

![alt text](image-23.png)

The main point to remember with the confusion matrix and the various calculated metrics is that they are all fundamentally ways of comparing the predicted values versus the true values. What constitutes “good” metrics, will really depend on the specific situation! Still confused on the confusion matrix? No problem! Check out the Wikipedia page for it, it has a really good diagram with all the formulas for all the metrics.

### **2. Regression.**

Regression is a task when a model attempts to predict continuous values (unlike categorical values, which is classification). You may have heard of some evaluation metrics like accuracy or recall, these sort of metrics aren’t useful for regression problems, we need metrics designed for **continuous** values!

For example, attempting to predict the price of a house given its features is a **regression task**, but attempting to predict the country house is in given its features would be a classification task.

Let’s discuss some of the most common evaluation metrics for regression:

+ Mean Absolute Error
+ Mean Squared Error
+ Root Mean Square Error

**a. Mean Absolute Error (MAE):**

This is the simplest metric used to analyze the loss over the whole dataset. As we all know the error is basically the difference between the predicted and actual values. Therefore MAE is defined as the average of the errors calculated. Here we calculate the modulus of the error, perform the summation and then divide the result by the number of data points.  It is a positive quantity and is not concerned about the direction. The formula of MAE is given by:

$$\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|$$

In there:

+ $n$: Total number of samples or data.
+ $y_i$: The actual or observed value of the ith sample.
+ $\hat{y}_i$: Predicted value of the ith sample.
+ $\left| y_i - \hat{y}_i \right|$: Absolute error between observed value and predicted value.

Now is the issue with mean absolute error, MAE won’t punish large errors. For example, let's take a look at this specific situation where we have one point that's a huge outlier:

![alt text](image-24.png)

We want our error metrics to account for these!

![alt text](image-25.png)

**b. Mean Square Error (MSE):**

The most commonly used metric is Mean Square error or MSE. It is a function used to calculate the loss. We find the difference between the predicted values and the truth variable, square the result and then find the average over the whole dataset. MSE is always positive as we square the values. The small the MSE better is the performance of our model. The formula of MSE is given:

$$\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2$$

In there:

+ $n$: Total number of samples or data.
+ $y_i$: The actual or observed value of the ith sample.
+ $\hat{y}_i$: Predicted value of the ith sample.
+ $(y_i-\hat{y}_i)^2$: Square error between observed value and predicted value.

Larger errors are noted more than with MAE, making MSE more popular.

However, there is another problem we have with MSE which is that the square of the true label minus our prediction is actually the square of the units themselves. For example, if we predict the price of a house our MAE error measure will be in dollars, but with MSE we will get an error measure in dollars squared, which very difficult to interpret.

**c. Root Mean Square Error (RMSE):**

RMSE is a popular method (has same units as `y`) and is the extended version of MSE. This method is basically used to evaluate the performance of our model. It indicates how much the data points are spread around the best line. It is the standard deviation of the Mean squared error. A lower value means that the data point lies closer to the best fit line:

$$\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}$$

Most common question: “Is this value of RMSE good?”. It really depends on the context of the situation, if you build a machine learning model using RMSE to predict house prices in a city, it's about $10. An RMSE of $10 is great for predicting the price of a house, but terrible for predicting the price of a candy bar!

***Conclusion:*** Compare your error metric to the average value of the label in your dataset to try to get an intuition of its overall performance. Domain knowledge also plays an important role here! Context of importance is also necessary to consider. We may create a model to predict how much medication to give, in which case small fluctuations in RMSE may actually be very significant. You should now feel comfortable with the various methods of evaluating a regression task.

## **Unsupervised Learning.**

Unsupervised learning is a type of machine learning that learns from unlabeled data. This means that the data does not have any pre-existing labels or categories. The goal of unsupervised learning is to discover patterns and relationships in the data without any explicit guidance.

Unsupervised learning is the training of a machine using information that is neither classified nor labeled and allowing the algorithm to act on that information without guidance. Here the task of the machine is to group unsorted information according to similarities, patterns, and differences without any prior training of data. 

Unlike supervised learning, no teacher is provided that means no training will be given to the machine. Therefore the machine is restricted to find the hidden structure in unlabeled data by itself. 

You can use unsupervised learning to examine the animal data that has been gathered and distinguish between several groups according to the traits and actions of the animals. These groupings might correspond to various animal species, providing you to categorize the creatures without depending on labels that already exist.

![alt text](image-27.png)

Unsupervised Learning process:

![alt text](image-28.png)

1. **Data Acquisition:** This is the first step, where data is collected from various sources.
2. **Data Cleaning:** Collected data is prepared and cleaned for analysis. This process includes removing noisy data, filling in missing values, and normalizing the data.
3. **Model Training & Building:** Here, algorithms are applied to learn patterns from cleaned data.
4. **Transformation:** Changes made to optimize or fine-tune the model.
5. **Model Deployment:** After transformation, the model is deployed for actual use or decision making.

Key Points:

+ Unsupervised learning allows the model to discover patterns and relationships in unlabeled data.
+ Clustering algorithms group similar data points together based on their inherent characteristics.
+ Feature extraction captures essential information from the data, enabling the model to make meaningful distinctions.
+ Label association assigns categories to the clusters based on the extracted patterns and characteristics.

For example, imagine you have a machine learning model trained on a large dataset of unlabeled images, containing both dogs and cats. The model has never seen an image of a dog or cat before, and it has no pre-existing labels or categories for these animals. Your task is to use unsupervised learning to identify the dogs and cats in a new, unseen image. For instance, suppose it is given an image having both dogs and cats which it has never seen. Thus the machine has no idea about the features of dogs and cats so we can’t categorize it as ‘dogs and cats ‘. But it can categorize them according to their similarities, patterns, and differences, i.e., we can easily categorize the above picture into two parts. The first may contain all pics having dogs in them and the second part may contain all pics having cats in them. Here you didn’t learn anything before, which means no training data or examples. It allows the model to work on its own to discover patterns and information that was previously undetected. It mainly deals with unlabelled data.

***Types of Unsupervised Learning:*** Unsupervised learning is classified into two categories of algorithms: 

+ Clustering: A clustering problem is where you want to discover the inherent groupings in the data, such as grouping customers by purchasing behavior.
+ Association: An association rule learning problem is where you want to discover rules that describe large portions of your data, such as people that buy X also tend to buy Y.

There are certain tasks that fall under unsupervised learning:

+ Clustering: Grouping together unlabeled data points into categories/clusters. Data points are assigned to a cluster, based on similarity.
+ Anomaly Detection: Attempts to detect outliers in a dataset. For example, fraudulent transactions on a credit card.
+ Dimensionality Reduction: Data processing techniques that reduces the number of features in a dataset, either for compression, or to better understand underlying trends within a dataset.

***Application of Unsupervised learning:*** Non-supervised learning can be used to solve a wide variety of problems, including:

+ Anomaly detection: Unsupervised learning can identify unusual patterns or deviations from normal behavior in data, enabling the detection of fraud, intrusion, or system failures.
+ Scientific discovery: Unsupervised learning can uncover hidden relationships and patterns in scientific data, leading to new hypotheses and insights in various scientific fields.
+ Recommendation systems: Unsupervised learning can identify patterns and similarities in user behavior and preferences to recommend products, movies, or music that align with their interests.
+ Customer segmentation: Unsupervised learning can identify groups of customers with similar characteristics, allowing businesses to target marketing campaigns and improve customer service more effectively.
+ Image analysis: Unsupervised learning can group images based on their content, facilitating tasks such as image classification, object detection, and image retrieval.

***Advantages of Unsupervised learning:***

+ It does not require training data to be labeled.
+ Dimensionality reduction can be easily accomplished using unsupervised learning. 
+ Capable of finding previously unknown patterns in data.
+ Unsupervised learning can help you gain insights from unlabeled data that you might not have been able to get otherwise.
+ Unsupervised learning is good at finding patterns and relationships in data without being told what to look for. This can help you learn new things about your data.

***Disadvantages of Unsupervised learning:***

+ Difficult to measure accuracy or effectiveness due to lack of predefined answers during training. 
+ The results often have lesser accuracy.
+ The user needs to spend time interpreting and label the classes which follow that classification.
+ Unsupervised learning can be sensitive to data quality, including missing values, outliers, and noisy data.
+ Without labeled data, it can be difficult to evaluate the performance of unsupervised learning models, making it challenging to assess their effectiveness.

## **Supervised vs. Unsupervised Machine Learning.**

| Parameters | Supervised Learning | Unsupervised learning |
| --- | --- | --- |
| Input data | Algorithms are trained using labeled data. | Algorithms are used with unlabeled data. |
| Computational complexity | Simple method | High computational complexity |
| Accuracy | Very accurate | Less accurate |
| No. of classes | No. of classes is known | No. of classes is not known |
| Data Analysis | Use offline analytics | Use real-time data analytics |
| Algorithm used | Linear regression and logistic regression, random forests, multiclass classifiers, decision trees, Support vector machines, Neural networks, etc. | K-Means Clustering, Hierarchical Clustering, KNN, Apriori Algorithm, etc. |
| Output | The desired output is provided. | Expected output is not provided. | Parameters | Supervised Learning | Unsupervised learning |
| Training data | Use training data to infer the model. | Do not use training data. |
| Complex Models | It is not possible to learn larger and more complex models than with supervised learning. | Larger and more complex models can be learned with unsupervised learning. |
| Model | We can test our model. | We cannot test our model. |
| Called | Supervised learning is also known as classification. | Unsupervised learning is also known as clustering. |
| Example | For example: Optical character recognition. | Example: Find faces in images. |
| Supervision | Supervised learning requires supervision to train the model. | Unsupervised learning does not require any supervision to train the model. |

## **Reference source:**

$[1].$ Pierian Data, [Python for Data Science Course](https://pieriantraining.com/learn/python-for-data-science/).

$[2].$ GreeksforGreeks, [Supervised and Unsupervised Learning](https://www.geeksforgeeks.org/supervised-unsupervised-learning/).

$[3].$ GreeksforGreeks, [ML | Underfitting and Overfitting](https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/).

$[4].$ Wikipedia, [Machine learning](https://en.wikipedia.org/wiki/Machine_learning).

$[5].$ Wikipedia, [Supervised learning](https://en.wikipedia.org/wiki/Supervised_learning).

$[6].$ Wikipedia, [Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix).

$[7].$ GreeksforGreeks, [Machine Learning Model Evaluation](https://www.geeksforgeeks.org/machine-learning-model-evaluation/).