---
title: 'PyML_ch1'
description: 'Giving Computers the Ability to Learn from Data'
pubDate: 'Dec 06 2025'
heroImage: '../../assets/PyML.jpg'
category: 'Notebooks'
---

This chapter introduces you to the main subareas of machine learning in order to tackle various problem tasks. In addtion, it discussed the essential steps for creating a typical machine learning model by building a pipeline that will guide us through the following chapters.

### 1.1 Transforming data into knowledge

In the second half of the twentieth
century, machine learning evolved as a subfield of Artificial Intelligence (AI) that
involved self-learning algorithms that derived knowledge from data in order to make
predictions.
Thanks to machine learning, we enjoy robust email spam filters, convenient text and voice recognition software, reliable web search engines, challenging chess-playing programs, and, hopefully soon, safe and efficient self-driving cars.

### 1.2 The three different types of machine learning

There are three different types of machine learning: supervised learning, unsupervised learning, and reinforecement learning.

#### 1.2.1 Supervised Learning

The main goal in supervised learning is to learn a model from labeled training data that allows us to make predictions about unseen or future data. Here, the termsupervised refers to a set of samples where the desired output signals (labels) are already known.

#### 1.2.2 Reinforcement Learning

Another type of machine learning is reinforcement learning. In reinforcement learning, the goal is to develop a system (agent) that improves its performance based on interactions with the environment. Since the information about the current state of the environment typically also includes a so-called reward signal, we can think of reinforcement learning as a field related to supervised learning. 

However, in reinforcement learning this feedback is not the correct ground truth label or value,but a measure of how well the action was measured by a reward function. Through its interaction with the environment, an agent can then use reinforcement learning to learn a series of actions that maximizes this reward via an exploratory trial-and-error approach or deliberative planning.

#### 1.2.3 Unsupervised Learning

In supervised learning, we know the right answer beforehand when we train our model, and in reinforcement learning, we define a measure of reward for particular actions by the agent.

In unsupervised learning, however, we are dealing with unlabeled data or data of unknown structure. Using unsupervised learning techniques, we are able to explore the structure of our data to extract meaningful information without the guidance of a known outcome variable or reward function.

### 1.3 The basic terminology and notations
- Training Sample/Example
    -  A single row of data used to train the model. It contains    both the information used to make a prediction (features) and the actual answer (target).
- Training
    - The process of teaching the machine learning model. During this phase, the model iterates through the data to learn the relationship between the inputs (features) and the outputs (target). It adjusts its internal parameters to minimize errors.
- Feature
    - An input variable used to make predictions. These are the specific characteristics or properties of the data point.
- Target/Label
    - The value that the model is trying to predict. In the context of training data, this is the "correct answer" or the "ground truth."
- Loss Function
    - A mathematical formula that measures how "wrong" the model's predictions are compared to the actual target values. The goal of training is to minimize the output of this function.

### 1.4 A road map for building machine learning systems

#### 1.4.1 Preprocessing - get idea into shape

Data preprocessing is important because of a fundamental concept in computing called **"Garbage In, Garbage Out" (GIGO)**.

If you feed a machine learning model messy, unorganized, or "dirty" data, it will produce poor predictions, no matter how advanced the algorithm is. Real-world data is rarely ready to be used straight out of the box.

Here are the four main reasons why preprocessing is critical:

##### 1. Machines only understand numbers (Encoding)
Most machine learning algorithms are mathematical formulas. They cannot calculate the sum of "Blue" and "Red."
*   **The Problem:** Your data might contain text (e.g., "Male"/"Female" or "New York"/"London").
*   **Preprocessing Solution:** You must convert these text categories into numbers (e.g., 0 and 1) so the model can process them. This is often called *Encoding*.

##### 2. Differing scales confuse the model (Scaling)
Imagine you are predicting a person's health. You have two features: **Age** (0–100) and **Annual Income** (20,000–100,000).
*   **The Problem:** To a computer, the number 100,000 is massive compared to 50. The model might falsely assume that *Income* is 2,000 times more important than *Age* simply because the number is bigger. This ruins distance-based algorithms (like K-Nearest Neighbors).
*   **Preprocessing Solution:** You squash both features into the same range (e.g., between 0 and 1). This is called *Normalization* or *Standardization*.

##### 3. Real data is messy and incomplete (Cleaning)
Data collection is rarely perfect. Sensors break, surveys get skipped, or humans make typos.
*   **The Problem:**
    *   **Missing Values:** If a row is missing data, the math in the algorithm will break (return an error).
    *   **Outliers:** If everyone is 30 years old, but one person is entered as 300 years old (a typo), that one number can skew the average and mislead the model.
*   **Preprocessing Solution:** You need to fill in missing values (imputation) and remove or fix extreme errors.

##### 4. It improves speed and accuracy
*   **The Problem:** Messy or unscaled data makes the optimization process (how the model learns) very difficult. The model struggles to find the "bottom of the valley" (minimum loss) in the mathematical landscape.
*   **Preprocessing Solution:** Clean, scaled data allows the model to converge (finish learning) much faster and usually results in higher prediction accuracy.

There are another two concepts:

### 1. Dimensionality Reduction (Handling Redundant Data)

The text explains that having too many features (inputs) can sometimes be a bad thing, especially if those features overlap too much.

*   **The Problem: Redundancy & Correlation**
    Sometimes features tell you the exact same information.
    *   *Example:* Imagine a dataset for cars that has a column for "Horsepower" and another column for "Engine Power (in kW)." These two are **highly correlated**; if one goes up, the other goes up. Having both is **redundant**—it's duplicate information that confuses the model.

*   **The Solution: Dimensionality Reduction**
    This is a technique used to "compress" the data. It combines or selects features to reduce the total number of inputs without losing important information.

*   **The Benefits:**
    1.  **Storage:** Fewer features mean the dataset takes up less memory on the computer.
    2.  **Speed:** With fewer inputs to calculate, the learning algorithm runs much faster.
    3.  **Better Performance (Signal-to-Noise Ratio):** Sometimes a dataset has a lot of "noise" (irrelevant or garbage data). By reducing the dimensions, you strip away the useless noise and keep only the strong "signal" (the useful patterns). This helps the model predict better.

### 2. The Train/Test Split (Evaluating the Model)

The second part of the text explains how to prove that your model actually works.

*   **The Goal: Generalization**
    You don't want a model that simply memorizes the data it has already seen. You want a model that **generalizes**—meaning it can make accurate predictions on *new* data it has never seen before.

*   **The Method: Splitting the Data**
    To test this, you take your entire dataset and randomly cut it into two parts:
    1.  **Training Set:** You give this to the model to learn from. It optimizes its internal math based on this data.
    2.  **Test Set:** You hide this data from the model completely until the very end.

*   **The Analogy: The Final Exam**
    Think of the **Training Set** as the homework and textbook problems a student studies. Think of the **Test Set** as the questions on the final exam.
    *   If the student sees the exam questions (Test Set) while studying, they will memorize the answers (overfitting).
    *   To see if the student *actually* learned the subject, you must give them exam questions they have never seen before. This is why the text says we keep the test set "until the very end."

#### 1.4.2 Training and selecting a predictive model

Here is an explanation of the text, broken down into four key concepts.

### 1. The "No Free Lunch" Theorem (One Size Does Not Fit All)
The text begins by referencing a famous theorem in machine learning called "No Free Lunch."
*   **The Concept:** There is no single "super algorithm" that works best for every single problem. An algorithm that is excellent at recognizing faces might be terrible at predicting stock prices.
*   **The Analogy:** The text uses the saying, "If the only tool you have is a hammer, you treat everything as if it were a nail." You cannot use one specific algorithm to solve every problem.
*   **The Action:** Because of this, you cannot simply guess which model to use. You must try (train) several different algorithms and compare them to see which one works best for your specific data.

### 2. Measuring Success (Metrics)
To compare these different algorithms, you need a scoreboard.
*   **The Metric:** The text suggests **Classification Accuracy**. This is simply the percentage of answers the model got right (e.g., "The model correctly classified 90 out of 100 images").

### 3. The Validation Set (The "Practice Exam")
The text raises a problem: *If we save the Test Set for the very end, how do we choose the best model in the meantime?*
*   If you use the Test Set to choose your model, you are "cheating" (peeking at the final exam).
*   **The Solution:** You take your Training Set and split it *again*. You keep a small portion aside called a **Validation Set** (or use a technique called **Cross-Validation**).
*   **How it works:**
    1.  **Training Set:** Used to teach the model.
    2.  **Validation Set:** Used to compare different models and select the best one (like a practice quiz).
    3.  **Test Set:** Used only once at the very end to see how the winner performs (the final exam).

### 4. Hyperparameter Optimization (Turning the Knobs)
Finally, the text explains that algorithms come with default settings that usually aren't perfect.
*   **The Definition:** **Hyperparameters** are settings that the model *cannot* learn on its own. They must be set by the human before training starts.
*   **The Analogy:** Think of a radio. The music coming out is the data. The **Hyperparameters** are the volume and bass knobs. You have to manually turn these knobs to get the clearest sound (best performance). The text notes that later chapters will focus on how to tune these knobs.

### Summary
In short, because no algorithm is perfect for everything, we must **test multiple algorithms**. We use a **Validation Set** to pick the winner without touching our final Test Set, and we **tune the settings (hyperparameters)** to squeeze the best performance out of the model.

#### 1.4.3 Evaluating models and predicting unseen data instances

### 1.5 Using Python

Python is one of the most popular programming languages for data science and therefore enjoys a large number of useful add-on libraries developed by its great developer and and open-source community.

Although the performance of interpreted languages, such as Python, for computation-intensive tasks is inferior to lower-level programming languages, extension libraries such as NumPy and SciPy have been developed that build upon lower-layer Fortran and C implementations for fast and vectorized operations on
multidimensional arrays.

For machine learning programming tasks, we will mostly refer to the scikit-learn library, which is currently one of the most popular and accessible open source machine learning libraries. In the later chapter, when we focus on the subarea of machine learning called **deep learning**, we will use the TensorFlow library.

#### 1.5.1 Installing Python and packages

We can download Python from the official Python website: http://www.python.org.

The additional packages that we will be using through this book can be installed via the `pip` installer program. More information about pip can be found at https://docs.python.org/3/installing/index.html.

After we have successfully installed Python, we can execute pip from the Terminal to install addtional Python packages:      
`pip install SomePackages`

Already installed packages can be updated via the --upgrade flag:     
`pip install SomePackages --upgrade`

#### 1.5.2 Using the Anaconda

Anaconda is a free -- including for commercial use -- enterprise-ready Python distribution that bundles all the essential Python packages for data science, math, and engineering in one user-friendly cross-platform distribution.
Download at http://docs.anaconda.com/anaconda/install/ .

After Anaconda is succesfully installed, execute the following command to install new Python packages:    
`conda install SomePackage`

Existing packages can be updated using the following command:   
`conda update SomePackage`

#### 1.5.3 Useful packages

Please make sure that the version numbers of your installed packages are equal to, or greater than, those numbers to ensure the code examples run correctly: 
- NumPy 1.12.1
- SciPy 0.19.0
- scikit-learn 0.18.1
- Matplotlib 2.0.2
- pandas 0.20.1