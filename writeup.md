
# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/labels.png "labels distribution"
[image2]: ./examples/beforeprep.png "Before preprocessing"
[image3]: ./examples/10worstTrain.png "10 worst images of Train set"
[image4]: ./examples/10worstTest.png "10 worst images of Test set"
[image5]: ./examples/rotation.png "Rotation"
[image6]: ./examples/noise.png "Noise addition"
[image7]: ./examples/CLAHEtrain.png "Noise addition"
[image8]: ./examples/CLAHEtest.png "Noise addition"
[image9]: ./examples/afterprep.png "After preprocessing"
[image10]: ./examples/validationsplit.png 
[image11]: ./new_images/test1.jpg 
[image12]: ./new_images/test2.png 
[image13]: ./new_images/test3.png 
[image14]: ./new_images/test4.jpg 
[image15]: ./new_images/test5.jpg 
[image16]: ./new_images/test6.jpg 
[image17]: ./new_images/test7.jpg 
[image18]: ./new_images/test8.jpg 
[image19]: ./new_images/test9.jpg 
[image20]: ./new_images/test10.jpg
[image21]: ./examples/prob1.png
[image22]: ./examples/prob2.png
[image23]: ./examples/prob3.png
[image24]: ./examples/prob4.png
[image25]: ./examples/prob5.png
[image26]: ./examples/prob6.png
[image27]: ./examples/prob7.png
[image28]: ./examples/prob8.png
[image29]: ./examples/prob9.png
[image30]: ./examples/prob10.png


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy and python libraries to calculate summary statistics of the traffic
signs data set:

* The size of training set is **39209**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. 

* It is a bar chart showing how the **labels** of data are distributed. 
We can see labels are not identically distributed.
But train set and test set are balanced.

![label distribution][image1]

* It is a bar chart showing how the **values** of data are distributed for the whole data set. By channel color (R, G, B).
We can see colors are more dark by meaning, but white color is very present. We can see there are few differencies between different color channels for distrubution.

![values distribution][image2]

* It is a display of the **10 worst images in the data sets**. The worst ones, according to information measured by the minimum of standard deviation of values. It is not exacly the detection of the worst cases. But that enable to watch there are some bad qualities sample inside the data sets, hard to identify for human too.

![10 worst images of Train set][image3]

![10 worst images of Test set][image4]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

I decide to **keep the images in RGB**, because my tries don't demonstrate that model are more efficient or simpler in Grayscale or in other color code(YUV,...). And because finally i used **"Contrast Limited Adaptive Histogram Equalization"** for preprocessing. This is very efficient for picture with a low contrast. Also we can have always the true color of the signs and as we know color for road sign is one of the main component of signalization.

So in a first step, I add some transformed datas to the training set (training/validation set) in order to improve the learning especially for abnormal cases. I created a randomly rotated data set with the whole training set (angle between -20°,20°), and a gaussian noisy data set with the whole training set (standard deviation = 0.001). Then, with the 3 data sets: **initial training set, rotated set, noisy set,** I created a new training set with a size multiplied by 3.

I don't shuffle it, because shuffle is done just after, in the split implementation.

Here is an example of a traffic sign image before and after rotation.

![Rotation][image5]

Here is an example of a traffic sign image before and after noise addition.

![Noise Add][image6]


As a last step, I preprocess all the images i get (training and testing set). My preprocessing is a **"Contrast Limited Adaptive Histogram Equalization"** filter, follow by a **minmax normalization between value -0.5 and 0.5**. This normalization seems to be the more efficient. I have tried l1, l2 normalization and other range values.

![10 worst images of Train set after CLAHE][image7]

![10 worst images of Test set after CLAHE][image8]


At the end, after preprocessing, i obtain values distribution like below. Result is more distrubuted than before preprocessing.

![After preprocessing][image9]

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. 

The code for splitting the data into training and validation sets is contained in the seventh code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using **StratifiedShuffleSplit** function from sklearn. In this way, validation and training sets turn for each epoch. "Stratified" means that percentile of validation according to test set is always the same for each class.

I have chosen 5% of data for validation. (1% could be enough!!)

My final training set had **111745** number of images. My validation set and test set had **5882** and **12630** number of images.

![Trainning/Validation balancing][image10]

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the eigth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 4x4     	| 2x2 stride, valid padding, outputs 28x28x64 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Dropout	  	    	| probabilty of activation 0.5 					|
| RELU					|												|
| Convolution 4x4	    | 2x2 stride, valid padding, outputs 10x10x64	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 					|
| Dropout	  	    	| probabilty of activation 0.5 					|
| RELU					|												|
| Convolution 3x3	    | 2x2 stride, valid padding, outputs 3x3x128	|
| Max pooling	      	| 2x2 stride,  outputs 1x1x128 					|
| Dropout	  	    	| probabilty of activation 0.5 					|
| RELU					|												|
| Fully connected		| output 150        							|
| RELU					|												|
| Fully connected		| output 100        							|
| RELU					|												|
| Fully connected		| output 43        								|
| Softmax				|												|
| 						|												|



#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the ninth cell of the ipython notebook. 

To train the model, I used an **Adam optimizer**, batch size = **252**, number of epochs = **25**, learning rate = **0.001**

* I have tried other optimizer or other hyperparameters, but nothing seems to be better. **Adam optimizer** is a well known efficient optimizer.
Adaptative and momentum methodology.

* About **batch size**, i don't detect one optimal value. Just by trying, I know if value is high, training is slow. And by reading some paper, i know if value is too low, model is more sensitive to noise (maybe not converge!!)

* I choose this number of **epochs** because with my new model, validation accuracy looks as the model reaches the optimal value or a rooftop after 25 epochs.

* **Learning rate** give me a fast training time and a optimal accuracy. "0.0001" is slower and not more accurate. 


Inside my code, I have implemented some Tensorboard summaries. Tensorboard seems to be very usefull to improve the model.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* validation set accuracy of **0.992**
* test set accuracy of **0.978**

Difference between 2 accuracies indicates some overfiting. But one part of overfitting has been reduced by dropout

My approach have been iterative. I start from Lenet model. Then I try different size of layers. Then I add one CNNs layer. Finally I add dropout. I try too other thing but I have not seen improvements.

Some topics:

* **Convolutional layers** are adapted for classification of image, because they fix the image classification issues about translation and zooming invariance of object to detect. For traffic signs data, these issues are not pretty sharp. But Convolion layer are useful. I choose to use 3 CNN because commonly, we said that first one detects edges, the second one detects shape, and then they detect advanced feature. So i think for road sign 3 CNNs is a good number. And testing it, show me that it was a good thing.
* **Pooling** is one of important part of CNN. Pooling participate to the translation invariance. In addition, Pooling reduce the overfitting and the size of paramater matrix.
* **RELU** activation since Alexnet is well known like the more efficient (training efficiency) activation function.
* Sequence (CNN, pooling,  dropout, RELU) is not fixed. RELU just after CNN is possible too. 
* Finally, I use **dropout**. Dropout methodology is very efficient for limiting the overfiting. What is touchy, we have to implement code to freeze the dropout for prediction and accuracy measurement. In other words, dropout is only use during the training period. 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![image1][image11] ![image2][image12] ![image3][image13]![image4][image14] ![image5][image15] ![image6][image16] ![image7][image17] ![image8][image18]![image9][image19] ![image10][image20] 

The five last images might be difficult to classify because there are deformed or vandalized.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the eleventh cell of the Ipython notebook.

Here are the results of the prediction:

| Image									        |	Prediction									| 
|:---------------------------------------------:|:---------------------------------------------:|
| Priority road  								| Priority road									|
| Slippery road									| Slippery road									|
| No passing for vehicles over 3.5 metric tons	| No passing for vehicles over 3.5 metric tons	|
| Slippery road									| Slippery road					 				|
| Go straight or left							| Go straight or left							|
| Speed limit (80km/h)							| Speed limit (80km/h)							|
| No entry										| **Stop**										|
| Stop											| Stop											|
| Children crossing								| **Road work**									|
| Speed limit (50km/h)							| **Speed limit (30km/h)**						|

The model was able to correctly guess **7 of the 10 traffic signs**, which gives an accuracy of 70%. We can understand confusion for wrong results.

Accuracy for web images is 0.7 while test accuracy was 0.978. That could be a proof of overfitting. But 3 cases is too short to have serious proof. What you see here is that model is not well trained for deformed, vandalized or masked images. That means this cases are not covered in the training data set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is extremely sure that this is a Priority road sign (probability of 1.0), and the image does contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| **Priority road**								| 
| .00     				| Traffic signals 								|
| .00					| Right-of-way at the next intersection			|
| .00	      			| Bicycles crossing						 		|
| .00				    | Beware of ice/snow							|

![Probability 1][image21]

For the second image the model is extremely sure that this is a Slippery road sign (probability of 1.0), and the image does contain a Slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| **Slippery road**								| 
| .00     				| Bicycles crossing 							|
| .00					| Dangerous curve to the lef					|
| .00	      			| Beware of ice/snow							|
| .00				    | Wild animals crossing 						|

![Probability 2][image22]

For the third image the model is extremely sure that this is a No passing for vehicles over 3.5 metric tons sign (probability of 1.0), and the image does contain a No passing for vehicles over 3.5 metric tons sign. The top five soft max probabilities were

| Probability         	|     Prediction	        						| 
|:---------------------:|:-------------------------------------------------:| 
| 1.00         			| **No passing for vehicles over 3.5 metric tons**	| 
| .00     				| No passing 										|
| .00					| Speed limit (60km/h)								|
| .00	      			| Speed limit (80km/h)								|
| .00				    | End of no passing by vehicles over 3.5 metric tons|

![Probability 3][image23]

For the forth image the model is pretty sure that this is a Slippery road sign (probability of 0.811), and the image does contain a Slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .811         			| **Slippery road**								| 
| .077     				| Bicycles crossing 							|
| .063					| Wild animals crossing							|
| .023	      			| Dangerous curve to the left					|
| .008				    | Double curve									|

![Probability 4][image24]

For the fiveth image the model is extremely sure that this is a Go straight or left sign (probability of 1.00), and the image does contain a Go straight or left sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| **Go straight or left**						| 
| .00     				| Keep right 									|
| .00					| Turn left ahead								|
| .00	      			| Roundabout mandatory							|
| .00				    | Ahead only									|

![Probability 5][image25]

For the sixth image the model is very sure that this is a Speed limit (80km/h) sign (probability of 0.934), and the image does contain a Speed limit (80km/h) sign. The sign is deformed but the model pedicts correctly.
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .934         			| **Speed limit (80km/h)**						| 
| .045     				| Speed limit (60km/h) 							|
| .017					| Speed limit (50km/h)							|
| .004	      			| End of speed limit (80km/h)					|
| .00				    | Speed limit (30km/h)							|


![Probability 6][image26]

For the seventh image the model is pretty sure that this is a Stop sign (probability of 0.779), **BUT** the image does contain a No entry sign. The sign is artistically vandalized. But human could predict without trouble. Model is not good for this case. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .779         			| Stop											| 
| .221     				| **No entry** 									|
| .00					| No passing for vehicles over 3.5 metric tons 	|
| .00	      			| Bicycles crossing								|
| .00				    | No vehicles									|


![Probability 7][image27]

For the eigth image the model is extremely sure that this is a Stop sign (probability of 1.00), and the image does contain a Stop sign. The sign is masked by small stickers but model is good in this case. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| **Stop**										| 
| .00     				| Bicycles crossing 							|
| .00					| No entry										|
| .00	      			| Speed limit (30km/h)							|
| .00				    | Speed limit (80km/h)							|

![Probability 8][image28]

For the ninth image the model is very incertain that this is a Road work sign (probability of 0.248), **BUT** the image does contain a Children crossing sign. The good one is not inside the top 5 prediction. The sign is masked by branches. But human predict could predict without trouble. Model is not good for this case. This case is interesting because model is incertain. 
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .248         			| Road work										| 
| .166     				| Double curve									|
| .105					| Wild animals crossing							|
| .086	      			| Slippery road									|
| .078				    | Speed limit (50km/h)							|

**Wrong prediction**

![Probability 9][image29]

For the tenth image the model is not very sure that this is a Speed limit (30km/h) sign (probability of 0.68), **BUT** the image does contain a Speed limit (50km/h) sign. The sign is inclined, model is wrong for this prediction in this case but good response is close. Training more could give us a best prediction.
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .680         			| Speed limit (30km/h)							| 
| .313     				| **Speed limit (50km/h)** 						|
| .006					| Speed limit (80km/h)							|
| .001	      			| Speed limit (20km/h)							|
| .000				    | Speed limit (70km/h)							|

![Probability 10][image30]

### Possible improvements

* Model could be improve by adjusting some parameters or by changing the architecture. Possibility are infinite.

* Better preprocessing and add other training set will improve the accuracy of the model.

* Apply to sharper input image, for example with size of 128x128, should improve the accuracy of the model.


```python

```
