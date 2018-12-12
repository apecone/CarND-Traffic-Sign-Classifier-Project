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

[image1]: ./examples/sample_images.png "Sample Images"
[image2]: ./examples/bar_chart.png "Class Frequencies"
[image3]: ./examples/MobileNet_Layer.png "MobileNet Layer"
[image4]: ./examples/german_sign_1.jpg "Traffic Sign 1"
[image5]: ./examples/german_sign_2.jpg "Traffic Sign 2"
[image6]: ./examples/german_sign_3.jpg "Traffic Sign 3"
[image7]: ./examples/german_sign_4.jpg "Traffic Sign 4"
[image8]: ./examples/german_sign_5.jpg "Traffic Sign 5"
[image9]: ./examples/german_sign_6.jpg "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/apecone/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the class frequencies in training set.

| ![alt text][image2] |
|:--:|
| *Training Set Class Frequencies* |

I also find it helpful to actually see what each image looks like because it gives us important information about our training set; for example, looking at these samples we can tell that each image is cropped and therefore our trained model might not perform well on non-cropped images.

| ![alt text][image1] |
|:--:|
| *30 Sample Images from Training Set* |

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Since I already knew I was going to use MobileNet CNN, I didn't think it was extremely important for me to do any preprocessing other than your typical data normalization step.  Therefore, I used the keras mobilenet preprocessing function which normalizes RGB input by dividing each pixel by 127.5 and subtracting 1.  More details can be found here (https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py)  This data normalization step helps our model converge more quickly during gradient descent.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model utilized the MobileNet architecture.  The only changes I made was using a smaller input layer and a smaller output layer specific to my sign classification task.  Other than those changes, pretty much everything else was the same.

Many CNNs typically have layers which contain a convolution, possibly a batch norm, and then a relu activation.  MobileNet, however, implements their layers using a 3x3 depthwise convolution, batch norm, relu, then a 1x1 convolution, followed by another batch norm and relu.  This particular sequence of functions (in particular the depthwise convolution and 1x1 convolution) has proven to help reduce the number of parameters while still enabling the network to learn very complex non-linear functions.  The batch normalization helps the model converge quickly but it also helps the model generalize well under conditions of covariate shift where the distribution of the data being tested on is different than the training set.

| ![alt text][image3] |
|:--:|
| * Typical (Left) vs MobileNet (Right) |

The architecture is diagramed below:

| Type/Stride | Filter Shape | Input Size |
|:--:|:--:|:--:|
| Conv / s2 | 3 x 3 x 3 x 32 | 32 x 32 x 3 |
| Conv / s2 | 3 x 3 x 3 x 32 | 16 x 16 x 32 |
| Conv / s2 | 3 x 3 x 3 x 32 | 16 x 16 x 64 |
| Conv / s2 | 3 x 3 x 3 x 32 | 8 x 8 x 64 |
| Conv / s2 | 3 x 3 x 3 x 32 | 8 x 8 x 128 |
| Conv / s2 | 3 x 3 x 3 x 32 | 4 x 4 x 128 |
| Conv / s2 | 3 x 3 x 3 x 32 | 4 x 4 x 256 |
| Conv / s2 | 3 x 3 x 3 x 32 | 2 x 2 x 256 |
| Conv / s2 | 3 x 3 x 3 x 32 | 2 x 2 x 512 |
| Conv / s2 | 3 x 3 x 3 x 32 | 1 x 1 x 1024 |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer because it's a pretty awesome combination of exponentially weighted average and momentum that is reliable, stable, and requires very little hyperparameter tuning.  The batch size was chosen to be fairly large (128) so that gradient descent wouldn't be too noisy.  My GPU at home can work on 128 32x32 images at a time without much difficulty.  The number of epochs was just a random guess, I save weights along the way using Keras ModelCheckpoints.  Not much thought went into learning rate since I was using Adam which does a good job of dampening oscillations along dimensions of gradient descent.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 97%
* validation set accuracy of 97% 
* test set accuracy of 94%

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I chose MobileNet because it's a relatively small network which I knew I could easily implement and train.  The fact that it has batch normalization is a good thing because it helps protect the model against covariate shift which I figured could possibly happen to me as soon as I tried using this model on an image which came from a different distribution than the training, validation, and test set.

Because it has fewer parameters and uses Relu instead of Sigmoid or Tanh, I knew it would train pretty quickly and be less susceptible to vanishing gradients.

I felt MobileNet would be relevant for traffic sign application because it's proven useful for a variety of applications so using it for traffic signs didn't seem like much of a stretch.  The original MobileNet paper classifies 1000 different classes; so, 43 seems small by comparison.  Perhaps, most importantly, I chose MobileNet because I was thinking that a self-driving car should have a very fast neural network when classifying traffic signs (or anything for that matter).  So, given the fact that MobileNet is an efficient low latency netowrk, it seemed like a good addition to my self-driving car.

The fact that both my Training and Validation are around the same and are both very high in accuracy 97% is a very good sign, no pun intended.  The Test wasn't quite as good but still met 94% accuracy.  And, to be honest, the fact that I reached these scores without even gathering or augmenting the data is pretty awesome.  Had I augmented the data, I'm sure MobileNet could have done even better.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

All of these images were classified with nearly 100% accuracy.  A lot of this probably is because each image is cropped on the traffic sign.  If, however, the image had not been cropped, the model might not perform as well.  Also, each image is rather clear, so, it would be interesting to see how the model would perform on a set of blurry images especially since I never did any sort of blur augmentation in the training process.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      		| Yield   									| 
| Children Crossing     			| Children Crossing 										|
| Turn right ahead					| Turn right ahead											|
| Right-of-way at next intersection	      		| Right-of-way at next intersection				 				|
| Priority Road			| Priority Road      							|
| No Entry | No Entry |


The model was able to correctly predict 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94%.  Although the model seems to perform well on these images provided, I'm pretty sure we could trick it if we didn't have cropped/centered photos.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

My model is so sure of its prediction that its giving it a 100% probability for Yield, Priority Road, and No Entry.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield   									| 
| 0.99     				| Children Crossing										|
| 0.99					| Turn right ahead									|
| 0.99	      			| Right-of-way at next intersection				 				|
| 1.0				    | Priority Road      							|
| 1.0 | No Entry |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


