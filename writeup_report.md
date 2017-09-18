# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/RomansWorks/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Number of training examples = 104397  of them augmented are:  69598
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
Example sign classes: 

	 0  ->  Speed limit (20km/h)

	 1  ->  Speed limit (30km/h)

	 2  ->  Speed limit (50km/h)

	 3  ->  Speed limit (60km/h)

	 4  ->  Speed limit (70km/h)

	 5  ->  Speed limit (80km/h)

	 6  ->  End of speed limit (80km/h)

	 7  ->  Speed limit (100km/h)

	 8  ->  Speed limit (120km/h)

	 9  ->  No passing


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is spread across labels, as well as multiple images with their label value and sign names to show that we can load and match the data correctly. 

![Exploratory Visualization][https://raw.githubusercontent.com/RomansWorks/CarND-Traffic-Sign-Classifier-Project/master/exploratory_visualization.png]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


I augmented the data by providing slightly rotated images. I also tried flipping the images after classifying them with regards to horizontal symmetry, but that prooved of no help for the model. 

I added the actual normalization step as part of the tensorflow processing pipeline (see the LeNet function):
```
    mean, variance = tf.nn.moments(inputs, axes=[0,1,2])
    input = tf.nn.batch_normalization(inputs, mean, variance, None, None, 0.05)
```

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The input later is 32x32x32 RGB image.
It is then normalized after moments are computed.
Two convolutional layers follow, with 20 3x3 filters, and 1x1 strides under valid padding.
Each of those is followed by a RelU activation and max pooling (valid padding, 2x2 kernel and stride).
The output is flattened to a 720 neuron network, and fed into the fully connected layers.
The FC layers are 120, 84, 43 neurons large.
The first two are RelU activated and are followed by a dropout layer (more on that later).
The logits of the last (43 neurons) layer are provided as output. 

The index of the max logit is then taken as the prediction. 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The output of the prediction is a label, and to use back propagation I needed to code the correct label as a value for each output neuron. For that I used the one-hot encoding of the labels.  I then minimized mean cross entropy, and used an Adam optimizer (learning rate remained 0.001 after trying some other values - 0.1, 0.01 with different optimizers).  

I trained the classifier over 10 epochs with a batch size of 128. During training the dropout probability of the fully connected layers was 0.4. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.957
* test set accuracy of 0.949

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The predictive power of the basic LeNet architecture was too low to pass the threshold. 

* What were some problems with the initial architecture?

Overfitting - I had to add two dropout layers to fix this. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

One adjustment was adding the dropoout layers to prevent overfitting. I also increased the number of kernels and modified the convolutional layers to extract more features out of the images. I added another fully connected layer to increase the network power.  

Of course I had to adapt the last layer output to the number of classes (43 vs 10 in the original LeNet-5 architecture). 

* Which parameters were tuned? How were they adjusted and why?

I played with different activation functions and network sizes, different dropout values, 

Epochs were chosen so that learning was efficient (stop when the curve flattens).  I was free to choose a large batch size because memory was not a constraint with 32x32x3 images. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

A convolutional layer allows us to efficiently extract local features independent of their positions in space. A fully connected layer would be able to achieve the detection of these features at a much higher computational cost, and would probably require more training samples too. 

A dropout layer prevents overfitting by causing some paths in the network disappear with a certain probability, which forces other paths to learn the missing features. This creates some uncertainty for the network regarding it's success with predicting very specific images yet still allow multiple paths to find the correct features for overall prediction.  

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I didn't want to take images from Google due to unclear licensing, so instead I found the German Traffic Sign Benchmark database.

I put the images  under the GTRSB folder and am picking some images from there in random:

![Images picked during last run][https://raw.githubusercontent.com/RomansWorks/CarND-Traffic-Sign-Classifier-Project/master/benchmark_images.png]

Regarding difficulty, the first and last images are dark. The last image contains gradients in the treeline which are stronger than in the sign itself. The 4th image is slightly in angle to the camera. The 3rd image's resolution is so low that features are lost in color merge (arrow and edge of sign for example). Some images contain parts of other signs.  

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Each of a few runs with random images resulted in 100% accuracy, and the high confidence of the predictions is displayed by the strong bias of the  top-5 softmax probabilities towards the correct prediction. 

Last run predictions:
```
Class of sampled sign number  0  is predicted to be:  11  but/and is actually:  11
Class of sampled sign number  1  is predicted to be:  3  but/and is actually:  3
Class of sampled sign number  2  is predicted to be:  36  but/and is actually:  36
Class of sampled sign number  3  is predicted to be:  27  but/and is actually:  27
Class of sampled sign number  4  is predicted to be:  21  but/and is actually:  21
```

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

As mentioned the model had very high confidence in the predictions. 

```
Class of sampled sign number  0  is:  11 ; the top 5 predictions and their confidence: 
	class 11 with confidence: 0.9998455048
	class 30 with confidence: 0.0001524954
	class 21 with confidence: 0.0000012091
	class 27 with confidence: 0.0000006810
	class 26 with confidence: 0.0000000543

Class of sampled sign number  4  is:  21 ; the top 5 predictions and their confidence: 
	class 21 with confidence: 0.9999061823
	class 25 with confidence: 0.0000454951
	class 24 with confidence: 0.0000329263
	class 11 with confidence: 0.0000110194
	class 30 with confidence: 0.0000021349
```

and so on.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


