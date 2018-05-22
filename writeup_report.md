# **Behavioral Cloning** 

## Writeup Behavioral Cloning

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: nvidia_network.jpg "DNN"
[image2]: loss.png "Loss"
[image3]: center_2018_05_11_16_19_03_491.jpg "Center Image"
[image4]: center_2018_05_18_21_39_56_857.jpg "Recovery Image 1"
[image5]: center_2018_05_18_21_36_24_084.jpg "Recovery Image 2"
[image6]: center_2018_05_18_10_31_30_468.jpg "Recovery Image 2"
[image7]: center_2018_05_18_10_21_33_766.jpg "Normal Image"
[image8]: center_2018_05_18_10_21_33_766_f.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the paper by Nvidia 

![alt text][image1]

it consists of a convolution neural network with 3x3 and 5x5 filters sizes and depths between 24 and 64 (model.py function bc_model() ) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

There is also an option to initialize the model with previously saved model so we could only add new data to train and initialize with previos weights. But de-facto, this option was not really used because of the folder structure I used (all pics dumped to the same location).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (model.py function bc_model() ). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track for a full lap.

#### 3. Model parameter tuning

The model used an adam optimizer, with the learning rate of 1e-5 (trial & error).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (driving 2 full laps clockwise and 2 full laps counterclockwise). After trying to use the trained model in the simulator, I found out that it did pretty well but could not cross the bridge. So I went back to self driving and got some more training data close and on the bridge. Then I also collected data recovering from the left and right sides of the road in case that happens. After that, the trained model successfully drove a lap around the track. See save.mp4.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a model found successful in the literature (Nvidia).
Judging by the end result, this was a good choice.

In order to evaluate how well the model was working, I split my image and steering angle data into a training 80% and validation set 20%. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that in training mode I insert some randomness in the data. I used flipping of the image (and also the recorded steering angle) to allow the training data to be richer, changed the brightness and gave the picture some translation. Besides that, I have the dropout layer in the model ofcourse. 

![alt text][image2]

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (entering the bridge for instance). To improve the driving behavior in these cases, I went to those place manually and recorded extra data, especially on how to recover from the edges.

To help these small recordings make a difference, especially when the nominal track had much more data, I added them 10 times each. The information is added since I have some randomness added to each picture later on.

The drive.py was changed a bit too. First, I had to increase the speed setpoint. Since the yaw rate of the car as a function of the steering angle changes as speed increases, I had to set the speed closer to the speed I trained the model with. Second, I wanted all game resolutions to work so I added a resize function to insert into the model the expected image size. Third, I converted the image format to BGR, again because this is what the model expects.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py function bc_model()) consisted of a convolution neural network with the following layers and layer sizes ...

[None, 75, 320, 3] #Conv2D

[None, 36, 158, 24] #Conv2D

[None, 16, 77, 36] #Conv2D

[None, 6, 37, 48] #Conv2D

[None, 4, 35, 64] #Conv2D

[None, 2, 33, 64] #Conv2D

[None, 2, 33, 64] #Dropout

[None, None] #flatten

[None, 100] # Dense

[None, 100] # activation

[None, 50] # Dense

[None, 50] # activation

[None, 10] # Dense

[None, 10] # activation

[None, 1] # Dense

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one going counterclockwise using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image4]
![alt text][image5]
![alt text][image6]

Then I repeated this process going clockwise in order to get more data points.

To augment the data sat, I also flipped, translated and changed the brightness of random images (and angles accordingly) thinking that this would make the model not to overtrain on the recorded data and help focus on the road. For example, here is an image that has then been flipped:

![alt text][image7]
![alt text][image8]

Etc ....

After the collection process, I had 25k number of data points. I then preprocessed this data by normlization.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was ~10 as evidenced by the plot showed above. There was no improvement in the validation loss, but in the end, the model works :)