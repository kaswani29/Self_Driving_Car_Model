#**Behavioral Cloning** 

###Running a car in Udacity self driving car simulator using nvidia end to end self driving deep learning architecture


---

####1. Layout followed for this project

The steps used to build this model are as following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---

This  repository includes the following files:
* model.py containing the script to process data, create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network

####2.  How to run the model ?
Using the Udacity provided simulator which it has open sourced now and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

####3. Pipeline of the project code 

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

This model is based on the architecture of [Nvidia's end to end deep learning architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) 
The model consists of 5 convolution neural network layers with 3x3 and 5x5 filter sizes and depths between 24 and 64 

The model includes RELU layers to introduce nonlinearity and the data is normalized and cropped in the model using Keras lambda and cropping2d layer.

####2. Attempts to reduce overfitting in the model

The model contains two dropout layers between first two fully connected layers which were not a part of original Nvidia paper to avoid overfitting to this track.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer to optimize learning rate automatically.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, using images from left and right camera with steering correction etc 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to study similar which were used for similar use cases and try to implement them as well as tweaking them as needed.

My first step was to use a simple convolution neural network layer followed by some fully connected layers to run the car and create a data pipeline. Seeing the car moving on track gave huge motivation as well as feedback to understand at what places the model might be doing wrong. Soon, this model was replaced by much more sophisticated nvidia end to end deep learning model. I thought this model might be appropriate because it was easy to implement, quite straight forward and has been used for similar purposein real lfe. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it has dropout layers and added more data. Then I removed around 70 pixels from the top and 25 pixels from the bottom in the image of height 320 and width 160 to discard the sky and bottom part which is not useful for the model. This also resulted in model training a lot faster. that useful to 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture:

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.




