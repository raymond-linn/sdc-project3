# SDC Behavior Cloning Project

1. **Requirements**
	* collecting the data of own driving behavior from provided simulator tools or using Udacity collected data
	* using the data to train Neural Network model in python 
	* file naming requirements: 
		* main file: model.python
		* script to drive the car: drive.py
		* model architecture: model.json
		* weights: model.h5
		* design and structure of the network: README.md (this file)

	* track to be evaluated on: the first track (the one to the left from the simulator options)
	* simulation: 
		* no tire may leave the drivable portion of the track surface.
		* the car may not pop uo onto ledges or roll over any surfaces that would otherwise be considered unsafe


2. **Data Observation**


	I was using notebook to observe the training data. The file "sdc_bc_data_gen.ipynb" is attached in the project submission zip file.
	* I first tried to view the 12 random images from Udacity data to visually identify what could be done. Found out that we have extra non useful part of the scene in those images. (top and bottom part of the images and decided to start to crop out 50 pixels from the top and 25 pixels from the bottom)-- updat -- ended up cropping 40px and 25px from the top and bottom respectively.
	* And I used the DataFrame from pandas to summarize the data to see how data look like. And plot out the distribution of steering angle. Noted more than 6000 images angles are on negative values and the rest are in postivie values. Seems like the track might have unenven turns. And learned from the Traffic Sign text project (hard lesson) I am sure that the data need to be augmented in a way that it does not bias to certain curves only.
	* One of the hints from Udacity for this project is how do we recover from driving too much right and left to the center (recoveries), I think I might be able to use left and right images to simulate recoveries. So for simple approach that I would take it is that adding or substracting a factor to the steering unit that associates to either left or right image. Fot center image, it will be as is.
	* Provided by Udacity, I had a chance to go through NVDIA paper (refernced at the end of this README) and from <a href="https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.ulbf1oge4">this post</a>, I should add some random brightness to the images to augment the trining data.
	* Since more left turns than right turns in the Data, I decide to flip half of the images horizontally to simulate some right turns to distribute the data some what left and right turns are closer.	
	* So to recap from this, I will preprocess the data (add, augment) as follows:
		* crop
		* flip
		* resize
		* change brightness



3. **Design Model**
	* Loading Data
		* 	read in csv data with pandas library
		*	split the data into 80-20 to be training and validation data
		
	* Preprocessing Data
		* 	pull the each row from the data
		*	choose image randomly from out of three
		*	adjust angle depends on the left, right or center image
		*	change the color space to RGB -- update: having trouble training and this one is one of the tunnings that I did which improve model but still having gone off track
		* 	alter brightness randomly
		* 	cropping images -- update -- thought it was memory issue when having exception while in training, so try to reduce the in-memory use. but it was not the memory issue rather some stack overflow said there is some problem with TF_Delete... in keras using TensorFlow backend but setled with this configuration. And still going off track (40px from top, 25px from bottom)
		* 	resizing images (95px x 320px)
		* 	flipping images (half) -- update -- was flipping 100% of images that were loaded and later I read from discussion in slack and forum that some cohorts are successful with flipping only 50% of the images so I settled with it and I got my car to pass the bridge and still going off track.
	
	* Using keras generator
		* 	batch size 32 -- update -- starts with 256, 128, 64 and all those gave me some Thread exception which I cannot trace any problems and reduce to 32 and when I got there it was successfully finihsed running script. In the mean time I have fixed some codes and don't know why it works at this size and I will get back to it to investigate when I have time. 
		* 	3 epochs -- update -- starts with 10 and the loss and val_acc does not seem to improve so I rather run few epochs to see if it is okay.
		* onto Model
	

4. **Model Architecture**
	* -- update -- NVIDIA paper is the one that I first go to. I implemented the same layers as NVIDIA I cannot 
	* I implemented the network resemblence to the post mentioned above with changing some parameters. Input shape being (64x64x3)
	* 2-CNN layers with (32, 3, 3) followed by (2x2) Max Pooling and Dropout of 0.5. I used the Activation function as Leaky ReLU after these CNN layers. Output shape after these layers is: 32, 32, 32 and parameters total being: 10,144
	
	* After above was 2-CNN layers with (64, 3, 3) followed by, the same set up as above, (2x2) Max Pooling and Dropout of 0.5 with Activation Leaky ReLU. After these layers Output shape becomes: 16, 16, 64  and parameters for this step: 55,424
	
	* Another 2-CNN layers with (128, 3, 3) as the same set up as above and output shape after this is: 8, 8, 128  and parameters: 221,440
	
	* After flatenning, I have 3 layers of Fully Connected layers with size of 256, 64, 16 and Leaky ReLU, with Dropout of 0.5 in each layer for these three FC layers: The output shapes are: 256, 64, 16  respectively and parameters for these layers are: 2,097,408, 16,488, 1,040 respectively
	
	* at last, one FC layer with size of 1 to get the single output. 1 and parameters for this layer is 17
	* Total parameters for this architecture is 2,401,921. 
	* -- update -- trying out the different parameters and I got passed through to the bridge and after the left turn I got off from the track again at the right curve. 

5. **Training approach and process**
	* I first have the problem with this error: 
		"tensorflow backend error of using dropout and maxpooling fix AttributeError:
		module 'tensorflow.python' has no attribute 'control_flow_ops'"
		After searching and reading for a while and I found on github and now I can use Dropout and MaxPooling
		ref: (https://github.com/udacity/sdc-issue-reports/issues/125)
	* I started to setup NVDIA Architecture, in keras Sequential Model. The car got off track after one second. And I said to myself, this would be a tough project. I try to tune the network as learning keras, tensorflow, CNN architecture with developing Python skill, my car moves for 2 seconds and then go off track. Meanwhile I was following a lot of cohorts conversation on the forum and slack then start looking at the Comma.ai architecture. After all, try to use transfer-learning technique is better than starting Network Architecture from scratch although it is not quite feature extraction, it is still based on some architecture is always better I guess.
	* Then setup the Comma.ai network and train the network again. Try to drive on the track and it just gone off almost right after started driving. So I am not sure what is happening. And I found out the both networks start with different input shapes than what I have althoug I reduced input dimension to what I want (64x64) it is still having problem.
	* I decided to go with the architecture that above blog post use and I set it up and running. Tried on autonomous mode in simulator, it just started driving and I thought it is going good at first and until after I passed the bridge, I got off track again and I was in really nervous that I will not be able to finish this project. I changed the parameters and the other constants that I tweak like all the things that I marked above as in "--update--". It does not get improved.
	* I turned to my mentor, and ask him what do I need to tune. I am out of any idea. He told me that I should normalize the input. I just remember I did normalize the input for Traffic Sign project and why did I miss that part.
	* I normalize the input images and still only get passed the right curves of track after the bridge and It is already 5 days before the deadlines. And I am really in stress. I looked through the Comma.ai code again and I found out it is using Lambda function from keras mdoel. I read the documentation of that function and decided to normalize the input using keras Lambda plus my mentor confirmed that it will be better to use keras Lambda.
	* After I changed that my model started alive. With some out put layers size adjustment, I have totally model.
	* I remember one of the lecture videos from this calss talk about Neural Network is more like Science than Engineering because like Science, we have to experiment it. And this is what it meant by experimenting to get the network working.

6. **Performance** (links to the youtube)
	* Here is the capture of my model running. It is a little shaky. I know I need to improve a lot.


7. **What to improve**
	* I saw some posts on slack and some blog sites that say, people can get it working by small network, small amount of layers and parameters. I am really curious to design and test.
	* You can see in the captured video that the car is a litlle shaky like a first time learner of the bicycle, so how to smooth it out? And what are the other columns data that we have such as throttle, speed and brake. How could I use those. What if I have Gyro, GPS and Radar data, how could I coorelate to this. 
	* I might be able to improve a lot on this model. 
	* what can I do more with augmenting the input images aside from what I did minimally. 
	* these are the some of the first thoughts that come into my mind. I need to learn more about Deep Learning and use all the data that are available in the driving_log file to get small network that can be drive autonoumously and smmothly.
	
8. **References**
	* Udacity class
	* Udacity Project 3 rubric
	* VGG16 Architecture - Keras blog [https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html]
	* NVDIA Paper [https://arxiv.org/pdf/1604.07316v1.pdf]()
	* Comma.ai [https://github.com/commaai/research/blob/master/train_steering_model.py]()
	* Carnd Forum Cheatsheet [https://carnd-forums.udacity.com/cq/viewquestion.action?id=26214464&questionTitle=behavioral-cloning-cheatsheet]()
	* Vivek's Post [https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.ulbf1oge4]()
	* Using Very Little Data - Keras blog [https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html]()
