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
		* no tire may leave the drivable portion of the track suface.
		* the car may not pop uo onto ledges or roll over any surfaces that would otherwise be consdered unsafe


2. **Data Observation**


	I was using notebook to observe the training data. The file "sdc_bc_data_discovering.ipynb" is attached in the project submission zip file.
	* I first tried to view the 10 random images from Udacity data to visually identify what could be done. Found out that we have extra non useful part of the scene in those images. (top and bottom part of the images and decided to start to crop out 50 pixels from the top and 25 pixels from the bottom)
	* And I used the DataFrame from pandas to summarize the data to see how data look like. And plot out the steering angle vs number of images that are associated with steering angles visually. More than 6000 images angles are closed to less than 0 and the rest are greater than 0 angle. That tells me that the track might have more left curves than the right. So I am certain that I might need to augment the data in a way that it does not bias to certain curves only while in training.
	* By looking at view images of center, left and right of the particular steering angle, left and right camera seems to be mounted on angle paralle to the left and righ body of the car respectively.	One of the hints from Udacity for this project is how do we recover from driving too much right and left to the center (recoveries), we can use these left and right image captures from the data for this recovery simulation.
	* Provided by Udacity, I had a chance to go through NVDIA paper (refernced at the end of this README), I am inspired to alter the brightness of the images. 
	* Since more left turns than right turns in the Data, I decide to flip the images horizontally to simulate some right turns to distribute the data some what left and right turns are closer.	
	* I first think that adding shadow to images randomly will not be necessary the technique that Vivek Yadav use in his solution (referenced at the end of this README). But for the model to work for real situation, this might be a good augmentation of the data so I decided to add shadow to the images randomly.
	* So to recap from this, I will preprocess the data (add, augment) as follows:
		* crop
		* flip
		* resize
		* change brightness
		* add shadow

3. **Design Model**
	* Preprocessing Data
		* 	cropping images (50px from top, 25px from bottom)
		* 	resizing images (66px x 200px)
		* 	flipping images (all)
		* 	altering brightness randomly
		* 	adding shadow randomly 
	
	* Using keras generator
		* 	batch size 128
		* 	10 epochs
	
	* Using NVIDIA model with transfer learning
	

4. **Model Architecture**
	> draw the network structure
	> explain the model achitecture in words


5. **Training approach and process**


6. **Performance** (links to the youtube)


7. **What to improve**


9. **References**
	* Udacity class
	* Udacity Project 3 rubric
	* [https://arxiv.org/pdf/1604.07316v1.pdf]()
	* [https://github.com/commaai/research/blob/master/train_steering_model.py]()
	* [https://carnd-forums.udacity.com/cq/viewquestion.action?id=26214464&questionTitle=behavioral-cloning-cheatsheet]()
	* [https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.ulbf1oge4]()
	* [https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html]()
=======
## 1. Requirements 	
	(i) 	collecting the data of own driving behavior from provided simulator tools or using Udacity collected data
	(ii)	using the data to train Neural Network model in python 
	(iii) 	file naming requirements: 
			- main file: model.python
			- script to drive the car: drive.py
			- model architecture: model.json
			- weights: model.h5
			- design and structure of the network: README.md (this file)
	(iv)	track to be evaluated on: the first track (the one to the left from the simulator options)
	(v)	simulation: 
			- no tire may leave the drivable portion of the track suface.
			- the car may not pop uo onto ledges or roll over any surfaces that would otherwise be consdered unsafe


## 2. Data Observation 
	I was using notebook to observe the training data. The file "sdc_bc_data_discovering.ipynb" is attached in the project
	submission zip file.
	(i) 	I first tried to view the 10 random images from Udacity data to visually identify what could be done. Found out that we have extra non useful part of the scene in those images. (top and bottom part of the images and decided to start to crop out 50 pixels from the top and 25 pixels from the bottom)
	(ii)	And I used the DataFrame from pandas to summarize the data to see how data look like. And plot out the steering angle vs number of images that are associated with steering angles visually. More than 6000 images angles are closed to less than 0 and the rest are greater than 0 angle. That tells me that the track might have more left curves than the right. So I am certain that I might need to augment the data in a way that it does not bias to certain curves only while in training.
	(iii) 	By looking at view images of center, left and right of the particular steering angle, left and right camera seems to be mounted on angle paralle to the left and righ body of the car respectively.	One of the hints from Udacity for this project is how do we recover from driving too much right and left to the center (recoveries), we can use these left and right image captures from the data for this recovery simulation.
	(iv)	Provided by Udacity, I had a chance to go through NVDIA paper (refernced at the end of this README), I am inspired to alter the brightness of the images. 
	(v) 	Since more left turns than right turns in the Data, I decide to flip the images horizontally to simulate some right turns to distribute the data some what left and right turns are closer.	
	(vi)	I first think that adding shadow to images randomly will not be necessary the technique that Vivek Yadav use in his solution (referenced at the end of this README). But for the model to work for real situation, this might be a good augmentation of the data so I decided to add shadow to the images randomly.

	So to recap from this, I will preprocess the data (add, augment) as follows:
	- crop
	- flip
	- resize
	- change brightness
	- add shadow

## 3. Design Model
	(i)	Preprocessing Data
			- cropping images (50px from top, 25px from bottom)
			- resizing images (66px x 200px)
			- flipping images (all)
			- altering brightness randomly
			- adding shadow randomly 

	(ii) 	Using keras generator
			- batch size 128
			- 10 epochs


	(iii)	Using NVIDIA model with transfer learning
			- 
			-

## 4. Model Architecture
	# draw the network structure
	# explain the model achitecture in words


## 5. Training approach and process


## 6. Performance (links to the youtube)


## 7. What to improve


## 9. References
	- Udacity Term 1
	- Udacity Project 3 rubric
	- https://arxiv.org/pdf/1604.07316v1.pdf
	- https://github.com/commaai/research/blob/master/train_steering_model.py
	- https://carnd-forums.udacity.com/cq/viewquestion.action?id=26214464&questionTitle=behavioral-cloning-cheatsheet
	- https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.ulbf1oge4
	- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

