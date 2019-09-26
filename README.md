Facial Expression Classifier API
---
This API utilizes a Convolutional Neural Network to classify facial expressions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) from an image.

Tested on Elementary OS Juno.

How to install:
1. install Anaconda or Miniconda if you haven't yet.
2. clone and checkout the latest release by entering: git clone https://github.com/kdponce/expression-classifier-api.git -b v1.0.1
3. cd to the project directory and recreate the conda environment using: conda env create -f environment.yml

How to run:
1. activate the environment with 'conda activate expression-classifier-api-nogpu' and run app.py
2. to use, cd to the directory with the image to be tested and enter: curl -F 'file=@xxx.jpg' localhost:5000.

where xxx is the filename of the input image. 

ex. curl -F 'file=@angry.jpg' localhost:5000

Also works with PNG files. 
Sample images are included for testing.

 Limitations:
 1. No support for face detection. The face to be classified should be clear and occupy as much of the image as possible.
 2. No support for multiple faces. 
 3. Low model accuracy. To be rectified in a later version via better architectures. 
