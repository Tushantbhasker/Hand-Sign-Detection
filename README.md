# Sign Language Detection
Sign languages (also known as signed languages) are languages that use the visual-manual modality to convey meaning.
Language is expressed via the manual signstream in combination with non-manual elements. Sign languages are full-fledged
natural languages with their own grammar and lexicon. This means that sign languages are not universal and they are not
mutually intelligible although there are also striking similarities among sign languages.

Deep learning methods have demonstrated state-of-the-art results on hand sign detection. What is most impressive 
about these methods isit uses only a simple CNN model to tarin.

## Notebook Overview
* Prepare Photo Data
* Develop Deep Learning Model
* Train With Photo Data
* Evaluate Model
* Save Model

## Python Environment
To run this progarm you have a Python SciPy environment installed, ideally with Python 3. You must have Keras (2.1.5 or higher) 
installed with either the TensorFlow or Theano backend. The tutorial also assumes you have scikit-learn, Pandas, NumPy, 
and Matplotlib installed.

## Dataset
I have used hand sign dataset provided by Kaggle.<br/>
Dataset is provided in the repository if yo want it.

## Defining the Model
I have used a CNN model for training.
* First layer is a 2-D Convolutional layer with 32 nodes.
* Second layer is also a 2-D Convolutional layer but with a 64 nodes.
* Next is a Dense layer wih a 128 nodes and Relu as a activation fuction.
* Last layer is output layer with 26 nodes and Softmax as a activation function. 

## Output
![Inout Image](/sample_image.png)<br/>
__Predicted Output: __ A
## Dependencies
* Keras
* Tensorflow
* Numpy
* pandas
* Matplotlib
* Pickle
* OS module
