**HIGHLY EXPERIMENTAL** - This is not meant to look anything like production quality code.

Detection of autism through the machine learning and deep learning analysis of Magnetoencephalography scans.

Work done for pf. Karim Jerbi of the

Systems Neuroscience and Cognitive Neuroimaging 
Psychology Department 
University of Montreal 
http://www.karimjerbi.com/


This only includes the code. The data isn't currently directly available to the public.


#### Usage: #### 
*The brackets mean "pick one"*.
  * python main.py single_dim_classification [SVM, KNN, RandomForests, SKL_LR, ]


  * python main.py spatial [keras_dense, keras_conv, tflearn_resnet, tflearn_bn_vgg, tflearn_vgg, vgg, cnn, resnet]


  * *[Todo, Not fully functional]* python main.py sequence_classification 

There is a wide range of command line parameters to fiddle with everything. They are detailed in main.py.


#### Dependencies: ####
A whole bunch of stuff, which includes, non-exhaustively:
- numpy
- Tensorflow (tensorflow-gpu ideally)
- Tflearn
- Keras
- Scikit-learn
- mne



(
Keywords : 
mne, magnetoencephalography, meg, 
python, machine learning, ml, 
deep learning, dl, scikit-learn, 
skl, support vector machines, svm, 
random forests, neural networks, nn, 
convnets, cnn, tensorflow, 
tf, neuroscience, neuroimaging, 
neuro psychiatry, resnet, densenet
)
