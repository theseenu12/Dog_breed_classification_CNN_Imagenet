import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay,multilabel_confusion_matrix
from pathlib import Path
import tensorflow_hub as hub
from PIL import Image
import os
import datetime
import seaborn as sns



## end to end multiclass dog breed classification
## this notebook builds end to end multiclass classification using tensorflow and tensorflow hub

## 1 problem 
## 2 data
## 3 evaluation 
## 4 features

## 1.Problem

## when im sitting at a cafe and want to take photo of the dog and identify it 
## this can be done using the classification we are doing right now

## 2.Data
## all the data downloaded from kaggle

## 3.Evaluation
## the evaluation is a file with prediction probabilities for each dog breed of each test image

## 4.features
## some information about the data
## were dealing with images (unstructured Data)
## it's probably best we use deep learning and transfer learning
## its a multiclass problem as there are 120 breeds of dogs so its 120 class classification
## there are around 10000 images in train set and there are about 10000 inmages in test set

## get our workspace ready
## Import tf.2.0
## make sure were using a gpu
## import tensorflow hub

print(tf.__version__)

# print('Gpu','Available' if tf.config.list_physical_devices() else "Not available")

## getting our data ready and turning it into tensors

## importing our labels_csv

labels_csv = pd.read_csv('labels.csv')

print(labels_csv.head(20))

## how many images are there of each breed

print(labels_csv['breed'].value_counts())
print(len(labels_csv['breed'].unique()))

# print(labels_csv['breed'].unique())


# labels_csv['breed'].value_counts()[:20].plot.bar()


# plt.show()

## what is the median number of images per class

print(labels_csv['breed'].value_counts().median())

## lets view an image

# img = Image.open('Train/00a338a92e4e7bf543340dc849230e75.jpg')

# img.show()


## getting images and their labels_csv

# print(labels_csv.head())

## create pathnames from IMageid's

filenames = ["Train/" + fname + '.jpg' for fname in labels_csv['id']]

## check the first 10
print(filenames[:20])


## check whether the number of filenames matches number of actual image files

print(len(filenames))

if len(os.listdir('Train')) == len(filenames):
    print("Filenames match actual amount of files")
    
else:
    print("Filenames not matched")


# print(os.listdir('Train'))
## one more check

# Image.open(filenames[9000]).show()
print(labels_csv['breed'][9000])

    
## since weve now got our training image filepaths in a list,lets prepare our labels

## turning the data labels into numbers

labels = labels_csv['breed'].to_numpy()

## see if number of labels matches the filenames

if len(labels) == len(filenames):
    print("Labels matches filenames")
else:
    print("Labels does not match filenames")


## find the unique labeled values
unique_breeds = np.unique(labels)

print(np.unique(labels))

## turn a single labels into array of booleans

print(labels[0] == unique_breeds)


## turn every label into boolean array

boolean_labels = []

for label in labels:
    boolean_labels.append(label == unique_breeds)



print(boolean_labels[1])


## turning the boolean array into integers


boolean_labels  = np.array(boolean_labels,dtype=bool)

print(boolean_labels[0])

print(np.argmax(boolean_labels[0]))

print(np.where(unique_breeds == [labels[0]]))

    
## setup x and y Variables
x = filenames
y = boolean_labels

NUM_IMAGES = 250
print("FILENAMES")

print(filenames)

print("LABELSS")
print(boolean_labels)

## start splitting the data into train and valid

x_train,x_valid,y_train,y_valid = train_test_split(x[:NUM_IMAGES],y[:NUM_IMAGES],test_size=0.2)


## lets have a geez at the training data

print(x_train[:5])

print(y_train[:5])


## preprocessing images (Turning images into Tensors)

## to preprocess images into Tensors were going to write a function which does a few things:
## 1.Take an Image Filepath as an input
## 2.Use Tensorflow to read the file and save it to a Variable,'Image'
## 3.Turn our Image(Jpg) into 'Tensors'
## 4.Resize the image to be shape of (224,224) (For Ease of Performing Tensorflow operations)
## 5.Return the modified Image

## lets convert sample file to nparray
image = plt.imread(filenames[45])

print(image)


# print(unique_breeds[np.argmax(boolean_labels[45])])

# plt.imshow(image)

# plt.show()

## turn the numpy array to tensor using tf.constant()

image_tensor = tf.constant(image)

print(image_tensor)


## now we've seen what an image look like as a tensor,now we will make a function to preprocess them

## 1.Take an Image Filepath as an input
## 2.Use Tensorflow to read the file and save it to a Variable,'Image'
## 3.Turn our Image(Jpg) into 'Tensors'
## 4.Normalize our Image ,convert color channels from 0-255 to 0-1 
## 5.Resize the image to be shape of (224,224) (For Ease of Performing Tensorflow operations)
## 6.Return the modified Image

## define Image size

IMG_SIZE = 224

# img_size = tf.constant([224,224])


## create a function for preprocessing images

def process_image(image_path,img_size=IMG_SIZE):
    ## takes an image file path and turns the image into tensor
    
    
    ## read in an image file and it return the tensor and dtype as string
    image = tf.io.read_file(image_path)
    
    ## one more method
    
    # image1 = tf.constant(plt.imread(image_path))

    ## turn the jpeg into numerical(int) tensor with 3 color channels
    image = tf.image.decode_jpeg(image,3)
    
    ## convert the color channel values from 0-255 to 0-1 values (This is called Normalization)

    image = tf.image.convert_image_dtype(image,tf.float32)
    
    ## resize the image to our desired shape
    image = tf.image.resize(image,size=[224,224])
    
    return image


## turning our data into batches
## why turn data into batches

# lets say youre trying to process 10000 images in one go
# they all might not fit into memory

## thats why we do 32 images at a time(you can manually adjust batch size if you need be)

## in order to use tensorflow effectively we need our data in the form of Tensor Tuples Which look like this:
## (image,label)


## create a simple function to return a tuple
def get_image_label(image_path,label):
    
    """ takes an image file path name and label and returns the tuple of image and label"""

    image = process_image(image_path)
    
    return image,label

## testing our process_image function

print(process_image(x[39]))

print(unique_breeds[np.argmax(y[39])])


# plt.imshow(process_image(x[39]))

# plt.show()



## now weve got a way to turn data into tuple of tensors in the form ('image,label)

## lets make a function to turn all our data  (x and y) into batches

## define the batch size #32 is a good number

## create a function to turn data into batches

batch_size = 32

def create_data_batches(x,y=None,batch_size=batch_size,valid_data=False,test_data=False):
    
    '''
    create batches of data out of image x and label y pairs
    it shuffles the data if its training data,But  doesnt shuffle data if its validation data
    also accepts test data as input(thats why we have y=None in case)
    '''
    
    ## if the data is a test dataset we probably dont have labels
    if test_data:
        print("Creating Test data batches..")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))  ## only filepaths no labels
        data_batch = data.map(process_image).batch(batch_size=batch_size)
        
        return data_batch
    
    ## if the data is valid data we dont need to shuffle it

    elif valid_data:
        print("Creating valid Data Batches ......")
        
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y)))
        
        data_batch = data.map(get_image_label).batch(batch_size=batch_size)
        
        return data_batch
        
    else:
        print("Creating Training Data batches ...... ")
        ## turn filepaths and labels into tensors
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y)))
        
        ## shuffling pathnames and labels before mapping image processor function is faster than shuffling images

        data = data.shuffle(buffer_size=len(x))
        
        ## create image and label tuples (this also turns the image path into image tensor with range between 0-1 instaed of 0-255)
        data_batch = data.map(get_image_label).batch(batch_size)

        return data_batch


## lets create training and validation data

train_data = create_data_batches(x=x_train,y=y_train)

val_data = create_data_batches(x=x_valid,y=y_valid,valid_data=True)

## checkout the different batches as an numpy array


##check out the different attributes of data batches
print(train_data.element_spec)

print(val_data.element_spec) 


print(train_data.as_numpy_iterator().next())

print(val_data.as_numpy_iterator().next())

## visualizing data batches

## our data is now in batches it is a bit hard to understand ,lets visualize the data


# create a function for viewing images in data batch

def show_25_img(images,labels,train=True):
    
    ## loop through 25 for displaying 25 images
    fig = plt.figure(figsize=(10,10))
    fig.tight_layout(pad=5.0)
    if train:
        fig.suptitle("Train Images and Thier Labels")
    else:
        fig.suptitle("Valid Images and Thier Labels")
    for i in range(0,25):
        
        
        plt.subplot(5,5,i+1)
        plt.imshow(images[i])
        # plt.grid(False,axis='both') 
        plt.axis('off')
        plt.title(unique_breeds[np.argmax(labels[i])],loc='center',fontdict=dict(fontsize='large',fontfamily='sans-serif',fontstyle='oblique'))
        
    plt.show()
    
    


train_images,train_labels = next(train_data.as_numpy_iterator())

print(train_images,train_labels)

## lets visualize the training data
print(len(train_images),len(train_labels))
# show_25_img(train_images,train_labels,True)

##lets visualize the validation Data

valid_images,valid_labels = next(val_data.as_numpy_iterator())

# show_25_img(valid_images,valid_labels,False)

## building a model

##  before we build a model there are few things we need to define
# the input shape("the shape of our Images")

## the output shape(image labels,in the form of tensors ) of our model
## the url of the model we want to use

## setup input shape to the model

INPUT_SHAPE = [None,IMG_SIZE,IMG_SIZE,3]  # batch,height,width,colorchannels

## set up the output shape

OUTPUT_SHAPE = 120
## setup model URL From tensorflow_hub

MODEL_URL = 'https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4'
# https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4

MODEL_URL2 = "https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow2/variations/130-224-classification/versions/2"

## now we've got our inputs and outputs and model
## now we will put them into keras deeplearning model

## knowing this lets create a function which :
## takes the input shape and output shape and the model we've choosen as parameters
## defines the layers in keras model in sequential fashion
## compiles the model (Says it should be evaluated and improved)
## builds the model(tells the model the input shape it will be getting)
## returns the model

## create a function which builds a keras model

def create_model(input_shape=INPUT_SHAPE,output_shape=OUTPUT_SHAPE,model_url=MODEL_URL2):
    print("Building model with :",model_url)
    
    ## setup the model layers

    model = tf.keras.Sequential([hub.KerasLayer(model_url),
                                 
                                 tf.keras.layers.Dense(units=output_shape,activation='softmax')## layer2  Output Layer
                                 ])
    
    
    ## compile the model

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
    
    
    ## build the model
    model.build(input_shape)

    return model

model = create_model()

print(model.summary())
    
## creating callbacks
## callbacks are helper functions a model can use during training to do such things as save its progress ,check its progress or training early if model stops improving.

## well create two callbacks one for tensorboard which helps track our progress and another for early stopping which prevents our model from training too long

## tensorboard callbacks

## load Tensorboard notebook extension

## to setup a tensorboard callback we need three things
# 1.Load the tensorboard notebook
# 2.Create a tensorboard callback which is able to save the logs to a directory and pass it to our models fit function
# 3.Visualize our models training logs with tensorboard magic function
 
# Creat a function to build tensorboard callback

def create_tensorboard_callback():
    ## create a log directory for storing tensorboard logs
    logdir = os.path.join('D:/Visual Studio Projects/Machine Learning Project Daniel Boruke/logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    return tf.keras.callbacks.TensorBoard(logdir)

## early stopping callback
## https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping

## early stopping helps our model from overfitting by stopping training if a certain evaluation metric stops improving

## create early stopping callback
early_stoppping  = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=3)



## training a model on subset of data
## our first model is only going to train on 1000 images to make sure everything is working

NUM_EPOCHS = 20

## check to make sure we are still running on a gpu

print("GPU Available " if tf.config.list_physical_devices('GPU') else "Not available")

## lets create a function which trains a model 
## create a model using create_model()

## setup tensorboard callback using create_tensorboard_callback()
## call the fit function on our model passing it the training_data,validation_data,number of epochs to train for which is num_epochs and callbacks which we like to use
## return the model


## build a function to train and return a trained model

def train_model():
    '''Trains a given model and returns a trained version'''
    
    ## create model
    model = create_model()
    
    ## create a Tensorboard session everytime we train a model
    tensorboard = create_tensorboard_callback()
    
    ## fit the model to the data passing it the callbacks we created    
    model.fit(train_data,epochs=NUM_EPOCHS,validation_data=val_data,validation_freq=1,callbacks=[early_stoppping])
    
    ## return the fitted model

    return model

    
## fit the model to the data

model1 = train_model()

print(model1.summary())

## looks like our model is overfitting because its performing well on the training dataset but poor on the validation dataset,What are some ways to prevent model overfitting in deep learning neural networks
## Note:Overfitting is a good thing | it means our model is learning


## making and evaluating predictions using trained model

## make predictions on the validation data (Not used to train on)

predictions = model1.predict(val_data,verbose=1)

print(predictions)

arg_max_predict = []


for i in range(0,len(predictions)):
    
    arg_max_predict.append(unique_breeds[np.argmax(predictions[i])])
    
    
## get a sample prediction of 50th element in valid data
# print(arg_max_predict[50])
# print("Predicted label is",arg_max_predict[50])

# print(np.argmax(y_valid[50]))

# print("Actual Label is",unique_breeds[np.argmax(y_valid[50])])    

# plt.imshow(plt.imread(x_valid[50]))

# plt.show()

## plot a function to dispaly the predictions

def plot_predictions():
    
    for i in range(0,20):
        color = ("green" if arg_max_predict[i] == unique_breeds[np.argmax(y_valid[i])] else "red")
        plt.subplot(5,4,i+1)
    
        plt.title(f"Pred:{arg_max_predict[i]} True:{unique_breeds[np.argmax(y_valid[i])]}",fontdict=dict(fontsize='small',fontfamily='sans-serif',fontstyle='oblique'),loc='center',color=color)
        plt.imshow(plt.imread(x_valid[i]))
        plt.axis('off')
        
        
       
    plt.show()    
    
# plot_predictions()


## lets predict the dog image downloaded from internet

lab = ['lab1.jpg','lab2.jpg','gold2.jpg']


# lab_array = plt.imread(lab)

test_data = create_data_batches(lab,y=None,test_data=True)

## predict the label of the image

lab_predict = model1.predict(test_data)

print(unique_breeds[np.argmax(lab_predict[0])])

print(unique_breeds[np.argmax(lab_predict[1])])

print(unique_breeds[np.argmax(lab_predict[2])])



## test image after processing and converting as tensor
test_image1 = process_image('lab2.jpg')

# plt.imshow(test_image1)

# plt.show()


## daniel boruke cont
## turn prediction probabilities into their Respective labels

def get_pred_label(prediction_probabilities):
    """ 
    turns an array of prediction probabilities into labels
    """
    
    return unique_breeds[np.argmax(prediction_probabilities)]

## get a predicted label based on an array of prediction probabilties 

# print(get_pred_label(predictions[9]))


## since our validation data is in batch_dataset well have to unbatchify it to make predictions on the validation images
# and then compare those predictions to the validation labels(Truth labels or ideal labels)

## create a function to unbatch a batch dataset

def unbatchify(data):
    """
    Takes a bactch dataset of (image,labels) tensors and returns seperate arrays of images,labels
    
    """
    images = []
    labels = []
    
    ## loop through the data batch
    for image,label in data.unbatch().as_numpy_iterator():
        images.append(image)
        ## return the label names directly without using the get_pred_label function
        # labels.append(unique_breeds[np.argmax(label)])

        labels.append(unique_breeds[np.argmax(label)])
            
        
    return images,labels       
       
val_images,val_labels = unbatchify(val_data)

## lets check the length of the valid data and its corresponding labels


print(len(val_images))

print(len(val_labels))

# print(get_pred_label(val_labels[60]))

# images_unbatch = []

# label_unbatch = []

# for image,label in val_data.unbatch().as_numpy_iterator():
#     images_unbatch.append(image)

#     label_unbatch.append(label)


## create a sample Dataset from a list of 4 numbers and batch them into 2 batches 
# dataset = tf.data.Dataset.from_tensor_slices(tf.constant([1,2,3,4])).batch(2)

# dataset = dataset.unbatch()


# for element in dataset.as_numpy_iterator():
#     print(element)

# print(list(dataset.as_numpy_iterator()))


## now we've got ways to get 
#prediction labels and 
#validation labels(truth labels ) 
#validation images

## lets make a function to visualize all these above parameters

#well create a function which takes 
## an array of prediction probabilities and an array of truth labels and an array of integers and images
## convert the prediction probabilities into prediction labels
## plot the Predicted label,its predicted probability,the truth label and target image on single plot

def plot_pred(pred_probs,labels,images,n=59):
    '''
    view the prediction ground truth and image for sample n
    '''
    pred_prob,true_label,image = pred_probs[n],labels[n],images[n]
    ## get the pred label
    pred_label = get_pred_label(pred_prob)
    
    ## plot image and remove ticks

    plt.imshow(image)
    plt.axis('off')

    plt.xticks([])
    plt.yticks([])
    
    plt.title(f"{np.max(pred_prob)*100 :.2f} % | pred:{pred_label} | True:{true_label} ",color=("green" if pred_label == true_label else "red"))
    
    # plt.show()
    
# plot_pred(predictions,val_labels,val_images)

## lets make another function to visualize our model top 10 predictions

## this function will
# Take an input of prediction probabilities array and a ground truth array and integer
# find the predicted label using get_pred_label()
## find the top 10
# prediction probability indexes
# prediction probabilty values
# prediction labels

## plot the top 10 prediction probability values and labels coloring the true label green

def plot_pred_conf(prediction_probabilities,labels,n=1):
    """
    plus the top 10 prediction confidences along with the truth label for sample n
    """
    
    pred_prob,true_label = prediction_probabilities[n],labels[n]
        
    ## get the prediction label
    pred_label = get_pred_label(pred_prob)
    
    ## find the top 10 prediction confidence indexes
    top_10_pred_index = pred_prob.argsort()[::-1][:10]

    ## find the top 10 prediction confidence values
    top_10_pred_values = pred_prob[top_10_pred_index] 
    
    ## find the top 10 labels
    top_10_pred_labels = unique_breeds[top_10_pred_index]
    
    ## setup plot
    top_plot = plt.bar(np.arange(len(top_10_pred_labels)),top_10_pred_values,color='blue')
    plt.xticks(np.arange(len(top_10_pred_labels)),labels=top_10_pred_labels,rotation='vertical')

    ## change the color of true label
    if np.isin(true_label,top_10_pred_labels):
        top_plot[np.argmax(top_10_pred_labels == true_label)].set_color('green')
    else:
        pass
    
    # plt.show()     
     
# print(np.argsort(predictions[0])[::-1][:10])

# print(predictions[0][predictions[0].argsort()[::-1][:10]])

print("Val Labels after unbatch")
print(val_labels)
    
## now we've got some functions to help us visualize our predictions and evaluate our model,Lets check out a few

i_multiplier = 0
num_rows = 5
num_cols = 2
num_images = num_rows * num_cols

plt.figure(figsize=(10 * num_cols,5 * num_rows))

ran = np.random.randint(1,len(y_valid),10)


# for i,j in enumerate(ran):
#     plt.subplot(num_rows,2 * num_cols,2*i+1)
#     plot_pred(predictions,val_labels,val_images,j)
    

#     plt.subplot(num_rows,2 * num_cols,2*i+2)
#     plot_pred_conf(predictions,val_labels,j)
    

# plt.show()
   
## plot the confusion matrix

conf = multilabel_confusion_matrix(val_labels,arg_max_predict)

# ConfusionMatrixDisplay(conf).plot()

# plt.show()
    

## saving and reloading a trained model

def save_model(model,suffix=None):
    
    '''
    Saves a given model in Models Directory and appends a suffix (string)
    '''
    modeldir = os.path.join('D:/Visual Studio Projects/Machine Learning Project Daniel Boruke/Models',datetime.datetime.now().strftime("%Y%M%D - %H%M%S"))

    model_path = modeldir + '-' + suffix + ".h5"  ## save format of the model
    
    print(f"Saving model to: {model_path}...")
    model.save(model_path)
    
    return model_path
    

## create a function to load a trained model

def load_model(model_path):
    print("loading saved model from:",model_path)
    
    model = keras.models.load_model(model_path,{"KerasLayer":hub.KerasLayer})
    
    return model


## now we've got functions to save and load models lets make sure they work
## save our model on 1000 images

# save_model(model1,suffix='1000-Images-Mobilenetv2')

# model1.save('Models')

# ## load our saved model

# loaded_model = load_model('Models/')

# print(loaded_model.summary())

# ## save the weight of the model
# model1.save_weights('Models')

## training a big dog model on full Data

## create a databatch with full dataset

# full_data = create_data_batches(x,y)

## create a model for full Model 

# full_model = create_model()

## create full model callbacks
full_model_tensorboard = create_tensorboard_callback()

## no validation set when we are training on all the available data , so we cant monitor validation accuracy

full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy',patience=2)

## fit the full model to the full data


## note : RUnning the cell below take a little while(Maybe upto 30 Minutes for the first epoch)
# full_model.fit(x=full_data,epochs=NUM_EPOCHS,batch_size=batch_size,callbacks=[full_model_tensorboard,full_model_early_stopping])

## save the full_model
# full_model.save('Model2')


## load the fullysaved model
full_loaded_model = keras.models.load_model('Model2/')


print(full_loaded_model.summary())


full_predictions = full_loaded_model.predict(val_data,verbose=1)

arg_max_full_predict = []

for i in range(0,len(full_predictions)):
    
    arg_max_full_predict.append(unique_breeds[np.argmax(full_predictions[i])])

## plot the confusion matrix for full_loaded model
conf_full = multilabel_confusion_matrix(val_labels,arg_max_full_predict,labels=unique_breeds)

print("COnfusion Matrix Display")

print(conf_full)

print(conf_full.shape)

## plot the confusion matrix

# def plot_multi_label_confusion_matrix(multi_label_conf_matrix,labels):
    

#     for i in range(0,120):
#         plt.subplot(12,10,i+1)     
#         sns.heatmap(multi_label_conf_matrix[i],annot=True,cbar=False,cmap='Blues')
#         plt.axis('off')
#         plt.title(labels[i],fontdict=dict(fontsize='small'))

#     plt.tight_layout()
        
#     plt.show() 

     
# plot_multi_label_confusion_matrix(conf_full,unique_breeds)
 

# print(unique_breeds[15])

def full_plot_predictions():
    
    for i in range(0,30):
        color = ("green" if arg_max_full_predict[i] == unique_breeds[np.argmax(y_valid[i])] else "red")
        plt.subplot(6,5,i+1)
    
        plt.title(f"Pred:{arg_max_full_predict[i]} True:{unique_breeds[np.argmax(y_valid[i])]} | P:{np.max(full_predictions[i])*100:.2f}",fontdict=dict(fontsize='x-small',fontfamily='sans-serif',fontstyle='oblique'),loc='center',color=color)
        plt.imshow(plt.imread(x_valid[i]))
        plt.axis('off')
    plt.show()    
    

    
full_plot_predictions()

# print(os.listdir('Test/'))

## get all the images of the Test Folder
test_images = ["Test/" + fname for fname in os.listdir('Test/')]

# print(test_images)

## convert the Test images into Test dataset and perform the Preprocessing 

test_images_dataset = create_data_batches(test_images,y=None,test_data=True)

## and predict the test_images_dataset

test_data_predictions = full_loaded_model.predict(test_images_dataset)

# create a random number b/w 1,10000 upto 20 Images


## lets plot the random test image prediction and its probability
def plot_test_predictions(test_predictions,test_images_path,images_to_plot=20,no_of_test_images=10000):
    random_test_numbers = np.random.randint(1,no_of_test_images,images_to_plot)
    for i,j in enumerate(random_test_numbers):
        plt.subplot(4,5,i+1)
        plt.imshow(plt.imread(test_images_path[j]))
        
        plt.title(f"Pred : {unique_breeds[np.argmax(test_predictions[j])]} | Prob: {np.max(test_predictions[j])*100 :.2f}",color='blue',fontdict=dict(fontsize='small'))
        plt.axis('off')

    plt.show()
        

plot_test_predictions(test_data_predictions,test_images,20,10000)






        










