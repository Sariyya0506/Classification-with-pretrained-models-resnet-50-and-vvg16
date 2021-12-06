#!/usr/bin/env python
# coding: utf-8

# <a href="https://cognitiveclass.ai"><img src = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/Logos/organization_logo/organization_logo.png" width = 400> </a>
# 
# <h1 align=center><font size = 5>Peer Review Final Assignment</font></h1>

# ## Introduction
# 

# In this lab, you will build an image classifier using the VGG16 pre-trained model, and you will evaluate it and compare its performance to the model we built in the last module using the ResNet50 pre-trained model. Good luck!

# ## Table of Contents
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# <font size = 3>    
# 
# 1. <a href="#item41">Download Data 
# 2. <a href="#item42">Part 1</a>
# 3. <a href="#item43">Part 2</a>  
# 4. <a href="#item44">Part 3</a>  
# 
# </font>
#     
# </div>

#    

# <a id="item41"></a>

# ## Download Data

# Use the <code>wget</code> command to download the data for this assignment from here: https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week4.zip

# Use the following cells to download the data.

# In[1]:


get_ipython().system('wget https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week4.zip')


# In[2]:


get_ipython().system('unzip concrete_data_week4.zip')


# After you unzip the data, you fill find the data has already been divided into a train, validation, and test sets.

#   

# <a id="item42"></a>

# ## Part 1

# In this part, you will design a classifier using the VGG16 pre-trained model. Just like the ResNet50 model, you can import the model <code>VGG16</code> from <code>keras.applications</code>.

# You will essentially build your classifier as follows:
# 1. Import libraries, modules, and packages you will need. Make sure to import the *preprocess_input* function from <code>keras.applications.vgg16</code>.
# 2. Use a batch size of 100 images for both training and validation.
# 3. Construct an ImageDataGenerator for the training set and another one for the validation set. VGG16 was originally trained on 224 × 224 images, so make sure to address that when defining the ImageDataGenerator instances.
# 4. Create a sequential model using Keras. Add VGG16 model to it and dense layer.
# 5. Compile the mode using the adam optimizer and the categorical_crossentropy loss function.
# 6. Fit the model on the augmented data using the ImageDataGenerators.

# Use the following cells to create your classifier.

# In[3]:


pip install tensorflow


# In[1]:


from tensorflow import keras 


# In[2]:


import tensorflow.keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[6]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[8]:


from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


# In[10]:


batch_size = 100
image_size = (224, 224)
num_classes = 2


# In[11]:


generator = ImageDataGenerator(preprocessing_function=preprocess_input)

training_generator = generator.flow_from_directory(
    "concrete_data_week4/train",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
)
validation_generator = generator.flow_from_directory(
    "concrete_data_week4/valid",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
)


# In[12]:


model_vgg16 = Sequential()

model_vgg16.add(VGG16(include_top=False, pooling="avg", weights="imagenet",))
model_vgg16.add(Dense(num_classes, activation="softmax"))

model_vgg16.layers[0].trainable = False

model_vgg16.summary()


# In[13]:


model_vgg16.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)


# In[14]:


num_epochs = 1
steps_per_epoch_training = len(training_generator)
steps_per_epoch_validation = len(validation_generator)

history_vgg16 = model_vgg16.fit_generator(
    training_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
)


# In[15]:


model_vgg16.save("classifier_vgg16_model.h5")


# In[ ]:





# In[ ]:





# In[ ]:





# I train resnet50 model here as I had problem with lab in week 3 where resnet50 model was trained.

# In[17]:


from tensorflow.keras.applications import ResNet50

from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input


# In[18]:


resnet50_data_generator = ImageDataGenerator(
    preprocessing_function=resnet50_preprocess_input,
)


# In[20]:


resnet50_train_generator = resnet50_data_generator.flow_from_directory(
    'concrete_data_week4/train',
    target_size=image_size,
    batch_size=batch_size,
      
    class_mode='categorical')


# In[21]:


resnet50_validation_generator = resnet50_data_generator.flow_from_directory(
    'concrete_data_week4/valid',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical')


# In[22]:


resnet50_model = Sequential()

resnet50_model.add(ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))

resnet50_model.add(Dense(num_classes, activation='softmax'))

resnet50_model.layers[0].trainable = False

resnet50_model.summary()


# In[23]:


resnet50_model.layers


# In[24]:


resnet50_model.layers[0].layers


# In[25]:


resnet50_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[26]:


resnet50_steps_per_epoch_training = len(resnet50_train_generator)
resnet50_steps_per_epoch_validation = len(resnet50_validation_generator)
num_epochs = 1


# In[30]:


resnet50_fit_history = resnet50_model.fit_generator(
    resnet50_train_generator,
    steps_per_epoch=resnet50_steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=resnet50_validation_generator,
    validation_steps=resnet50_steps_per_epoch_validation,
    verbose=1
)


# In[31]:


resnet50_model.save('classifier_resnet50_model.h5')


# In[ ]:





#    

# <a id="item43"></a>

# ## Part 2

# In this part, you will evaluate your deep learning models on a test data. For this part, you will need to do the following:
# 
# 1. Load your saved model that was built using the ResNet50 model. 
# 2. Construct an ImageDataGenerator for the test set. For this ImageDataGenerator instance, you only need to pass the directory of the test images, target size, and the **shuffle** parameter and set it to False.
# 3. Use the **evaluate_generator** method to evaluate your models on the test data, by passing the above ImageDataGenerator as an argument. You can learn more about **evaluate_generator** [here](https://keras.io/models/sequential/).
# 4. Print the performance of the classifier using the VGG16 pre-trained model.
# 5. Print the performance of the classifier using the ResNet pre-trained model.
# 

# Use the following cells to evaluate your models.

# In[ ]:





# In[27]:


testing_generator = generator.flow_from_directory(
    "concrete_data_week4/test", target_size=image_size, shuffle=False,
)


# In[28]:


performance_vgg16 = model_vgg16.evaluate_generator(testing_generator)
print("Performance of the VGG16-trained model")
print("Loss: {}".format(round(performance_vgg16[0], 5)))
print("Accuracy: {}".format(round(performance_vgg16[1], 5)))


# In[4]:


from tensorflow.keras.models import load_model

resnet50_model = load_model('classifier_resnet50_model.h5')


# In[32]:


performance_resnet50 = resnet50_model.evaluate_generator(testing_generator)
print("Performance of the ResNet50-trained model")
print("Loss: {}".format(round(performance_resnet50[0], 5)))
print("Accuracy: {}".format(round(performance_resnet50[1], 5)))


# In[ ]:





#    

# <a id="item44"></a>

# ## Part 3

# In this model, you will predict whether the images in the test data are images of cracked concrete or not. You will do the following:
# 
# 1. Use the **predict_generator** method to predict the class of the images in the test data, by passing the test data ImageDataGenerator instance defined in the previous part as an argument. You can learn more about the **predict_generator** method [here](https://keras.io/models/sequential/).
# 2. Report the class predictions of the first five images in the test set. You should print something list this:
# 
# <center>
#     <ul style="list-style-type:none">
#         <li>Positive</li>  
#         <li>Negative</li> 
#         <li>Positive</li>
#         <li>Positive</li>
#         <li>Negative</li>
#     </ul>
# </center>

# Use the following cells to make your predictions.

# In[29]:


predictions_vgg16 = model_vgg16.predict_generator(testing_generator, steps=1)


def print_prediction(prediction):
    if prediction[0] > prediction[1]:
        print("Negative ({}% certainty)".format(round(prediction[0] * 100, 1)))
    elif prediction[1] > prediction[0]:
        print("Positive ({}% certainty)".format(round(prediction[1] * 100, 1)))
    else:
        print("Unsure (prediction split 50–50)")


print("First five predictions for the VGG16-trained model:")
for i in range(5):
    print_prediction(predictions_vgg16[i])


# In[ ]:





# In[ ]:





#   

# ### Thank you for completing this lab!
# 
# This notebook was created by Alex Aklson.

# This notebook is part of a course on **Coursera** called *AI Capstone Project with Deep Learning*. If you accessed this notebook outside the course, you can take this course online by clicking [here](https://cocl.us/DL0321EN_Coursera_Week4_LAB1).

# <hr>
# 
# Copyright &copy; 2020 [IBM Developer Skills Network](https://cognitiveclass.ai/?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/).
