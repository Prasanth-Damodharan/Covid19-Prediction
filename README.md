# COVID19-Prediction

## Objective
The time taken in running a test to identify COVID-19 positive cases among patients has been a problem. Therefore the objective of this project is to create a deep learning model to identify if a patient is Covid-19 positive or not using their X-ray lung Images

## Dataset:
There are two data sets the training data set and the testing data set. Both data sets consist of X-ray images of the patients. The training data have a label if the image is COVID positive or not.

## Procedure:
The X-ray images under these labels had to be converted into matrices to pass them as input to the Neural Network. Therefore, the images from the training and testing data were fetched individually and were converted using the cv2 package. The shape of the images was first converted to a particular dimension so that all the images have the same dimension. Then the RGB code for each of the pixels in the image is fetched. These values are then stored in a matrix format. The same procedure is repeated for every image present in the data.

## Challenges faced :
There were multiple challenges while implementing the CNN on our X-ray images dataset. Below are some of the major challenges:
- Converting Images into matrices and passing them as inputs into the CNN
Converting every image into the right dimension and  format which can be used as input for the CNN
- Optimize metrics of the Neural Network
Analyzing and finding the right metrics to obtain the optimistic Neural Network model
- Improving the accuracy of the Neural Network
Multiple trial attempts with a different combination of neural layers, activation function, neurons, epochs had to be implemented to find the best metric for maximum accuracy
- Computation time
The computation time was very high due to the converted size of the images. Therefore, GPU functionality in the google collab

## CNN Layer metrics:
- loss = binary_crossentropy
The binary_crossentropy loss function is chosen since this problem deals with a binary problem
- optimizer = adam()
The adam optimizer is chosen since this is one of the best optimizer function used in Neural Network models
- Weight = imagenet
The imagenet weight function is selected for allocating the weights in this Neural model since this project deals with image classification.
- layers  ( Pooling / Flatten / Dense / Dropout )
These are the layers used in this Neural model. The input matrices are first passed through the pooling layer to reduce the size of the feature map and then passed through the flatten layer to have them as inputs for the dense neural layer with activation function as relu. Then finally it is passed to the output layer which has the activation function as sigmoid.
- Epochs = 25
The optimistic number of epochs was finalized as 25 after multiple trial runs
- Batch_size (BS) = 8
The optimistic Batch size was finalized as 8 after multiple trial runs
- steps_per_epoch = length of X_train / BS
The metric used in calculating the steps_per_epoch was the length of the X_train / BS

## Conclusion:
The Convolutional Neural Network proved to be an effective model on predicting if an X-ray image is COVID positive or negative with an accuracy of 94%

## Remarks:
- On plotting the confusion matrix, the number of False Negative cases was found to be 9. Since this is a medical experiment, the False Negative cases need to be handled carefully.
- Therefore the user is suggested to run the model thrice and the majority of the three results for an X-ray image should be considered as the result.
