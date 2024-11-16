# Pulmo-AI
## Aim:
Our goal with this project is to provide healthcare professionals and patients a universal tool for diagnosing and predicting lung cancer based on several factors including CT-scans, and other biometric data as well as providing preventive care.

## Description:
There are various components to this project which include the implimentation of risk predictions, lung cancer classification based on medical imagery, and a healthcare chatbot for preventative care. 

### CT-Scan model
This model is used to detect various types of lung cancers and its location based on a patient's ct-scans. By utilizing advanced deep learning architectures such as ResNet, it processes high-resolution CT images to identify abnormalities in lung tissues. The model classifies different forms of lung cancer, including adenocarcinoma, large cell carcinoma, squamous cell carcinoma. It is trained on a large dataset of labeled CT images to ensure high accuracy and generalizability. This model can help in early detection, diagnosis, and potentially assist in treatment planning by providing insights into tumor location and size.
    The model is based on the PyTorch ResNet model which is a Convolutional Neural Network (CNN). By leveraging ResNet's architecture, the model can effectively analyze CT-scan images, detecting various types of lung cancers and their specific locations. This approach enhances accuracy in distinguishing different forms of lung cancer while maintaining computational efficiency. 


![1*9SrzCTHIVgxzPu3VmvWmVw](https://github.com/user-attachments/assets/d578050a-9c4c-4ca2-9d0e-06403f15cff2)

^ Resent model

![000122 (6)](https://github.com/user-attachments/assets/83cfec45-60ca-4b7f-9db5-ba474a204f00)

^ example ct scan of Squamous Cell Carcinoma of a patient's lung. This cancer effects skin and organ lining.



### Patient data model
The aim for this model is to provide a percentage prediction of a patient's likelihood of developing a form of lung cancer based on thier medical data. The medical data includes factors like air polution, genetic risks, diet, shortness of breath, fatigue, etc.

The model utilizes Random Forest Regression, a machine learning algorithm that combines multiple decision trees to improve predictive accuracy and reduce overfitting. By using an ensemble of trees, the model can capture complex interactions between different features, making it well-suited for predicting lung cancer risk from diverse data. The model is trained on a dataset of patient records, ensuring that it can effectively weigh the significance of each factor and provide a robust probability score indicating a patient's risk level.

This predictive model can be used as a tool for early detection, personalized health monitoring, and guiding preventive care strategies.


![Unknown](https://github.com/user-attachments/assets/533db45d-67fd-4688-83d2-1a69f5f92afa)

^ Random Forest Regression Model

![image](https://github.com/user-attachments/assets/75495557-33ff-41ee-8360-3dcb4036d3b0)

^ this is the output of the regression model. It creates weights and assigns correlations to the specfic factors.


![image](https://github.com/user-attachments/assets/87ac9a9b-af61-4936-880e-faeaf2a48f7f)

^ correlation graph

### Healthcare Chatbot
As an important addition to this app, we added chat bot integration for disease preventative care and further patient engagement. 


# Installation

### Instructions

1. Clone this repository into NVIDIA AI Workbench and build with Docker.
2. Navigate to 'code/sdataprediction.ipynb' and run all the cells to run the data prediction model.
3. Navigate to 'code/GPT.ipynb' and run the cells to test the chatbot implementation.
4. Navigate to 'code/Limagediagnosis.ipynb' and run all cells to run the model that classifies the ct scans
5. Navigate to 'ImagePredict.py' and run the command
   **python3 ImagePredict.py**
   in the terminal. You can add your own ct scan file and update the path in ImagePredict if you would like to apply further test
   cases.
6. To test the Web Application, navigate to the environment tab on AI Workbench and add a new app.


## Submission Comments:
This app is still in development and is a prilimary step into what we believe will be a better and more comprehensive app. We plan to make changes to the Image Diagnosis, and chat bot implementation and the overall application. We plan to use NVIDIA Monai for the segmentation and detection of various types of lung anomalies based on a larger database of XRAYS.

## Disclaimer:
This model is intended for informational and research purposes only and should not be used as a substitute for professional medical diagnosis, advice, or treatment.
