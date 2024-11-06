# Pulmo-AI
## Aim:
Our goal with this project is to provide healthcare professionals and patients a universal tool for diagnosing and predicting lung cancer based on several factors including CT-scans, and other biometric data.

## Description:
There are two parts to this project, one being the ct-scan classifier model, and the other based on patient specific data.

### CT-Scan model
This model is used to detect various types of lung cancers and its location based on a patient's ct-scans. By utilizing advanced deep learning architectures such as DenseNet, it processes high-resolution CT images to identify abnormalities in lung tissues. The model classifies different forms of lung cancer, including adenocarcinoma, large cell carcinoma, squamous cell carcinoma. It is trained on a large dataset of labeled CT images to ensure high accuracy and generalizability. This model can help in early detection, diagnosis, and potentially assist in treatment planning by providing insights into tumor location and size.
    The model is based on the PyTorch Densenet model which is a dense Convolutional Neural Network (CNN). By leveraging DenseNet's architecture, the model can effectively analyze CT-scan images, detecting various types of lung cancers and their specific locations. This approach enhances accuracy in distinguishing different forms of lung cancer while maintaining computational efficiency. 

### Patient data model
The aim for this model is to provide a percentage prediction of a patient's likelihood of developing a form of lung cancer based on thier medical data. The medical data includes factors like air polution, genetic risks, diet, shortness of breath, fatigue, etc.

The model utilizes Random Forest Regression, a machine learning algorithm that combines multiple decision trees to improve predictive accuracy and reduce overfitting. By using an ensemble of trees, the model can capture complex interactions between different features, making it well-suited for predicting lung cancer risk from diverse data. The model is trained on a dataset of patient records, ensuring that it can effectively weigh the significance of each factor and provide a robust probability score indicating a patient's risk level.

This predictive model can be used as a tool for early detection, personalized health monitoring, and guiding preventive care strategies.

### Disclaimer:
This model is intended for informational and research purposes only and should not be used as a substitute for professional medical diagnosis, advice, or treatment.