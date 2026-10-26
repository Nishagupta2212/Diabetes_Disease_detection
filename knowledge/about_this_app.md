# About this app, its dataset and model
Source: Diabetes Risk Check app documentation; Smith et al. 1988, "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus" | https://github.com/Nishagupta2212/Diabetes_Disease_detection

## The PIMA Indians Diabetes dataset
The model was trained on the PIMA Indians Diabetes dataset, originally from the US National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK). It contains 768 women aged 21 or older of Pima Native American heritage living near Phoenix, Arizona, of whom 268 (about 35%) had diabetes. The Pima community has one of the highest recorded rates of type 2 diabetes in the world.

## Features in the dataset
The eight inputs are: number of pregnancies; plasma glucose 2 hours after an oral glucose tolerance test (mg/dL); diastolic blood pressure (mm Hg); triceps skin fold thickness (mm); 2-hour serum insulin (µU/mL); body mass index (kg/m²); the diabetes pedigree function; and age in years.

## The diabetes pedigree function
The diabetes pedigree function is a score that summarises family history of diabetes, giving more weight to closer relatives and to relatives who developed diabetes at a younger age. Higher values mean a stronger family history. In this dataset it ranges from about 0.08 to 2.4 with a median near 0.37.

## Missing values
In the original dataset a value of 0 for glucose, blood pressure, skin thickness, insulin or BMI means the measurement was not recorded. The app treats a 0 in these fields as unknown and fills it with the median value from the training data.

## How the model works
The app tested several algorithms (linear and RBF support vector machines, random forest and logistic regression) and selected the one with the best cross-validated accuracy, which was logistic regression. Inputs are standardised before prediction. In repeated cross-validation the model is correct about 77% of the time. Glucose has by far the strongest effect on the prediction, followed by BMI, number of pregnancies and the diabetes pedigree function.

## Limitations
The model was trained only on adult women of one specific population, so its estimates may be less accurate for men, younger people or other ethnic groups. It uses a 2-hour glucose test value, which most people do not know offhand, and it cannot replace tests such as fasting glucose or HbA1c. It is an educational tool and not a medical diagnosis; anyone concerned about diabetes should talk to a healthcare professional.
