# TOM_Thesis
Code for MSc Thesis:
### Fairness-aware development of algorithms: An explanaotry study on bias in occupational stress& fatigue datasets.

All code was written through Google Collab, due to technical limitations. Helper funcitons are therefore saved in the notebook, rather than saving them separately and using the import function. 

For future work, with the use of Jupyter Notebooks, this can be improved. 

# Introduction
In this notebook, we consider a scenario where algortihmic tools are used to detect stress patterns in an occupational setting.

We train diverse fairness-unaware algortihms on two datasets containing information on occupational stress and fatigue, with the aim to prove that there is a underlying bias when applying these ML models. Additionally, diverse bias mitigation techniques will be discussed, applied, and compared.

The objective of the algorithms used is to detect (classification algorithm) the state of the employee to better accomodate their working conditions and allow for more productive work.

# Fairness in the context of Operatiosn Management
Fairness is complex & contextual, there is no one-size fits all appraoch.

For Operations Management, practitioners often have to make cost/benefit decisions, facing the problem of accuracy vs. fairness, as well as the question of weather models should be made on a general, or personal level.

The question therefore arises:

- Can we create generalised models (vs. personalised ones), that are fair towards all groups present in the field/ that don't discriminate minority groups?

This is an important question as personalized models are significantly more expensive to set up, and general models that work with person-specific data can have underlying biases.

Furthermore, when creating general codes for people, it is important to consider:

- Who will the product empower/service and who will be left out? The objectives have to be clear

- Who is writing the code & is the team/ person aware of diverse and inclusive practices

- Whose data is inlcuded in the process, and how this is collected. Topics to consider include: - historical patterns, consent, privarcy, exclusion of specific gorups, etc.

- Who can monitor the outcome and how? Is the model transparent & has it been well documented?

- Does the outcome have a disciminatory impact? and, if so, how can the negative impact be rectified?

We also have to consider the taxonomy of possible bias sources:

- Selection bias, reporting bias, sampling bias
- Interpretation bias - correlation falacy (correlation =! causation); overgeneralization, automation bias


In practice, these questions should be covered through the creation of a model card. They will also be covered in the companion report to the repository. 

# Methodology:

For each dataset, the experimental design is split into four stages:
1. An exploratory analysis of the data, to understand the realtionship between the sensitve attributes (age & gender) and other attributes.
2. The set up of three ml models (RandomForest/k-NN, LogReg, SVM) for the detection of stress/fatigue, followed by an analysis of the performance metric.
3. Fairness metrics for each model in regards to False Negative Rate and Equal Opportunity in regards to the age & gender.
4. Implementation of pre-processing (reweighing) and post-processing bais mitigation methods & comparison of effectiveness.

For further discussion, a convoluted nn has been set up, to detect stress on an individual level for the WESAD dataset. After the set up of the code, we analysed the results for bias. 
As a next step to the research, we propose applying some of the most commonly used mitigation methods on this ML model (CFAIR, etc.) & comparing results to understand if group fairness can be improved. 

# The Datasets:
## WESAD
WESAD is a publicly available multimodal physiological dataset proposed for wearable stress and affect detection. The signals were recorded during a lab study in which 15 participants with a mean age of 27.5 years (SD = 2.4) were exposed in three different affective states: neutral, stress, and amusement. Regarding stress, the Trier Social Stress Test (TSST) was employed by the researchers in order to elicit the specific emotion.

The dataset consists of the following physiological signals (continuous variables):

- blood volume pulse (BVP)
- electrocardiogram (ECG)
- electrodermal activity (EDA)
- electromyogram (EMG)
- respiration (RESP)
- skin temperature (TEMP)
- three axis acceleration (ACC)

Categorical variables available are:
- gender & age

The data was extracted from the raw data files through the code presented in the notebook: Data creation - WESAD.

ON TOP: REDUCED REPRESENTATION OF THE UNPRIVILEDGED GROUP (less women in sample) - it is assumed that the sensitive/ biased attribues are knon apriori & can be adressed by removing these features from the learned representation. However, some features can correlate with age/ gender indirectly --> if this is not known, this bias cannot be removed.

https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29

Dataset instances: 15 participants

Data collection: refer to https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29

# MMA & WLK Data
The MMA& WLK datasets has been provided by -----, with the main use to detect and predict fatigue in employees doing manual material handling activities.

The first present a simulated manual material handling (MMH) task, and the second is a supply pick-up and insertion (SI) task.

Twenty four participants (9 females, 15 males; mean age 36.37 years with the standard deviation of 16.67 years) partook in the test. All participants reported that they were in good physical and mental health. Participants completed one three-hour experimental session for the simulated MMH task and another for the SI task. The order of the two experiments was randomized and participants had to complete the experiments in different days. As a result, we ended up with 15 participants whose data were deemed reliable for analysis.

Note that the data has been cleaned, and the needed parameters for the model have been calculated by the owner of the dataset (sensor selection, data preprocessing and feature generation (HRR, jerk, etc)), and unchaged features have been removed.

Fatigue is measured through HR and INN data.

The age of participants will be rounded to the closest 10 years, in order to group participants into groups.

The columns contain mostly boolean and categorical data (including age and various test results).
