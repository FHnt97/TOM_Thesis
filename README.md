# TOM_Thesis
Code for MSc Thesis:
### Fairness-aware development of algorithms: An explanaotry study on bias in occupational stress& fatigue datasets.

Add code was written through Google Collab, due to technical limitations. Helper funcitons are therefore saved in the notebook, rather than saving them separately and using the import function. 

For future work, with the use of Jupyter Notebooks, this can be improved. 

# Introduction
In this notebook, we consider a scenario where algortihmic tools are used to detect stress patterns in an occupational setting.

We train diverse fairness-unaware algortihms on datasets on occupational stress and fatigue, with the aim to prove that there is a underlying bias in the methodology.

The objective of the algorithms is to detect (classification algorithm) the state of the employee to better accomodate their working conditions and allow for more productive work.

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


In practice, these questions should be covered through the creation of a model card.


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
