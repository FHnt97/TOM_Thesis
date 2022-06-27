# Fair AI in workspaces: an exploratory study on data-driven occupational stress and fatigue understanding

Master Thesis
Faculty of Economics and Business
MSc. Technology and Operations Management

> **NOTE** - 
All code was written through Google Collab due to technical limitations. <br />
Helper functions are therefore saved in the notebook, rather than saving them separately and using the import function. <br />
Additionally, due to time limitations, the graphics for the results were done using Excel, using copy past - an improvement will be to extract results into dataframes and save as csv. <br /> 
For future work, with the use of Jupyter Notebooks, this can be improved. 

# Introduction
In this notebook, we consider a scenario where algorithmic tools are used to detect stress patterns in an occupational setting.
<br /><br />
We train diverse fairness-unaware algorithms on two datasets containing information on occupational stress and fatigue, with the aim to prove that there is an underlying bias when applying these ML models. Additionally, diverse bias mitigation techniques will be discussed, applied, and compared.
<br /><br />
The objective of the algorithms used is to detect (classification algorithm) the state of the employee to better accommodate their working conditions and allow for more productive work.

# Fairness in the context of Operations Management
Machine learning algorithms are increasingly used for diverse decision-making processes. Consequently, the impact of algorithmic decisions on people’s lives, and with it, the effect of unintended biases in algorithms, has increased. These give rise to undue discrimination and inequalities between separate groups, leading to social, ethical, and legal issues. Therefore, data bias must be considered, especially in fields involving data use on human beings. 
<br /><br />
Especially in the field of Operations Management (OM), and specifically in the study of Human-Robot Interaction and Collaboration, bias is not a topic that is often discussed when building ML models. It is, however, crucial for the health and safety of employees, that models used in applications such as performance-based fatigue and stress detection systems perform equally for all, to avoid the safety and health risks involved with these factors. 

> The managing fatigue and stress is important in OM, as they can lead to reduced performance, quality of work and productivity, alongside increased risk of human errors and labour accidents. It is widely recognised that work-related stress and fatigue can be predecessors to many detrimental short-term and long-term health  outcomes. As stated by Maman et al. in their work “[…] the health-related lost productivity time for fatigued workers exceeds double their non-fatigued counterparts. The financial ramifications of fatigue outcomes are estimated to cost U.S. employers approximately $136 billion annually”. This highlights the importance of evading stress and fatigue equally for all employees, independently of how they are categorised, in the occupational setting. 

<br /><br />
Additionally, in Operations Management, practitioners often have to make cost/benefit decisions, facing the problem of accuracy vs. fairness, as well as the question of whether models should be made on a general or personal level. This is an essential question as personalised models are significantly more expensive to set up, and general models that work with person-specific data can have underlying biases.
<br /><br />
Specifically for the deployment of performance-based fatigue and stress detection systems in the field of Operations Management, an unfair model can lead to reduced performance, quality of work and productivity, alongside increased risk of human errors and labour accidents.
<br /><br />
The question therefore arises:

- Do datasets on occupational stress and fatigue contain significant bias originating from sensitive attributes? 
- Would improved fairness allow practitioners to create generalised models (vs personalised ones) that are fair towards all groups present in the field/ that don't discriminate against minority groups?
- What state-of-the-art methodologies proposed by the research community are adequate to de-bias a model used to detect the levels of stress in employees?

# Methodology:

For each dataset, the experimental design is split into four stages:
1. An exploratory analysis of the data to understand the relationship between the sensitive attributes (age & gender) and other attributes.
2. The set up of three ml models (RandomForest/k-NN, LogReg, SVM) for the detection of stress/fatigue, followed by an analysis of the performance metric.
3. Fairness metrics for each model regarding False Negative Rate and Equalised Odds regarding age & gender.
4. Implementation of preprocessing (reweighing) and post-processing bias mitigation methods & comparison of effectiveness.

For further discussion, a convoluted nn has been set-up to detect stress on an individual level for the WESAD dataset. After the set-up of the code, we analysed the results for bias. 
As a next step in the research, we propose applying some of the most commonly used mitigation methods on this ML model (CFAIR, etc.) & comparing results to understand if group fairness can be improved. 


### Repository set-up
Due to the extensiveness of evaluating the different cases, this research project focuses on the analysis of single sensitive attributes for each experimental setting. <br /><br />
For each setting a new notebook has been created, for the clarity of the process and the code. 
Additionally, to avoid repetition and for brevity, comments on the methodology and the data exploration have been removed from the 2.0 labelled notebooks. <br /><br />
Furthermore, only the first notebook labelled WLK+MMH 1.0 contains an extensive analysis of the individual models, containing feature selection and minimal parameter tuning. 
<br /><br />
The notebooks in the repository are as follows:
  1. WLK_+_MMH_Debiasing_Classification_Algorithms_1_0_AGE
  2. WLK_+_MMH_Debiasing_Classification_Algorithms_2_0_TASK
  3. WESAD_Debiasing_Classification_Algorithms_1_0_AGE
  4. WESAD_Debiasing_Classification_Algorithms_1_0_GENDER

Additionally, for the section of further discussion, the notebook with the CNN has also been made available. Note that this is an incomplete project, as bias has only been detected, but not further analysed using fairness metrics nor mitigated. 
 - WESAD_Fairness_evaluation_of_NN


# The Datasets:
## WESAD
This  dataset is a publicly available multimodal physiological dataset for wearable stress and affects detection created in 2018 through a corporate research project by Robert Bosch. Physiological signals were recorded during a laboratory study in which 15 participants were exposed to three different affective states: neutral, stress, and amusement. The records on amusement have been disregarded for this study, leaving, on average, 66 entries per participant. <br /><br />
The dataset was created in 2018 by Schmidt et al., with the primary objective to fill the gap for commonly used standard datasets for wearable stress detection, and incorporate multiple affective states into the study of occupational stress. In the initial paper published on the dataset, the researchers propose common machine learning methods, ranging from Decision Trees (DT) to Random Forest (RF), Ada Boost (AB), to k-Nearest Neighbour (k-NN) with k=9. All proposed models have additionally been evaluated for their cross-validation (CV) score .


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

The data was extracted from the raw data files through the code presented in the notebook: Data creation - WESAD, found in the code folder.

**SOURCE:**

[https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)


# MMA & WLK Data
These datasets have been provided by a research consortium from the Adelphi University, Auburn University and the University of Buffalo. The primary use of the data is to detect and predict fatigue in employees doing manual material handling activities, simulating manufacturing tasks. The first of the datasets presents a simulated manual material handling (MMH) task, the second (WLK) entails information on a supply pick-up and insertion (SI) task. Overall, the dataset contains information on 28 participants, with 18 entries on average per subject. The publicly available dataset for this research has been cleaned, parameters calculated, and outliers removed prior to the upload. The raw data is not available. 
<br /><br />
The authors of the original research paper, written by Maman et al. use RF classifiers (n esti-mators = 100), logistic regression (solver = “newton-cg”, C = 1,000,000), and a Support-Vector-Machine (SVM) model (kernel= “rbf”, C = 64, class weight = “balanced”) for the classification task.
<br /><br />
For this experiment, and contrary to the proposed research from the original author, the MMH and WLK datasets have been merged to analyse whether a second, non-human attribute “task” could be considered a sensitive attribute. 
<br /><br />
Fatigue is measured through HR and INN data.
<br /><br />
The age of participants will be rounded to the closest 10 years in order to group participants into groups.
<br /><br />
The columns contain mainly boolean and categorical data (including age and various test results).

**SOURCE:**

[https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29
](https://zahrame.github.io/FatigueManagement.github.io/)
