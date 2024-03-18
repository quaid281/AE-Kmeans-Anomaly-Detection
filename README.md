Overview
==============================


Stakeholders
==============================


Goals
==============================

Project Answers
==============================
 

**ABOUT YOUR DATA**
- Where did you get it? 
- Do you think you have enough? 
- How much data did you start with (rows, columns, size) and how much did you use to model? 
- Do you think you have the RIGHT data?  
- How could your dataset be improved? 

 

 

**FEATURE ENGINEERING AND DIMENSIONALITY STRATEGY**
- Are you sure you have the right target variable? How did you confirm? 
- What features are key? How do you know? 
- Did you reduce dimensionality? How did you decide to alter the size? 

 

 

**DATA PREP/PIPELINE**
- How do you know your data is appropriate and complete prior to modelling? 

**MODELLING** 
- What was your model selection process? 
- How did you go about the train / test split? 
- Are you sure there’s no data leakage? How do you know? 
- What are your thoughts on cross validation? 
- Was any hyperparameter tuning required? How did you tune? 
- What final model did you decide on? Why? 

 

**HOW ARE YOU THINKING ABOUT PRODUCTIZATION**
- What checks are in place, or are you envisioning, to ensure that final models are using appropriate, complete, unbiased data sets? 






Project Folder Organization
===========================

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── sql            <- sql queries
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),the creator's initials, and a short `-` delimited description, e.g. `1.0-jqp-initial-data-exploration`
    │                   
    |   ├── formal         <- Well documented Jupyter notebooks that serves as a starting point. Naming convention is a number
    |   └──informal        <- Exploration Jupyter notebooks that aren't necessarily ready to be shown to the business
    │                         .
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py


--------

<p><small>Project based on a modified version of <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
