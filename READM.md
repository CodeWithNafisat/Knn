# KNN Classification: Nursery Applications and Steel Plate Faults

Two multi-class classification projects built with K-Nearest Neighbors, both on datasets with uneven class sizes. The first predicts nursery application outcomes and reaches 0.95 accuracy. The second predicts the type of fault in steel plates and reaches 0.72 accuracy, a harder problem with seven fault types and only 1,941 records.

The common thread is how class imbalance shaped the work: choosing resampling, choosing K, and reading per-class results instead of trusting one overall score.

---

## Project 1: Nursery Application Outcomes

### The problem

Predict the outcome of a nursery application (one of five decision categories) from family and social factors such as parents' situation, housing, finances, social circumstances and health.

### The data

The dataset has 12,960 records with eight categorical features (`parents`, `has_nurs`, `form`, `children`, `housing`, `finance`, `social`, `health`) and the target `decision`. There were no missing values or duplicates, and I dropped the `id` column.

The target is badly uneven:

- not_recom: 4,320
- priority: 4,266
- spec_prior: 4,044
- very_recom: 328
- recommend: 2

One class has only two records in the entire dataset, which shaped most of the decisions below.

### What I did

- Label-encoded every column and split the data 80/20 with stratification (10,368 train, 2,592 test).
- Applied SMOTE to the training set only, with `sampling_strategy='minority'` and `k_neighbors=1`. The single neighbour was necessary because the smallest class has so few records.
- Swept K from 1 to 15. Test accuracy rose from 0.79 at K=1 to a plateau around 0.95 to 0.96 between K=7 and K=11, so I chose K=7.
- Used distance weighting (`weights='distance'`) so that closer neighbours count for more.

### Results

On the test set the model reached an accuracy of 0.95, with weighted precision of 0.96, recall of 0.95 and F1 of 0.95.

The per-class results matter more than the overall figure:

- **not_recom** was classified perfectly (precision and recall of 1.00).
- **priority** (F1 0.93) and **spec_prior** (F1 0.95) were strong.
- **very_recom** is the weak spot: precision of 1.00 but recall of only 0.48 across its 66 test records, meaning the model missed about half of them.
- **recommend** has no records in the test set, so it could not be evaluated. With two records in the whole dataset, KNN did not learn it.

SMOTE with the `'minority'` strategy only resamples the single smallest class, which here is `recommend`. It does nothing for `very_recom`, which is the class that is actually underperforming. K was also picked by comparing accuracy on the test split, so the 0.95 is slightly optimistic. A resampling strategy covering every small class, and K chosen by cross-validation, are the natural next steps.

---

## Project 2: Steel Plate Fault Classification

### The problem

Identify the type of fault in a steel plate from measurements taken during production. Catching the fault type early supports quality control, so defective plates can be handled sooner.

### The data

The dataset has 1,941 records, 27 numeric features and a target with seven fault types: Pastry, Z_Scratch, K_Scratch, Stains, Dirtiness, Bumps and Other_Faults. The raw columns were named V1 to V27, so I renamed them to descriptive names such as `Pixels_Areas`, `Sum_of_Luminosity`, `Steel_Plate_Thickness` and `Orientation_Index`. There were no missing values or duplicates.

The classes are uneven. Dirtiness is the smallest (11 records in the test set) and Other_Faults the largest (135).

### What I did

- Looked at boxplots for every numeric feature. Many had heavy outliers, and I grouped them by how severe the outliers were.
- Label-encoded the target and split the data 80/20 with stratification (1,552 train, 389 test).
- Applied SMOTE to the training set only, to balance the fault types.
- Standardized the features with a scaler fitted on the resampled training data, keeping the column names so the features stay traceable.
- Used RFE with a logistic regression to keep 19 of the 27 features. RFE cannot be run on KNN directly, so logistic regression stood in as the selector.
- Swept K from 1 to 15 and chose K=7.

### Results

Performance varies a lot by fault type (classes are numbered in alphabetical order by the encoder, so I have matched them to names here). K_Scratch is the best at F1 0.95, with precision of 0.96 and recall of 0.94. Z_Scratch, Dirtiness and Stains follow with F1 between 0.75 and 0.83. The weakest are Pastry (F1 0.46, precision 0.37) and Other_Faults (recall 0.53 on the largest class, 135 records), followed by Bumps (F1 0.66).

Test accuracy stayed between roughly 0.70 and 0.72 for every K from 1 to 15, while training accuracy fell from 1.00 to 0.86. So the choice of K was not what limited the score. K was again chosen by looking at accuracy on the test split.

---

## What I Took From Both

- The overall accuracy hides the real behaviour. In both projects the interesting findings were at the class level: a class too small to learn, a class the model misses half the time, and a catch-all class with low recall.
- Resampling has to match the problem. In the nursery model, SMOTE targeted the two-record class and did not address `very_recom`, the class that was actually failing, so imbalance handling needs to target the classes that underperform.
- A flat accuracy curve across K says the limit is in the features and classes, not the value of K.

## Tech Stack

Python, pandas, NumPy, scikit-learn (KNeighborsClassifier, LogisticRegression, RFE, StandardScaler, LabelEncoder), imbalanced-learn (SMOTE), SciPy, seaborn, Matplotlib and Jupyter Notebook.

## Project Structure

```
├── Knn_Task.ipynb
├── nursery.csv
├── php5s7Ep8.csv
└── README.md
```
