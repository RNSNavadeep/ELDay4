Breast Cancer Detection — Logistic Regression

An end-to-end Machine Learning classification project that predicts whether a tumor is Malignant (cancerous) or Benign (non-cancerous) using Logistic Regression, built on the Breast Cancer dataset.


📁 Dataset


File: data.csv
Target Column: diagnosis (M = Malignant, B = Benign)
Features: 30 numeric features including radius_mean, texture_mean, perimeter_mean, and more
Dropped Column: Unnamed: 32 — empty/irrelevant column



🔄 Workflow

1. 🔍 Data Exploration

Loaded the dataset and performed initial checks:


Displayed first 5 rows, shape, and column names
Checked data types and info
Checked for null values and duplicate rows


2. 🧹 Data Cleaning


Dropped the irrelevant column Unnamed: 32
No duplicate rows found


3. 🏷️ Label Encoding

Converted the diagnosis column from text to numbers:


M (Malignant) → 1
B (Benign) → 0


pythonle = LabelEncoder()
df["diagnosis"] = le.fit_transform(df[["diagnosis"]])

4. 🔎 Outlier Detection

Used Z-score to detect extreme values:


Computed Z-scores for all columns
Counted values with Z-score > 3
Visualized radius_mean using a Boxplot


5. ⚖️ Feature Scaling

Applied StandardScaler to normalize all features before training — essential for Logistic Regression to perform well:

pythonss = StandardScaler()
x = ss.fit_transform(x)

6. ✂️ Train-Test Split

Split data into 80% training and 20% testing:

pythonxtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=42)

7. 🤖 Model Training

Trained a Logistic Regression model on the training data:

pythonmo = LogisticRegression()
mo.fit(xtr, ytr)

8. 📊 Model Evaluation

Evaluated using four key classification metrics:

MetricDescriptionAccuracyOverall correct predictions out of totalPrecisionOf all predicted malignant, how many were actually malignantRecallOf all actual malignant, how many were correctly identifiedConfusion MatrixTable showing TP, TN, FP, FN counts

9. 🟥 Confusion Matrix Heatmap

Visualized the confusion matrix using seaborn.heatmap() to clearly see:


✅ True Positives & True Negatives (correct predictions)
❌ False Positives & False Negatives (wrong predictions)



🛠️ Libraries Used

LibraryPurposepandasData loading and manipulationnumpyNumerical operationsmatplotlibBoxplot visualizationseabornConfusion matrix heatmapscipyZ-score outlier detectionscikit-learnEncoding, scaling, model training, evaluation


🧠 What is Logistic Regression?

Despite its name, Logistic Regression is a classification algorithm. It predicts the probability that a sample belongs to a class (0 or 1), using the sigmoid function:

P(y=1) = 1 / (1 + e^(-z))

If probability > 0.5 → Malignant, else → Benign.


📌 Why These Metrics Matter in Medical Data?


Recall is most important here — missing a malignant tumor (False Negative) is far more dangerous than a false alarm
High recall means the model correctly catches most cancer cases
Precision and Accuracy together confirm the model isn't raising too many false alarms



✅ Conclusion

This project demonstrates a complete ML classification pipeline — detecting breast cancer from tumor measurements. Logistic Regression with StandardScaling proved effective, and the confusion matrix heatmap provides a clear picture of where the model succeeds and where it needs improvement.
