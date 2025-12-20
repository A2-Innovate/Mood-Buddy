import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("face_train.csv")
test_df = pd.read_csv("face_test.csv")

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

print("Training label distribution:")
print(y_train.value_counts())

print("\nTest label distribution:")
print(y_test.value_counts())

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print("\nEncoded classes:")
print(dict(zip(le.classes_, range(len(le.classes_)))))


print("\nTraining XGBoost Model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",     
    num_class=len(le.classes_),     
    tree_method="hist",             
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train_encoded)
y_pred = np.argmax(model.predict_proba(X_test), axis=1)

accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, "buddy_xgb_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\n✅ Model and LabelEncoder saved successfully.")
