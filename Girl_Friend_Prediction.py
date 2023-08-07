import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay


def label_conversion(column):
    unique_values = column.unique()
    labels = {value: index for index, value in enumerate(unique_values)}
    return column.map(labels)


def get_encoded_value(value, mapping):
    return mapping.get(value, -1)  # Returns -1 if value is not in the mapping.


def get_user_input(feature, mapping):
    while True:
        print(f"Enter {feature} from {', '.join(mapping.keys())}: ")
        user_input = input().strip()
        encoded_value = get_encoded_value(user_input, mapping)
        
        if encoded_value != -1:
            return encoded_value
        else:
            print("Invalid input. Please enter a valid value.")


# Load data
pd.set_option('display.max_columns', None)
df = pd.read_csv("girl1.csv")

# Features for label conversion
features_to_convert = ["Coffee", "Program", "Music", "Movies", "Book", "Pet", "Travel", "Color", "Outcome"]

# Encode mappings for reference and label conversion
encodings = {}
for feature in features_to_convert:
    encodings[feature] = {value: index for index, value in enumerate(df[feature].unique())}
    df[feature] = label_conversion(df[feature])

# Split data
features = ["Coffee", "Program", "Music", "Movies", "Book", "Pet", "Travel", "Color"]
X = df[features]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the model
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)

# Make predictions
y_pred = dtree.predict(X_test)

# Define outcome_decoding here
outcome_decoding = {v: k for k, v in encodings["Outcome"].items()}

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
display_labels = [outcome_decoding[i] for i in df["Outcome"].unique().tolist()]
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
cm_display.plot()
plt.show()


tree.plot_tree(dtree,feature_names=features)
plt.show()



# User input section
user_data = []
for feature in features:
    encoded_value = get_user_input(feature, encodings[feature])
    user_data.append(encoded_value)

# Predict based on user input
predicted_outcome = dtree.predict([user_data])[0]
outcome_decoding = {v: k for k, v in encodings["Outcome"].items()}
print(f"The prediction is: {outcome_decoding[predicted_outcome]}")
