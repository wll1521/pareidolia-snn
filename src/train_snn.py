import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import load_faces_in_things, load_lfw_faces, make_pairs
from model_snn import build_snn

# Load data
faces = load_lfw_faces()
nonfaces = load_faces_in_things()

pairs, labels = make_pairs(faces, nonfaces, n_pairs=4000)
train_pairs, test_pairs, y_train, y_test = train_test_split(pairs, labels, test_size=0.2)

# Prepare inputs
def split_pairs(pairs):
    a = np.array([x[0] for x in pairs])
    b = np.array([x[1] for x in pairs])
    return [a, b]

# Build and train model
model = build_snn()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(split_pairs(train_pairs), y_train, validation_split=0.2, epochs=10, batch_size=32)

model.save("snn_model.h5")

# Evaluate
loss, acc = model.evaluate(split_pairs(test_pairs), y_test)
print(f"Test Accuracy: {acc*100:.2f}%")
