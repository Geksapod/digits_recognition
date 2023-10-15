from keras.datasets import mnist
import cv2
from keras.utils import to_categorical
from keras.models import load_model
from keras import models
from keras import layers

# Load dataset of digits
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

cv2.imshow("1", train_images[1])
print(test_labels[1])

# Normalisation of these arrays of data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# Preparation of labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Create the model
network = models.Sequential()

# Add layers
network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation="softmax"))

# Compile model
network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
network.fit(train_images, train_labels, epochs=6, batch_size=128)

# Estimation of the model
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(test_acc)

# Save trained model to the file
network.save("my_model.h5")

# Load trained model from file
model = load_model("my_model.h5")


def img_proc(img_path: str):
    """Return processed image that contains digit to be ready for recognition by neural network"""

    tst = 255 - cv2.imread(img_path, 0)
    tst = cv2.resize(tst, (28, 28))
    tst = tst.reshape((1, 28 * 28))
    tst = tst.astype("float32") / 255
    return tst


# Recognition of digits from other source using model from loaded file
correct = 0

for i in range(10):
    pred = list(model.predict(img_proc(f"digits_images/{i}.jpg"))[0])
    digit = pred.index(max(pred))
    if i == digit:
        correct += 1
    print(f"{i} => {pred.index(max(pred))}")

print(f"Accuracy of recognition digits from non built-in dataset by neural network is {correct / 10 * 100}%")

# Conclusion
# Accuracy of recognition by neural network trained on built-it set of digits is 60%.
# It means that this network needs to be trained using real data to increase the accuracy of recognition.
