import streamlit as st
from PIL import Image
import tensorflow as tf

# Define class names globally
class_names = ["Aston Martin","Bugatti","Ferrari","Jaguar","Koenigsegg","Lamborghini","Maserati","McLaren","Mercedes-Benz","Pagani","Porsche","Rolls-Royce"]



# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize to model input size
    img = tf.convert_to_tensor(img)  # Convert to TensorFlow tensor
    img = tf.cast(img, dtype=tf.float32)  # Convert to float32
    img = img / 255.0  # Normalize pixel values
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Function to make predictions using your loaded model
def make_prediction(model, image):
    predictions = model.predict(image)  # Pass the preprocessed image to your model
    sorted_indices = tf.argsort(predictions[0])[::-1]  # Sort indices of predictions in descending order
    predicted_class = class_names[sorted_indices[0]]
    second_highest_class = class_names[sorted_indices[1]]
    return predicted_class, second_highest_class, predictions[0], sorted_indices


def main():
    st.title("Car Brand Image Classification App")

    st.write("May not perform good for rear view images of cars due to less training data and 70% model accuracy.")
    st.write("Upload an image to classify it. The app supports the following car brands:")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        - Aston Martin
        - Bugatti
        - Ferrari
        - Jaguar
        - Koenigsegg
        - Lamborghini
        """)
    with col2:
        st.write("""
        - Maserati
        - McLaren
        - Mercedes-Benz
        - Pagani
        - Porsche
        - Rolls-Royce
        """)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            with st.spinner("Loading and processing the image...."):
                # Preprocess the image
                image = preprocess_image(uploaded_file)

                # Load your pre-trained model (replace with your model loading logic)
                model = tf.keras.models.load_model("Car_clasif_for_app_brnd.h5")  # Replace with your model path

                # Make prediction and display results
                predicted_class, second_highest_class, probabilities, sorted_indices = make_prediction(model, image)

                st.subheader(
                    f"The car shown in the image is   : {predicted_class} ({probabilities[sorted_indices[0]] * 100:.2f}%)")
                st.write(
                    f"The second most probable car is   : {second_highest_class} ({probabilities[sorted_indices[1]] * 100:.2f}%)")
                st.image(uploaded_file)



        except Exception as e:
            st.error(f"Error processing image: {e}")


main()
