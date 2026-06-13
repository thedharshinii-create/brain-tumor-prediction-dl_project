from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

try:
    from keras.models import load_model as keras_load_model
except ModuleNotFoundError:
    try:
        from tensorflow.keras.models import load_model as keras_load_model
    except ModuleNotFoundError:
        keras_load_model = None


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "brain_tumor_dataset"
MODEL_DIR = PROJECT_DIR
DEFAULT_MODEL_FILENAME = "brain_tumor_model-2.keras"
IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_NAMES = ["No Tumour", "Brain Tumour"]
CLASS_NAME_MAP = {
    "no": "No Tumour",
    "yes": "Brain Tumour",
}


def get_model_path(filename: Optional[str] = None) -> Path:
    """Return a local Keras model path from the project folder."""
    if filename:
        clean_filename = filename.strip().strip("\"'")
        model_path = Path(clean_filename)
        if not model_path.is_absolute():
            model_path = MODEL_DIR / clean_filename

        if model_path.suffix.lower() not in [".h5", ".keras"]:
            raise ValueError(
                "Enter a real model file ending with .h5 or .keras, for example model.h5. "
                "Folder names like .virtual_documents are not model files."
            )
    else:
        model_files = sorted(MODEL_DIR.glob("*.h5")) + sorted(MODEL_DIR.glob("*.keras"))
        if not model_files:
            raise FileNotFoundError(
                f"No .h5 or .keras model file found in: {MODEL_DIR}. "
                "Place your trained model in this project folder, for example: "
                "brain_tumor_model.h5"
            )
        model_path = model_files[0]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return model_path


def discover_class_names(dataset_path: Path) -> list[str]:
    """Use dataset folder names as class labels when available."""
    if not dataset_path.exists():
        return []

    class_dirs = [
        CLASS_NAME_MAP.get(path.name.lower(), path.name.replace("_", " ").title())
        for path in sorted(dataset_path.iterdir())
        if path.is_dir()
    ]
    return class_dirs


@st.cache_resource
def load_model(model_filename: Optional[str]):
    if keras_load_model is None:
        raise RuntimeError(
            "Keras/TensorFlow is not installed. Install the project requirements with: "
           "pip install -r requirements.txt"
        )

    model_path = get_model_path(model_filename)
    model = keras_load_model(str(model_path), compile=False)
    return model, model_path


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_with_model(model, image: Image.Image):
    """Run prediction with a CNN deep learning model."""
    prediction_input = preprocess_image(image)
    prediction = model.predict(prediction_input, verbose=0)

    prediction = np.asarray(prediction).squeeze()

    if prediction.size == 1:
        # Binary classification with a single sigmoid output.
        score = float(prediction)
        class_index = int(score >= 0.5)
        confidence = score if class_index == 1 else 1 - score
    else:
        # Multi-class softmax output.
        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

    return class_index, confidence


def format_prediction(prediction_label, class_names: list[str]) -> str:
    if isinstance(prediction_label, str):
        return prediction_label.replace("_", " ").title()

    class_index = int(prediction_label)
    if class_names and 0 <= class_index < len(class_names):
        return class_names[class_index]
    return f"Class {class_index}"


def main() -> None:
    st.set_page_config(page_title="Brain Tumour Detection", page_icon=":brain:")
    st.title("Brain Tumour Detection")

    st.sidebar.header("CNN Deep Learning Model")
    model_filename = st.sidebar.text_input(
        "CNN Model filename",
        value=DEFAULT_MODEL_FILENAME,
        placeholder="Example: model.h5",
        help=(
            "Enter your trained CNN model file (.h5 or .keras). "
            "Leave empty to automatically use the first .h5/.keras file in this project folder. "
            "This should be a Keras/TensorFlow deep learning model with CNN layers."
        ),
    )
    model_filename = model_filename.strip() or None

    class_names = discover_class_names(DATASET_PATH)
    if not class_names:
        class_names = DEFAULT_CLASS_NAMES

    if DATASET_PATH.exists():
        st.sidebar.caption("Classes found from dataset folders:")
        st.sidebar.write(", ".join(class_names))
    else:
        st.sidebar.info("Using default classes because the local dataset folder was not found.")

    uploaded_file = st.file_uploader(
        "Upload an MRI image",
        type=sorted(extension.lstrip(".") for extension in IMAGE_EXTENSIONS),
    )

    if uploaded_file is None:
        st.info("Upload a brain MRI image to get a prediction.")
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width="stretch")

    if st.button("Predict", type="primary"):
        try:
            model, model_path = load_model(model_filename)
            class_index, confidence = predict_with_model(model, image)
            prediction_label = format_prediction(class_index, class_names)

            st.success(f"Prediction: {prediction_label}")
            st.metric("Confidence", f"{confidence * 100:.2f}%")
            st.caption(f"Deep Learning CNN Model used: {model_path}")
        except Exception as error:
            st.error(f"CNN Prediction failed: {error}")
            return


if __name__ == "__main__":
    main()
