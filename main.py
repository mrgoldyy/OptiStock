import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# Memuat model
model = load_model('model.h5')
def y_pred (image):
    img = np.array(image.resize((32, 32))) / 255.0
    img = np.expand_dims(img, axis=0)
    # Lakukan prediksi di sini
    y_pred = model.predict(img)
    return y_pred

# Main program
def main():
    st.title('Klasifikasi Kanker Kulit')

    # Upload gambar
    uploaded_image = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

    # Jika gambar sudah diunggah
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Gambar yang diunggah', use_column_width=True)

        # Tambahkan tombol 'Prediksi'
        if st.button('Prediksi'):
            prediction = y_pred(image)
            st.write('Prediksi:', prediction)

if __name__ == '__main__':
    main()