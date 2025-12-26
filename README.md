
# Emotion Detection from Facial Images

This project implements an emotion detection system using classical machine learning techniques, specifically Principal Component Analysis (PCA) for dimensionality reduction and a Support Vector Machine (SVM) classifier.

## Project Pipeline:

1.  **Data Loading and Preprocessing**: Images from the dataset are loaded, converted to grayscale, and resized to a uniform size (128x128 pixels).
2.  **Face Detection**: A pre-trained Haar Cascade classifier is used to detect faces within each preprocessed image. The largest detected face is then cropped and further resized to the target dimensions.
3.  **Feature Extraction (Flattening)**: The 2D facial images are flattened into 1D vectors to prepare them for PCA.
4.  **Data Splitting**: The flattened data and corresponding emotional labels are split into training and testing sets.
5.  **Dimensionality Reduction (PCA)**: Principal Component Analysis is applied to the training data to reduce its dimensionality while retaining 95% of the variance. The optimal number of components identified was `110`.
6.  **Model Training (SVM)**: A Support Vector Machine (SVM) classifier is trained on the PCA-transformed training data.
7.  **Model Evaluation**: The trained SVM model's performance is evaluated on the PCA-transformed test data using metrics such as accuracy, precision, recall, and F1-score.
8.  **Practical Verification**: The model can predict emotions from custom uploaded images.

## Key Findings:

*   The SVM model achieved an overall accuracy of 0.4141 on the PCA-transformed test data.
*   Performance varied by emotion: 'Happy' and 'Surprise' classes showed better performance (F1-scores around 0.67 and 0.77 respectively), while 'Angry', 'Fear', and 'Sad' had lower F1-scores (around 0.24-0.28).
*   110 PCA components were sufficient to explain 95% of the variance, reducing the feature space significantly from 16384 to 110 dimensions.

## How to Run:

1.  **Open in Google Colab**: Upload the `.ipynb` file to Google Colab.
2.  **Dataset**: Ensure your dataset (images organized by emotion folders) is accessible, for example, in `/content/dataset`.
3.  **Run All Cells**: Execute all cells in the notebook sequentially.
4.  **Upload Custom Image**: Follow the prompts to upload a custom image for real-time emotion prediction.

## Dependencies:

This project requires the following Python libraries:
*   `numpy`
*   `opencv-python` (cv2)
*   `scikit-learn`
*   `matplotlib`
*   `tqdm` (if used for progress bars in data loading)

These can be installed using `pip`:
`pip install -q numpy opencv-python scikit-learn matplotlib tqdm`
