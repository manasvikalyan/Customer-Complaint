## Customer Complaint NLP Analysis

This project analyzes customer complaint text data using a complete NLP pipeline implemented in `customer-complaint.ipynb`.

### Notebook workflow (high-level)

1. **Data loading**: Load the raw complaints dataset into a pandas DataFrame for further processing.
2. **Text preprocessing**: Clean and normalize the complaint text (e.g., lowercasing, removing punctuation, and stopword removal).
3. **Exploratory data analysis (EDA)**: Explore the distribution of complaints across products/issues and basic text characteristics (lengths, frequencies, etc.).
4. **Feature extraction**: Transform cleaned text into numeric features using **TF-IDF** (via `TfidfVectorizer`) and related representations.
5. **Topic modelling**: Apply **Non-negative Matrix Factorization (NMF)** on the TF-IDF document-term matrix to uncover latent complaint topics.
6. **Model building using supervised learning**: Build a **Logistic Regression** classifier to predict target labels from the engineered features.
7. **Model training and evaluation**: Train the supervised model and evaluate performance using standard metrics to assess predictive quality.
8. **Model inference**: Use the trained model to generate predictions on new or unseen complaint examples.

