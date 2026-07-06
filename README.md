# Spam SMS Classifier Using Naive Bayes

## Project Description

This project implements a Naive Bayes classifier to automatically detect spam SMS messages. The system classifies text messages into two categories:

- **Spam**: Unwanted messages (advertisements, scams, etc.)
- **Ham**: Legitimate messages

The project is developed in a Jupyter Notebook and covers the complete machine learning pipeline from data preprocessing to model evaluation.


## Objectives

- Implement a Naive Bayes classifier from scratch
- Apply Natural Language Processing (NLP) techniques for text classification
- Understand the bag-of-words model
- Evaluate model performance using appropriate metrics
- Learn proper model evaluation with train/test split

---

## Dataset

- **File**: `spam.csv`
- **Format**: Tab-separated values
- **Size**: 5,572 SMS messages
- **Distribution**:
  - Ham: 4,825 messages (86.6%)
  - Spam: 747 messages (13.4%)
- **Columns**:
  - `class`: Message category (ham/spam)
  - `message`: Text content of the SMS

---

## Technical Implementation

### Data Preprocessing

1. **Punctuation Removal**
   - Replace punctuation marks with spaces
   - Prevents words from being concatenated

2. **Stop Words Filtering**
   - Remove common English words (e.g., "the", "and", "I")
   - These words provide little discriminative information

3. **Lowercase Conversion**
   - Standardize all text to lowercase
   - Ensures consistent word matching

### Feature Extraction

The Bag-of-Words model is used to convert text into numerical features:
Message: "Free entry in 2 a wkly comp to win FA Cup final"
↓
Vocabulary: {'free': 0, 'entry': 1, '2': 2, 'wkly': 3, ...}
↓
Vector: [1, 1, 1, 1, 0, 0, 0, ...]

text

Each message is represented as a binary vector where:
- `1` = word is present in the message
- `0` = word is absent

### Naive Bayes Model

The model uses Bayes' theorem with the naive assumption of conditional independence:
$$p(X = x | Y) = ∏(word ∈ vocabulary) p(word | Y)^{x_{word}} * (1 - p(word | Y))^{(1 - x_{word})}$$


**Key Components:**

1. **Prior Probabilities**:
   - P(Y = ham) = proportion of ham messages
   - P(Y = spam) = proportion of spam messages

2. **Conditional Probabilities**:
   - P(word | Y) = frequency of each word in messages of class Y

3. **Laplace Smoothing (alpha = 1)**:
   - Prevents zero probabilities for unseen words
   - Formula: `(count + 1) / (total_words + vocabulary_size)`

4. **Logarithmic Computation**:
   - Uses `log(P)` to avoid numerical underflow
   - Product of probabilities becomes sum of log probabilities

### Model Functions

```python
def train_model(train_data):
    """Train Naive Bayes model on training data"""
    # Build vocabulary
    # Create presence matrix
    # Calculate prior probabilities
    # Calculate conditional probabilities with Laplace smoothing
    return parameters

def classify(message, parameters):
    """Classify a new message as ham or spam"""
    # Preprocess message
    # Create feature vector
    # Compute log probabilities
    # Return predicted class
    return predicted_class
```

# Model Evaluation

### Train/Test Split

- **Training set**: 80% of data (4,457 messages)
- **Test set**: 20% of data (1,115 messages)

### Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 98.12% |
| F1-score (ham) | 93.11% |
| F1-score (spam) | 98.91% |

### Confusion Matrix
Pred Ham Pred Spam
True Ham 961 6
True Spam 15 133

### Error Analysis

- **False Positives**: 6 messages (ham predicted as spam)
- **False Negatives**: 15 messages (spam predicted as ham)

---

## How to Run

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Required Libraries

```bash
pip install numpy pandas matplotlib
```

### Installation and Execution

1. Clone or download the repository
2. Navigate to the project directory
3. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open `main.ipynb`
5. Run all cells:
   - Click `Run` → `Run All Cells`
   - Or execute cells sequentially with `Shift + Enter`
