# AI FAQ Chatbot - Professional Desktop Application

A modern, professional-grade desktop FAQ Chatbot application built in Python using **CustomTkinter** for a premium dark/light mode user interface, and **scikit-learn** + **NLTK** for natural language processing (NLP), text cleaning, TF-IDF vectorization, and Cosine Similarity matching.

This project is designed as an interactive desktop suite containing a chat assistant, a real-time FAQ database manager with CRUD controls, and diagnostic configurations.

---

## Key Features

1. **🤖 Interactive Chatbot Assistant**
   - Styled message bubbles for users and the chatbot avatar.
   - Intelligent search using TF-IDF and Cosine Similarity.
   - **Alternative Suggestions**: Below-threshold queries trigger smart "Did you mean?" suggestions.
   - Bottom quick-ask "suggested questions" chips for one-click queries.
   - Real-time confidence matching percentage indicators.

2. **📚 FAQ Database Management**
   - Split-panel view: filter/search all FAQs instantly on the left, edit details on the right.
   - **Full CRUD operations**: Add new FAQs, update existing ones, or delete records.
   - Retrains the vectorizer model instantly in-memory upon saving changes.

3. **⚙️ Settings & Environment Diagnostics**
   - Slider to dynamically adjust the similarity match threshold (0.0 to 1.0).
   - Toggles to load different built-in FAQ datasets (e.g., E-commerce support, University admissions).
   - Import/Export tools to load or save datasets in JSON or CSV format.
   - Diagnostic status displaying file paths, total entries, and active system library versions.

4. **🎨 Premium Modern Design**
   - Built on CustomTkinter supporting responsive resizing, rounded frames, and clean typography.
   - Theme configurations: Light Mode, Dark Mode, or System theme synchronization.

---

## Technical Stack & NLP Pipeline

- **GUI Toolkit**: `CustomTkinter` (v5.2.2+)
- **NLP Text Cleaning**: `NLTK` (Word tokenization, lowercase normalization, punctuation removal, stop-words filtering, and `WordNetLemmatizer` word root extraction).
- **Matching Pipeline**: `scikit-learn` `TfidfVectorizer` (converts sentences to numerical features) and `cosine_similarity` metrics:
  $$\text{Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

---

## Repository Structure

```text
├── data/
│   ├── ecommerce_faq.json     # Default e-commerce questions database
│   └── university_faq.json    # Default university admissions database
├── faq_engine.py              # NLP matching pipeline & data CRUD logic
├── gui.py                     # CustomTkinter GUI layout & frontend components
├── main.py                    # Startup bootstrapper & NLTK downloader
├── verify_engine.py           # Automated test suite for matching logic
└── .gitignore                 # Standard Python gitignore rules
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
   cd YOUR_REPOSITORY_NAME
   ```

2. **Install dependencies:**
   ```bash
   pip install customtkinter scikit-learn nltk scipy numpy
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```
   *Note: On the first launch, the bootstrapper will programmatically download the required NLTK corpora (`punkt`, `wordnet`, `stopwords`, etc.) silently.*

---

## Running Verification Tests

To verify that the text pre-processing and TF-IDF matching engine works correctly on your system, execute the automated test script:

```bash
python verify_engine.py
```
