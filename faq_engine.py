import json
import csv
import os
import re

# Try to import scikit-learn and NLTK
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

class FAQEngine:
    def __init__(self, filepath=None):
        self.faqs = []  # List of dicts: [{'question': '...', 'answer': '...'}]
        self.vectorizer = None
        self.tfidf_matrix = None
        self.is_trained = False
        
        # Simple stop words list as fallback if NLTK is not initialized
        self.fallback_stopwords = {
            "is", "the", "a", "an", "at", "on", "for", "to", "of", "in", "and", "or",
            "what", "how", "why", "where", "who", "which", "are", "do", "does", "did",
            "can", "could", "would", "should", "will", "your", "my", "me", "you",
            "he", "she", "it", "they", "we", "us", "our", "about", "with", "from",
            "by", "that", "this", "these", "those", "have", "has", "had", "been", "was", "were"
        }
        
        # Initialize Lemmatizer if NLTK is available
        self.lemmatizer = None
        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
            except Exception:
                self.lemmatizer = None

        if filepath:
            self.load_data(filepath)

    def preprocess(self, text):
        """
        Cleans and tokenizes the input text. Uses NLTK if available, 
        otherwise falls back to regex-based cleaning.
        """
        if not text:
            return ""
            
        # Lowercase
        text = text.lower().strip()
        
        # Check NLTK availability
        if NLTK_AVAILABLE and self.lemmatizer:
            try:
                # Tokenize
                words = word_tokenize(text)
                # Remove punctuation
                words = [re.sub(r'[^\w\s]', '', w) for w in words]
                # Filter empty tokens and stop words
                try:
                    stop_words = set(stopwords.words('english'))
                except Exception:
                    stop_words = self.fallback_stopwords
                
                cleaned_words = []
                for w in words:
                    if w and w not in stop_words:
                        # Lemmatize
                        cleaned_words.append(self.lemmatizer.lemmatize(w))
                return " ".join(cleaned_words)
            except Exception:
                # If NLTK fails for any reason, fall back to basic preprocessing
                pass

        # Fallback preprocessing (no NLTK)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        # Filter stop words
        cleaned_words = [w for w in words if w not in self.fallback_stopwords]
        return " ".join(cleaned_words)

    def load_data(self, filepath):
        """
        Loads FAQs from a JSON or CSV file.
        Returns (success_boolean, message_string).
        """
        if not os.path.exists(filepath):
            return False, f"File path '{filepath}' does not exist."

        try:
            _, ext = os.path.splitext(filepath)
            ext = ext.lower()
            
            temp_faqs = []
            
            if ext == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        return False, "Invalid JSON structure: Must be a list of FAQ objects."
                    for i, item in enumerate(data):
                        if 'question' not in item or 'answer' not in item:
                            return False, f"Missing 'question' or 'answer' field at index {i}."
                        temp_faqs.append({
                            'question': item['question'].strip(),
                            'answer': item['answer'].strip()
                        })
                        
            elif ext == '.csv':
                with open(filepath, 'r', encoding='utf-8-sig') as f:
                    # Sniff format
                    try:
                        dialect = csv.Sniffer().sniff(f.read(1024))
                        f.seek(0)
                        reader = csv.DictReader(f, dialect=dialect)
                    except Exception:
                        f.seek(0)
                        reader = csv.DictReader(f)
                        
                    # Verify headers
                    headers = [h.strip().lower() for h in reader.fieldnames] if reader.fieldnames else []
                    q_col, a_col = None, None
                    for h in reader.fieldnames:
                        hl = h.strip().lower()
                        if 'question' in hl or hl == 'q':
                            q_col = h
                        if 'answer' in hl or hl == 'a':
                            a_col = h
                            
                    if not q_col or not a_col:
                        return False, "CSV must contain columns representing 'question' and 'answer'."
                        
                    for i, row in enumerate(reader):
                        q = row.get(q_col, '').strip()
                        a = row.get(a_col, '').strip()
                        if q and a:
                            temp_faqs.append({'question': q, 'answer': a})
                            
            else:
                return False, f"Unsupported file extension '{ext}'. Only JSON and CSV are supported."

            if not temp_faqs:
                return False, "The file contains no valid FAQ records."

            self.faqs = temp_faqs
            self.train()
            return True, f"Successfully loaded {len(self.faqs)} FAQ records."

        except Exception as e:
            return False, f"Failed to read file: {str(e)}"

    def save_data(self, filepath):
        """
        Saves current FAQ database as a JSON file.
        Returns (success_boolean, message_string).
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.faqs, f, indent=2, ensure_ascii=False)
            return True, f"Successfully saved {len(self.faqs)} FAQs to {filepath}."
        except Exception as e:
            return False, f"Failed to save data: {str(e)}"

    def train(self):
        """
        Trains/fits the TF-IDF Vectorizer on the current FAQ questions list.
        """
        if not SKLEARN_AVAILABLE or not self.faqs:
            self.vectorizer = None
            self.tfidf_matrix = None
            self.is_trained = False
            return

        try:
            # Collect and preprocess questions
            preprocessed_questions = [self.preprocess(faq['question']) for faq in self.faqs]
            
            # Re-initialize vectorizer to ensure clean vocabulary
            self.vectorizer = TfidfVectorizer(
                token_pattern=r'(?u)\b\w+\b'  # Capture single-character words too
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(preprocessed_questions)
            self.is_trained = True
        except Exception as e:
            print(f"Error during FAQ model training: {e}")
            self.is_trained = False

    def add_faq(self, question, answer):
        """Adds a new FAQ and retrains the model."""
        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            return False, "Question and Answer cannot be empty."
        
        # Check if question already exists (ignore case and punctuation)
        clean_q = self.preprocess(question)
        for faq in self.faqs:
            if self.preprocess(faq['question']) == clean_q:
                return False, "A similar question already exists in the database."

        self.faqs.append({'question': question, 'answer': answer})
        self.train()
        return True, "FAQ added successfully."

    def edit_faq(self, index, question, answer):
        """Edits an existing FAQ and retrains the model."""
        if index < 0 or index >= len(self.faqs):
            return False, "Invalid FAQ index."
        
        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            return False, "Question and Answer cannot be empty."
            
        self.faqs[index] = {'question': question, 'answer': answer}
        self.train()
        return True, "FAQ updated successfully."

    def delete_faq(self, index):
        """Deletes an FAQ at the given index and retrains the model."""
        if index < 0 or index >= len(self.faqs):
            return False, "Invalid FAQ index."
            
        deleted_q = self.faqs.pop(index)
        self.train()
        return True, f"Deleted FAQ: '{deleted_q['question'][:30]}...'"

    def get_response(self, user_query, threshold=0.3):
        """
        Finds the closest FAQ matching the user query.
        Returns:
            answer (str): The response answer (or fallback text).
            score (float): Cosine similarity score of the best match.
            suggestions (list): List of (index, question) tuples for close alternatives.
        """
        if not self.faqs:
            return "No FAQs are currently loaded in the database. Please add or import some questions.", 0.0, []

        # If scikit-learn is not available, do basic word overlap matching
        if not SKLEARN_AVAILABLE or not self.is_trained or self.tfidf_matrix is None:
            return self._fallback_match(user_query, threshold)

        try:
            # Preprocess user query
            cleaned_query = self.preprocess(user_query)
            if not cleaned_query:
                return "I'm sorry, I couldn't understand your query. Could you please rephrase it?", 0.0, []

            # Vectorize query
            query_vector = self.vectorizer.transform([cleaned_query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get index of best match
            best_idx = similarities.argmax()
            best_score = similarities[best_idx]
            
            # If best score is above threshold, return the answer
            if best_score >= threshold:
                return self.faqs[best_idx]['answer'], float(best_score), []
            
            # If best score is below threshold, find up to 3 closest questions that have similarity > 0
            # Get indices sorted by similarity descending
            sorted_indices = similarities.argsort()[::-1]
            suggestions = []
            for idx in sorted_indices:
                score = similarities[idx]
                if score > 0.05 and idx != best_idx and len(suggestions) < 3:
                    suggestions.append((int(idx), self.faqs[idx]['question']))
            
            # Add the best match itself to suggestions if its score is between 0.05 and threshold
            if best_score > 0.05:
                # Insert at the beginning of suggestions
                suggestions.insert(0, (int(best_idx), self.faqs[best_idx]['question']))
                
            # Keep unique and limit to top 3
            seen = set()
            unique_suggestions = []
            for idx, q in suggestions:
                if q not in seen:
                    seen.add(q)
                    unique_suggestions.append((idx, q))
            unique_suggestions = unique_suggestions[:3]

            fallback_msg = "I'm sorry, I couldn't find a direct match in the FAQs. "
            if unique_suggestions:
                fallback_msg += "Did you mean one of the questions listed below?"
            else:
                fallback_msg += "Could you please try rephrasing your question, or check out the FAQ list?"

            return fallback_msg, float(best_score), unique_suggestions

        except Exception as e:
            return f"An error occurred during matching: {str(e)}", 0.0, []

    def _fallback_match(self, user_query, threshold=0.3):
        """
        Simple keyword overlap similarity algorithm in case scikit-learn is not available.
        Calculates Jaccard-like keyword overlap coefficients.
        """
        q_words = set(self.preprocess(user_query).split())
        if not q_words:
            return "I'm sorry, I couldn't understand your query. Could you please rephrase it?", 0.0, []

        scores = []
        for idx, faq in enumerate(self.faqs):
            faq_words = set(self.preprocess(faq['question']).split())
            if not faq_words:
                scores.append(0.0)
                continue
            intersection = q_words.intersection(faq_words)
            union = q_words.union(faq_words)
            score = len(intersection) / len(union) if union else 0.0
            scores.append(score)

        if not scores:
            return "Could you please try rephrasing your question?", 0.0, []

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_idx]

        if best_score >= threshold:
            return self.faqs[best_idx]['answer'], best_score, []

        # Find suggestions
        suggestions = []
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for idx in sorted_indices:
            score = scores[idx]
            if score > 0.02 and len(suggestions) < 3:
                suggestions.append((idx, self.faqs[idx]['question']))

        fallback_msg = "I'm sorry, I couldn't find a matching FAQ. "
        if suggestions:
            fallback_msg += "Did you mean one of the questions below?"
        return fallback_msg, best_score, suggestions
