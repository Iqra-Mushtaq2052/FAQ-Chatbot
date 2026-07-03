import os
import sys

# Print startup information
print("==================================================")
print("[BOOT] FAQ Chatbot GUI Application Bootstrapper")
print("==================================================")

# Try downloading NLTK dependencies programmatically
try:
    print("Checking and downloading NLP dependencies...")
    import nltk
    
    # Download required datasets silently
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # For NLTK 3.9+ tokenization
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("[OK] NLP dependencies verified and loaded.")
except Exception as e:
    print(f"[WARN] Warning setting up NLTK resources: {e}")
    print("  Application will use local fallback NLP engines if necessary.")

# Make sure scikit-learn is present
try:
    import sklearn
    print("[OK] scikit-learn verified.")
except ImportError:
    print("[WARN] Warning: scikit-learn is not installed.")
    print("  Application will fall back to Jaccard word-overlap matching.")

# Import application components
try:
    from faq_engine import FAQEngine
    from gui import FAQChatbotGUI
except ImportError as e:
    print(f"[ERROR] Critical import error: {e}")
    sys.exit(1)

def main():
    # Setup data paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    default_db_path = os.path.join(data_dir, "ecommerce_faq.json")
    
    # Verify default dataset file exists
    if not os.path.exists(default_db_path):
        print(f"Creating default FAQ file at: {default_db_path}")
        # A simple fallback dataset to write if not found
        default_faqs = [
            {
                "question": "What are your shipping options and delivery times?",
                "answer": "We offer standard shipping (3-5 business days) and express shipping (1-2 business days)."
            },
            {
                "question": "What is your return policy?",
                "answer": "You can return any unused item within 30 days of purchase for a full refund."
            }
        ]
        import json
        try:
            with open(default_db_path, 'w', encoding='utf-8') as f:
                json.dump(default_faqs, f, indent=2)
        except Exception as e:
            print(f"Failed to create default data: {e}")
            
    # Start Engine
    print("Initializing FAQ NLP Engine...")
    engine = FAQEngine()
    
    # Start GUI
    print("Starting CustomTkinter Graphical User Interface...")
    app = FAQChatbotGUI(engine, default_db_path)
    print("GUI running. Displaying application window.")
    app.mainloop()

if __name__ == "__main__":
    main()
