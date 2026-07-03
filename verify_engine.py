import os
import sys
from faq_engine import FAQEngine

def run_tests():
    print("==================================================")
    print("[TEST] Running FAQ Engine NLP Verification Tests")
    print("==================================================")
    
    # Initialize engine
    engine = FAQEngine()
    
    # Load E-commerce FAQ dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_db_path = os.path.join(current_dir, "data", "ecommerce_faq.json")
    
    print(f"Loading test database: {test_db_path}")
    success, msg = engine.load_data(test_db_path)
    if not success:
        print(f"[FAIL] Failed to load test database: {msg}")
        sys.exit(1)
    print(f"[OK] {msg}\n")
    
    # Check if scikit-learn and NLTK are active
    print(f"NLP Engine Status:")
    from faq_engine import SKLEARN_AVAILABLE, NLTK_AVAILABLE
    print(f"  scikit-learn available: {SKLEARN_AVAILABLE}")
    print(f"  NLTK available        : {NLTK_AVAILABLE}")
    print(f"  Engine is_trained     : {engine.is_trained}\n")
    
    test_cases = [
        {
            "name": "Exact Match",
            "query": "What are your shipping options and delivery times?",
            "expected_match": True,
            "min_score": 0.8
        },
        {
            "name": "Rephrased Match",
            "query": "How can I track my order?",
            "expected_match": True,
            "min_score": 0.7
        },
        {
            "name": "Synonym/Keyphrase Match",
            "query": "track shipment please",
            "expected_match": True,
            "min_score": 0.3
        },
        {
            "name": "Out of Vocabulary (Unrelated)",
            "query": "What is the weather outside today?",
            "expected_match": False,
            "max_score": 0.2
        },
        {
            "name": "Near-match / Partial Query (Suggestions)",
            "query": "how to change shipping address or cancel my order",
            "expected_match": True,
            "min_score": 0.3
        }
    ]
    
    failed_tests = 0
    
    for tc in test_cases:
        print(f"Testing Case: [{tc['name']}]")
        print(f"  User Query: '{tc['query']}'")
        
        # We test with similarity threshold 0.3
        ans, score, suggestions = engine.get_response(tc['query'], threshold=0.3)
        
        print(f"  Similarity Score: {score:.4f}")
        print(f"  Response: '{ans[:80]}...'")
        
        if suggestions:
            print(f"  Suggestions returned: {len(suggestions)}")
            for s_idx, s_q in suggestions:
                print(f"    - Index {s_idx}: '{s_q}'")
                
        # Validate results
        if tc['expected_match']:
            if score < tc['min_score']:
                print(f"  [FAIL]: Expected score >= {tc['min_score']}, got {score:.4f}")
                failed_tests += 1
            else:
                print("  [PASS]")
        else:
            if score > tc['max_score']:
                print(f"  [FAIL]: Expected score <= {tc['max_score']}, got {score:.4f} (should not match)")
                failed_tests += 1
            else:
                print("  [PASS]")
        print("-" * 50)
        
    if failed_tests == 0:
        print("SUCCESS: ALL TEST CASES PASSED SUCCESSFULLY!")
        return True
    else:
        print(f"FAIL: TEST RUN COMPLETED WITH {failed_tests} FAILURE(S).")
        return False

if __name__ == "__main__":
    run_tests()
