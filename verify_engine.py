import os
import sys
from faq_engine import FAQEngine

def run_tests():
    print("==================================================")
    print("[TEST] Running FAQ Engine NLP Verification Tests")
    print("==================================================")
    
    # Initialize engine
    engine = FAQEngine()
    
    # Load General/Overall FAQ dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_db_path = os.path.join(current_dir, "data", "general_faq.json")
    
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
            "name": "Conversational Greeting (hy / hello)",
            "query": "hy",
            "expected_match": True,
            "min_score": 0.8
        },
        {
            "name": "Technology Concept (What is AI?)",
            "query": "can you explain artificial intelligence?",
            "expected_match": True,
            "min_score": 0.4
        },
        {
            "name": "Account Security (Password Protection)",
            "query": "how to protect account password",
            "expected_match": True,
            "min_score": 0.5
        },
        {
            "name": "Customer Support (Contact Help Desk)",
            "query": "how to contact support team?",
            "expected_match": True,
            "min_score": 0.5
        },
        {
            "name": "Services & Tracking (Order Status)",
            "query": "track my package delivery",
            "expected_match": True,
            "min_score": 0.4
        },
        {
            "name": "Out of Vocabulary (Unrelated)",
            "query": "What is the capital of Mars in 3050?",
            "expected_match": False,
            "max_score": 0.25
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
