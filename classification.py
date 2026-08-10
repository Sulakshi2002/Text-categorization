import json
import os
import re

class SinhalaTelecomClassifier:
    CATEGORY_FILES = {
        "Bill_Inquiries": "bill_inquiries.json",
        "Fault_and_Technical": "fault_and_technical.json",
        "Product_And_New_Service": "product_and_new_service.json",
        "Telephone_Number_Request_Or_Other": "telephone_number_request_or_other.json"
    }

    def __init__(self):
        self.categories = list(self.CATEGORY_FILES.keys())
        self.domain_patterns = self._load_category_patterns()

    def _load_category_patterns(self):
        base_dir = os.path.join(os.path.dirname(__file__), "categories")
        patterns = {}

        for category, filename in self.CATEGORY_FILES.items():
            file_path = os.path.join(base_dir, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                patterns[category] = json.load(file)

        return patterns

    def _levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _calculate_similarity(self, word1, word2):
        max_len = max(len(word1), len(word2))
        if max_len == 0:
            return 0.0
        dist = self._levenshtein_distance(word1, word2)
        return (max_len - dist) / max_len

    def predict(self, raw_noisy_text, match_threshold=0.80):
        normalized_text = str(raw_noisy_text).lower().strip()
        tokens = re.findall(r'[a-zA-Z0-9]+|[\u0D80-\u0DFF]+', normalized_text)
        
        category_scores = {cat: 0.0 for cat in self.categories}
        evidence_logs = []
        seen_tokens_in_category = set()

        for token in tokens:
            if len(token) < 2:
                continue
                
            for cat in self.categories:
                for kw, intent_weight in self.domain_patterns[cat].items():
                    sim = self._calculate_similarity(token, kw)
                    
                    if sim >= match_threshold:
                        match_key = f"{cat}_{kw}"
                        base_weight = 5.0 if sim == 1.0 else (3.0 * sim)
                        calculated_score = base_weight * intent_weight
                        
                        if match_key in seen_tokens_in_category:
                            calculated_score *= 0.25 
                        else:
                            seen_tokens_in_category.add(match_key)
                            
                        category_scores[cat] += calculated_score
                        evidence_logs.append(f"  - Token '{token}' matched with '{kw}' in {cat} (Sim: {sim:.2f}, Final Weight: {calculated_score:.2f})")
                        break 

        max_score = max(category_scores.values())
        if max_score == 0:
            predicted_category = "UNKNOWN_DIRECT_INQUIRY"
        else:
            predicted_category = [cat for cat, score in category_scores.items() if score == max_score][0].upper()

        return {
            "predicted_category": predicted_category,
            "raw_scoring_matrix": {c: round(v, 2) for c, v in category_scores.items()},
            "evidence": evidence_logs
        }

if __name__ == "__main__":
    classifier = SinhalaTelecomClassifier()
    
    print("=" * 70)
    print("       SINHALA TELECOM CLASSIFIER TERMINAL SYSTEM (2026)")
    print("   Type 'exit' or 'quit' to terminate the processing loop.")
    print("=" * 70)
    
    while True:
        print("\n" + "_" * 70)
        user_input = input("Enter/Paste Sinhala Transcript Log:\n> ")
        
        # Check termination condition safely
        if user_input.strip().lower() in ['exit', 'quit']:
            print("\nExiting Classifier Session. Goodbye!")
            break
            
        if not user_input.strip():
            print("[Warning] Empty input received. Please enter text.")
            continue
            
        # Execute prediction pipeline
        result = classifier.predict(user_input)
        
        # Display clean structured results down to terminal interface
        print(f"\nPrediction Outcome   : {result['predicted_category']}")
        print(f"Scoring Weights Matrix: {result['raw_scoring_matrix']}")
        print("Matched Evidence Logs :")
        if not result['evidence']:
            print("  No meaningful token alignments mapped above 0.80 threshold.")
        else:
            for log in result['evidence'][:15]:  # Displaying top 15 matches for readability
                print(log)
            if len(result['evidence']) > 15:
                print(f"  ... and {len(result['evidence']) - 15} more logs matched.")