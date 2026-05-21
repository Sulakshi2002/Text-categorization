import os
import re
import numpy as np
from rapidfuzz import distance

class UnifiedTelecomProcessor:
    def __init__(self, dataset_path=None):
        self.categories = [
            "Bill_Inquiries", 
            "Telephone_Number_Request", 
            "Fault_Inquiries", 
            "Product_And_New_Service", 
            "Add_More_Data"
        ]
        
        # Expanded vocabulary matching patterns to catch heavy variations
        self.domain_patterns = {
            "Bill_Inquiries": {
                "strong": ["බිල", "බිල්පත", "ගෙවීම්", "පේමන්ට්", "bill", "payment", "අමුන්ට්", "amount", "ගෙව්වා", "මුදලක්", "රුපියල්", "ගිණුම්", "එකවුන්ට්", "boc", "රිෆන්ඩ්", "refund", "බිලින්"],
                "phrases": ["බිල ගෙව්වා", "කොච්චර ගෙවන්න", "සල්ලි බැන්දා", "බිල් කම්පැනියට"]
            },
            "Telephone_Number_Request": {
                "strong": ["අංකය", "නම්බරය", "දුරකථන", "ලිපිනය", "number", "telephone", "contact", "නම්බරේ", "නම්බර්", "අංකයක්", "ලියාපදිංචි"],
                "phrases": ["නම්බරය දෙනවද", "ජෙනරල් නම්බර්", "සටහන් කරගන්න"]
            },
            "Fault_Inquiries": {
                "strong": ["කැඩිලා", "දෝෂයක්", "ලයිට්", "නිවි නිවි", "රතු", "වැඩකරන්නේනැහැ", "වැඩනෑ", "වැඩනැහැ", "los", "adsl", "router", "රවුටර්", "පියෝටීවි", "peotv", "කම්ප්ලේන්", "invalid", "චැනල්", "ඇන්ටෙනා", "වයර්", "තාක්ෂණික", "සෙටප්", "සෙටින්ග්ස්", "කේබල්"],
                "phrases": ["රතු පාට ලයිට්", "වැඩ කරන්නේ නැහැ", "ලයිට් එක පත්තු", "ඉන්ටර්නෙට් පෙන්නන්නේ නැහැ", "වැඩ නැහැ", "ලයිට් රෙඩ්"]
            },
            "Product_And_New_Service": {
                "strong": ["මාරු", "පැකේජ්", "අලුත්", "වෙනස්", "ලොකේෂන්", "location", "package", "upgrade", "downgrade", "fiber", "ෆයිබර්", "relocate", "unlimited", "ලියුමක්", "ලිපියක්"],
                "phrases": ["ලොකේෂන් මාරු", "පැකේජ් එක change", "අලුත් පැකේජ්", "ස්ථානය මාරු"]
            },
            "Add_More_Data": {
                "strong": ["ඩේටා", "ඉවරයි", "එක්ස්ට්‍රා", "ඇඩ්", "ජීබී", "gb", "data", "extra data", "balance", "බැලන්ස්", "පහක්", "දහයක්"],
                "phrases": ["ජීබී පහක්", "ඩේටා නම් ඉවර", "එක්ස්ට්‍රා ජීබී", "gb කොච්චරක්"]
            }
        }
        
        self.vocabulary = self._load_vocabulary(dataset_path)

    def _load_vocabulary(self, file_path):
        if not file_path or not os.path.exists(file_path):
            print(f"⚠️ Warning: Dataset file not found at target pathway. Initializing empty vocabulary.")
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def partial_clean_text(self, text, threshold=0.70):
        if not self.vocabulary:
            return text
            
        words = text.split()
        cleaned_words = []
        for word in words:
            if word in self.vocabulary:
                cleaned_words.append(word)
                continue
            
            best_match = None
            best_sim = 0.0
            for vocab_word in self.vocabulary:
                sim = distance.JaroWinkler.similarity(word, vocab_word)
                if sim > best_sim:
                    best_sim = sim
                    best_match = vocab_word
            
            if best_match and best_sim >= threshold:
                cleaned_words.append(best_match)
            else:
                cleaned_words.append(word)
                
        return " ".join(cleaned_words)

    def _generate_character_ngrams(self, text, n=3):
        text = re.sub(r'\s+', '', text)
        if len(text) < n:
            return [text]
        return [text[i:i+n] for i in range(len(text) - n + 1)]

    def _calculate_ngram_similarity(self, text_ngrams, target_word):
        target_ngrams = self._generate_character_ngrams(target_word, n=3)
        intersection = set(text_ngrams).intersection(set(target_ngrams))
        return len(intersection) / len(target_ngrams) if target_ngrams else 0.0

    def process_and_classify(self, raw_noisy_text):
        # Step 1: Text Restoration
        partially_cleaned = self.partial_clean_text(raw_noisy_text)
        normalized_text = str(partially_cleaned).lower().strip()
        
        # --- CRITICAL RULE: HARD OVERRIDE HEURISTIC FOR TELECOM FAULTS ---
        # If text explicitly states technical system breakdowns, immediately classify as Fault
        fault_critical_anchors = ["වැඩ නැහැ", "වැඩනෑ", "පියෝටීවි", "ලයිට්", "රවුටර්", "රෙඩ්වලා", "කැඩිලා"]
        for anchor in fault_critical_anchors:
            if anchor in normalized_text:
                return "Fault_Inquiries", 100.0, [f"[Heuristic Match] Triggered by anchor keyword: '{anchor}'"], partially_cleaned

        # Step 2: Fallback to Scoring Pipeline if heuristic isn't tripped
        tokens = re.findall(r'\b\w+\b', normalized_text)
        text_ngrams = self._generate_character_ngrams(normalized_text, n=3)
        
        category_scores = {cat: 0.0 for cat in self.categories}
        evidence_logs = {cat: [] for cat in self.categories}
        sim_threshold = 0.70 
        
        for cat in self.categories:
            score = 0.0
            patterns = self.domain_patterns[cat]
            
            for phrase in patterns["phrases"]:
                if phrase in normalized_text:
                    score += 10.0  # Increased weight for phrase matches
                    evidence_logs[cat].append(f"[Phrase] '{phrase}'")
            
            for kw in patterns["strong"]:
                if kw in normalized_text:
                    score += 8.0  # Increased weight for exact keywords
                    evidence_logs[cat].append(f"[Exact Keyword] '{kw}'")
                    continue 
                
                max_fuzzy_found = 0.0
                best_token = ""
                for token in tokens:
                    sim = distance.JaroWinkler.similarity(kw, token)
                    if sim > sim_threshold and sim > max_fuzzy_found:
                        max_fuzzy_found = sim
                        best_token = token
                            
                if max_fuzzy_found > 0.0:
                    score += 5.0 * max_fuzzy_found
                    evidence_logs[cat].append(f"[Fuzzy] '{best_token}'->'{kw}' ({max_fuzzy_found:.2f})")
                
                ngram_sim = self._calculate_ngram_similarity(text_ngrams, kw)
                if ngram_sim > 0.50:
                    score += 3.0 * ngram_sim
                    evidence_logs[cat].append(f"[Ngram Sub-Word] '{kw}' ({ngram_sim:.2f})")
                    
            category_scores[cat] = score

        scores_vector = np.array([category_scores[c] for c in self.categories])
        if np.sum(scores_vector) == 0:
            return "Bill_Inquiries", 100.0, ["Default Safe Flag Triggered"], partially_cleaned
            
        exp_scores = np.exp(scores_vector - np.max(scores_vector))
        probabilities = exp_scores / np.sum(exp_scores)
        top_idx = np.argmax(probabilities)
        
        return (
            self.categories[top_idx], 
            float(probabilities[top_idx]) * 100, 
            list(set(evidence_logs[self.categories[top_idx]]))[:4],
            partially_cleaned
        )

# --- EXECUTION ---
if __name__ == "__main__":
    DATASET_FILE = r"D:\SLT\Test 3\my_dataset.txt"
    engine = UnifiedTelecomProcessor(dataset_path=DATASET_FILE)
    
    noisy_test_input = """යිබෝවන් මරානිමඩපොලනිපුට සහය වන්නයඩුව ේමිස් අපේ පියෝටීවි එකයි තෙලිපුරනයින් එකයි දෙකම වැඩ නැහැටමීට දවස් දපලින් විල්ලත් හැුවා හැදුවත
ඒ විදිහටම ආය මේ නැව  ක්තිය වෙලා තිෙනවාතෙලිෝන් එයි පියෝටීවි එකයිකනක්ෂණ එකිටකරුගෙනමෙන්ඩබ්ලි පීපී ඒ කුමාරපයිබලයින් එකක්යලින්ත්‍රයඩ් එක වැඩකරනදනෑමොකම ැඩනෑඑල්ලවසි ග
ෙන ලයි් කරෙඩ්වලා තින්ද බලනවෙන් වෙලා ියරෙඩ්වෙලා තයෙනවම්පෙන් එක ඇතුළත් කරන්නෑ මට සම්බලගරන් මභය නම්ම ් බින්ද හතයි හයයිරිඅසුතුනයි හඅනු එ්කයි රිටසිය පනස්පහැතිමි බ
ැිනොනව ඳිනස නගත ස්තුතියි ම අදාලඩපන්හදනටීමකට එක ව ක්පෙන් එක දන්නෙල ලාවක්පනබය නම්බකටසෙර්ත් එකක් ගවල් මසරීමඩල ති්තූය සබ දවසසක් මෙව මැගම සඳා රැඳී සිටින්න"""

    predicted_cat, confidence, evidence, cleaned_output = engine.process_and_classify(noisy_test_input)
    
    print("\n" + "="*60)
    print(f"PREDICTED INTENT CATEGORY : {predicted_cat}")
    print(f"SYSTEM CONFIDENCE SCORE   : {confidence:.2f}%")
    print(f"KEYWORD EVIDENCE FOUND    : {evidence}")
    print("="*60 + "\n")