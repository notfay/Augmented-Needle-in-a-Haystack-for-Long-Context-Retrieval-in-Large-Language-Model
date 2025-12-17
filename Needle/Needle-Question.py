# Needle-Question.py - Calculate Needle-Question Similarity
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from abc import ABC, abstractmethod
import warnings
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sentence_transformers")

# --- 1. Define Abstract Model Class ---

class EmbeddingModel(ABC):
    """Abstract base class for all embedding models."""
    def __init__(self, model_name):
        self.model_name = model_name
        print(f"Initializing model: {self.model_name}")

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embeds a list of texts and returns a numpy array."""
        pass

# --- 2. Create Concrete Model Classes ---

class STModel(EmbeddingModel):
    """Wrapper for SentenceTransformers."""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__(model_name)
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name)
        except ImportError:
            print("Please install sentence-transformers: pip install sentence-transformers")
            raise

    def embed(self, texts: list[str]) -> np.ndarray:
        # Sentence Transformers typically normalize by default
        return self.encoder.encode(texts, show_progress_bar=False)

# --- 3. Haystack Checker Class (from Needle_Novelty.py) ---

class HaystackDuplicateChecker:
    """
    Check if needle sentences already exist in a haystack document
    using semantic similarity (cosine similarity of embeddings).
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 256,
                 chunk_overlap: int = 128,
                 similarity_threshold: float = 0.85,
                 haystack_max_sim: float = 0.5):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.haystack_max_sim = haystack_max_sim
        
        # We will use the model from the main script to avoid reloading if possible,
        # but for independence, we'll re-init if needed or use the one passed.
        # For this script, we'll just let it load its own or use the STModel logic.
        # To save memory, we will use the STModel instance passed to check_needle if we refactor,
        # but to keep it simple and matching the user request, we'll just instantiate.
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            pass
        
        self.haystack_chunks = None
        self.haystack_embeddings = None
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def _chunk_text(self, text: str) -> list[str]:
        words = self._clean_text(text).split()
        word_chunk_size = int(self.chunk_size / 5) 
        word_overlap = int(self.chunk_overlap / 5)
        chunks = []
        position = 0
        while position < len(words):
            chunk_words = words[position : position + word_chunk_size]
            chunk_text = ' '.join(chunk_words)
            if chunk_text:
                chunks.append(chunk_text)
            position += (word_chunk_size - word_overlap)
        return chunks
    
    def load_haystack(self, haystack_file: str):
        print(f"Loading haystack from file: {haystack_file}")
        with open(haystack_file, 'r', encoding='utf-8') as f:
            text = f.read()
        self.haystack_chunks = self._chunk_text(text)
        print(f"Created {len(self.haystack_chunks)} chunks")
        self.haystack_embeddings = self.model.encode(
            self.haystack_chunks,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        self.haystack_embeddings = normalize(self.haystack_embeddings)

    def get_max_similarity(self, text: str) -> float:
        if self.haystack_embeddings is None:
            return 0.0
        emb = self.model.encode([text], convert_to_numpy=True)
        emb = normalize(emb)
        sims = cosine_similarity(emb, self.haystack_embeddings)
        return float(np.max(sims))

# --- 4. Define Data ---

# List of (Needle, Question, Target_Needle_Question_Similarity) tuples
needle = "A study measured per run totals for each option and then checked which option yielded the most stable results."

pairs_to_test = [
    # --- Target 0.6 (High Similarity - Open-ended, No Yes/No) ---
    (needle, "How did the study measure the totals per run to determine which option yielded the most stable results?", 0.6),
    (needle, "What method involving per-run totals was used to check which option yielded the most stable results?", 0.6),
    (needle, "Explain how measuring per run totals for each option helped identify the one with the most stable results.", 0.6),
    (needle, "What were the findings when a study measured per run totals to check for the most stable option?", 0.6),
    (needle, "Describe the process where a study measured totals per run to find the option with the most stable results.", 0.6),
    (needle, "In what way did the measurement of per run totals lead to identifying the option with the most stable results?", 0.6),
    (needle, "What specific totals were measured per run to check which option yielded the most stable results?", 0.6),
    (needle, "How did the researchers use per run totals to check which option yielded the most stable results?", 0.6),
    (needle, "What analysis of per run totals was conducted to find the option that yielded the most stable results?", 0.6),
    (needle, "Explain the study's approach to measuring per run totals and checking for the most stable results.", 0.6),
    (needle, "How were the totals for each option measured per run to determine the stability of the results?", 0.6),
    (needle, "What procedure was followed to measure per run totals and check for the option with the most stable results?", 0.6),
    (needle, "Describe the analysis of per run totals used to check which option yielded the most stable results.", 0.6),
    (needle, "How did the measurement of totals for each option per run assist in finding the most stable results?", 0.6),
    (needle, "What steps were taken to measure per run totals and check which option yielded the most stable results?", 0.6),
    (needle, "Explain the relationship between measuring per run totals and identifying the option with the most stable results.", 0.6),
    (needle, "How did the study evaluate per run totals to check which option yielded the most stable results?", 0.6),
    (needle, "What technique involving the measurement of per run totals was used to find the most stable results?", 0.6),
    (needle, "Describe how the totals for each option were measured per run to check for result stability.", 0.6),
    (needle, "In what manner were per run totals measured to check which option yielded the most stable results?", 0.6),
    (needle, "How did the study use the data from per run totals to check which option yielded the most stable results?", 0.6),
    (needle, "What was the outcome of measuring per run totals for each option to check for stable results?", 0.6),
    (needle, "Explain how the study determined which option yielded the most stable results by measuring per run totals.", 0.6),
    (needle, "How were per run totals utilized to check which option yielded the most stable results?", 0.6),
    (needle, "What approach did the study take in measuring per run totals to find the most stable results?", 0.6),
    (needle, "Describe the method used to measure per run totals and check which option yielded the most stable results.", 0.6),
    (needle, "How did the investigation measure per run totals to check which option yielded the most stable results?", 0.6),
    (needle, "What role did measuring per run totals play in checking which option yielded the most stable results?", 0.6),
    (needle, "Explain the process of checking for stable results by measuring per run totals for each option.", 0.6),
    (needle, "How did the study go about measuring per run totals to check which option yielded the most stable results?", 0.6),
    (needle, "What kind of totals were measured per run to check which option yielded the most stable results?", 0.6),
    (needle, "Describe the study's method of measuring per run totals to check for the most stable results.", 0.6),
    (needle, "How was the option with the most stable results identified through the measurement of per run totals?", 0.6),
    (needle, "What specific data from per run totals was used to check which option yielded the most stable results?", 0.6),
    (needle, "Explain how the researchers measured per run totals to check which option yielded the most stable results.", 0.6),
    (needle, "How did the study assess per run totals to check which option yielded the most stable results?", 0.6),
    (needle, "What was the significance of measuring per run totals in checking for the most stable results?", 0.6),
    (needle, "Describe the way in which per run totals were measured to check which option yielded the most stable results.", 0.6),
    (needle, "How did the study compare per run totals to check which option yielded the most stable results?", 0.6),
    (needle, "What strategy involving per run totals was used to check which option yielded the most stable results?", 0.6),
    (needle, "Explain the use of per run totals in checking which option yielded the most stable results.", 0.6),
    (needle, "How did the study quantify per run totals to check which option yielded the most stable results?", 0.6),
    (needle, "What evidence from measuring per run totals helped check which option yielded the most stable results?", 0.6),
    (needle, "Describe the findings related to measuring per run totals and checking for stable results.", 0.6),
    (needle, "How did the study interpret per run totals to check which option yielded the most stable results?", 0.6),
    (needle, "What connection was found between measuring per run totals and the option yielding the most stable results?", 0.6),
    (needle, "Explain the logic behind measuring per run totals to check which option yielded the most stable results.", 0.6),
    (needle, "How did the study validate which option yielded the most stable results using per run totals?", 0.6),
    (needle, "What insights were gained by measuring per run totals to check for the most stable results?", 0.6),
    (needle, "Describe the experimental design that measured per run totals to check which option yielded the most stable results.", 0.6),

    # --- Target 0.8 (Very High Similarity - Open-ended, No Yes/No) ---
    (needle, "What study measured per run totals for each option and then checked which option yielded the most stable results?", 0.8),
    (needle, "Which study measured per run totals for each option and checked which option yielded the most stable results?", 0.8),
    (needle, "How did a study measure per run totals for each option and check which option yielded the most stable results?", 0.8),
    (needle, "What did the study measure regarding per run totals for each option to check which yielded the most stable results?", 0.8),
    (needle, "Describe how a study measured per run totals for each option and then checked which option yielded the most stable results.", 0.8),
    (needle, "In what way did a study measure per run totals for each option and check which option yielded the most stable results?", 0.8),
    (needle, "What investigation measured per run totals for each option and then checked which option yielded the most stable results?", 0.8),
    (needle, "How was it determined which option yielded the most stable results after a study measured per run totals?", 0.8),
    (needle, "What research measured per run totals for each option and then checked which option yielded the most stable results?", 0.8),
    (needle, "Explain the study that measured per run totals for each option and then checked which option yielded the most stable results.", 0.8),
    (needle, "How did the researchers measure per run totals for each option and then check which option yielded the most stable results?", 0.8),
    (needle, "What experiment measured per run totals for each option and then checked which option yielded the most stable results?", 0.8),
    (needle, "Describe the analysis that measured per run totals for each option and then checked which option yielded the most stable results.", 0.8),
    (needle, "How did the analysis measure per run totals for each option and then check which option yielded the most stable results?", 0.8),
    (needle, "What project measured per run totals for each option and then checked which option yielded the most stable results?", 0.8),
    (needle, "Explain how the investigation measured per run totals for each option and then checked which option yielded the most stable results.", 0.8),
    (needle, "How did the team measure per run totals for each option and then check which option yielded the most stable results?", 0.8),
    (needle, "What work measured per run totals for each option and then checked which option yielded the most stable results?", 0.8),
    (needle, "Describe the method that measured per run totals for each option and then checked which option yielded the most stable results.", 0.8),
    (needle, "How did the process measure per run totals for each option and then check which option yielded the most stable results?", 0.8),
    (needle, "What inquiry measured per run totals for each option and then checked which option yielded the most stable results?", 0.8),
    (needle, "Explain the research that measured per run totals for each option and then checked which option yielded the most stable results.", 0.8),
    (needle, "How did the trial measure per run totals for each option and then check which option yielded the most stable results?", 0.8),
    (needle, "What assessment measured per run totals for each option and then checked which option yielded the most stable results?", 0.8),
    (needle, "Describe the test that measured per run totals for each option and then checked which option yielded the most stable results.", 0.8),
    (needle, "How did the evaluation measure per run totals for each option and then check which option yielded the most stable results?", 0.8),
    (needle, "What examination measured per run totals for each option and then checked which option yielded the most stable results?", 0.8),
    (needle, "Explain the experiment that measured per run totals for each option and then checked which option yielded the most stable results.", 0.8),
    (needle, "How did the survey measure per run totals for each option and then check which option yielded the most stable results?", 0.8),
    (needle, "What review measured per run totals for each option and then checked which option yielded the most stable results?", 0.8),
    (needle, "Describe the project that measured per run totals for each option and then checked which option yielded the most stable results.", 0.8),
    (needle, "How did the report measure per run totals for each option and then check which option yielded the most stable results?", 0.8),
    (needle, "What paper measured per run totals for each option and then checked which option yielded the most stable results?", 0.8),
    (needle, "Explain the analysis that measured per run totals for each option and then checked which option yielded the most stable results.", 0.8),
    (needle, "How did the study go about measuring per run totals for each option and then checking which option yielded the most stable results?", 0.8),
    (needle, "What specific study measured per run totals for each option and then checked which option yielded the most stable results?", 0.8),
    (needle, "Describe the specific study that measured per run totals for each option and then checked which option yielded the most stable results.", 0.8),
    (needle, "How did the specific study measure per run totals for each option and then check which option yielded the most stable results?", 0.8),
    (needle, "What particular study measured per run totals for each option and then checked which option yielded the most stable results?", 0.8),
    (needle, "Explain the particular study that measured per run totals for each option and then checked which option yielded the most stable results.", 0.8),
    (needle, "How did the particular study measure per run totals for each option and then check which option yielded the most stable results?", 0.8),
    (needle, "What detailed study measured per run totals for each option and then checked which option yielded the most stable results?", 0.8),
    (needle, "Describe the detailed study that measured per run totals for each option and then checked which option yielded the most stable results.", 0.8),
    (needle, "How did the detailed study measure per run totals for each option and then check which option yielded the most stable results?", 0.8),
    (needle, "What comprehensive study measured per run totals for each option and then checked which option yielded the most stable results?", 0.8),
    (needle, "Explain the comprehensive study that measured per run totals for each option and then checked which option yielded the most stable results.", 0.8),
    (needle, "How did the comprehensive study measure per run totals for each option and then check which option yielded the most stable results?", 0.8),
    (needle, "What exact study measured per run totals for each option and then checked which option yielded the most stable results?", 0.8),
    (needle, "Describe the exact study that measured per run totals for each option and then checked which option yielded the most stable results.", 0.8),
    (needle, "How did the exact study measure per run totals for each option and then check which option yielded the most stable results?", 0.8),
]
    


# --- 5. Main Execution Function ---

def main():
    """
    Main function to initialize models, run similarity tests,
    and print results.
    """
    
    # Initialize Haystack Checker
    print("Initializing Haystack Checker...")
    HAYSTACK_MAX_SIM = 0.5  # Questions with haystack similarity > this will be filtered out
    haystack_checker = HaystackDuplicateChecker(model_name="all-MiniLM-L6-v2", haystack_max_sim=HAYSTACK_MAX_SIM)
    try:
        haystack_checker.load_haystack(haystack_file="CognitiveBias4.txt")
        print(f"Haystack similarity threshold set to: {HAYSTACK_MAX_SIM}")
    except Exception as e:
        print(f"Warning: Could not load haystack ({e}). Haystack similarity will be 0.")

    # List of models to test
    try:
        models_to_test = [
            STModel(model_name="all-MiniLM-L6-v2")
        ]
        print(f"\nInitialized {len(models_to_test)} models for cross-validation.")
    except Exception as e:
        print(f"Error initializing models: {e}")
        print("Please ensure all API keys are set (e.g., GEMINI_API_KEY in .env) and packages are installed.")
        return # Exit main function if models fail

    all_results = []
    print("\n" + "="*60)
    print(f"Calculating Needle-Question Similarities for {len(pairs_to_test)} pairs...")
    print("="*60)

    for model in models_to_test:
        print(f"\n--- Testing with Model: {model.model_name} ---")

        # 1. Separate needles, questions, and targets
        needles = [pair[0] for pair in pairs_to_test]
        questions = [pair[1] for pair in pairs_to_test]
        targets = [pair[2] for pair in pairs_to_test]

        # 2. Embed all at once (batching)
        print(f"Embedding {len(needles)} needles and {len(questions)} questions...")
        needle_embeddings = model.embed(needles)
        question_embeddings = model.embed(questions)

        if needle_embeddings.size == 0 or question_embeddings.size == 0:
            print(f"Error: Embedding failed for model {model.model_name}. Skipping...")
            continue

        # Check for embedding count mismatch
        if len(needle_embeddings) != len(needles) or len(question_embeddings) != len(questions):
            print(f"Warning: Mismatch in embedding count for {model.model_name}.")
            continue

        # 3. Calculate similarity for each pair
        for i, pair in enumerate(pairs_to_test):
            n_emb = needle_embeddings[i:i+1]  # Keep 2D shape (1, D)
            q_emb = question_embeddings[i:i+1] # Keep 2D shape (1, D)

            needle_question_similarity = cosine_similarity(n_emb, q_emb)[0][0]
            
            # Check similarity to haystack (Novelty Check)
            haystack_sim = haystack_checker.get_max_similarity(pair[1])

            all_results.append({
                "pair_id": i,
                "model": model.model_name,
                "target": targets[i],
                "needle_question_sim": needle_question_similarity,
                "haystack_sim": haystack_sim,
                "needle": f"'{pair[0][:50]}...'",
                "question": f"'{pair[1][:50]}...'"
            })

    # --- 6. Display Individual Results ---
    print("\n" + "="*60)
    print("Individual Results (All Models)")
    print("="*60)

    if not all_results:
        print("No results generated. Check for embedding errors.")
        return

    results_df = pd.DataFrame(all_results)
    # Reorder columns
    cols = ["pair_id", "target", "needle_question_sim", "haystack_sim", "question"]
    print(results_df[cols].to_string(index=False))

    # --- 7. Filter Questions that Meet Target AND Cannot Be Answered by Haystack ---
    print("\n" + "="*80)
    print(f"VALID QUESTIONS (Target ±0.05 AND Haystack Sim ≤ {haystack_checker.haystack_max_sim})")
    print("="*80)

    valid_questions = []
    rejected_questions = []
    
    for idx, row in results_df.iterrows():
        difference = abs(row['needle_question_sim'] - row['target'])
        meets_target = difference <= 0.05
        not_in_haystack = row['haystack_sim'] <= haystack_checker.haystack_max_sim
        
        pair = pairs_to_test[row['pair_id']]
        
        question_info = {
            "Pair #": row['pair_id'],
            "Target": row['target'],
            "Actual N-Q Sim": round(row['needle_question_sim'], 4),
            "Haystack Sim": round(row['haystack_sim'], 4),
            "Target Diff": round(difference, 4),
            "Question": pair[1][:80] + "..." if len(pair[1]) > 80 else pair[1]
        }
        
        if meets_target and not_in_haystack:
            valid_questions.append(question_info)
        elif meets_target and not not_in_haystack:
            rejected_questions.append({**question_info, "Reason": "Haystack collision"})
    
    if valid_questions:
        valid_df = pd.DataFrame(valid_questions)
        print(f"\n✅ Found {len(valid_questions)} valid questions that meet criteria:\n")
        print(valid_df.to_string(index=False))
        
        # Summary by target
        print("\n" + "="*80)
        print("SUMMARY BY TARGET SIMILARITY")
        print("="*80)
        for target in [0.2, 0.4, 0.6, 0.8]:
            count = sum(1 for q in valid_questions if q['Target'] == target)
            if count > 0:
                print(f"Target {target}: {count} valid questions")
    else:
        print("❌ No questions found that meet BOTH criteria (target similarity AND low haystack similarity).")
    
    # Show rejected questions that met target but collided with haystack
    if rejected_questions:
        print("\n" + "="*80)
        print(f"⚠️ REJECTED: Questions that met target but have haystack similarity > {haystack_checker.haystack_max_sim}")
        print("="*80)
        rejected_df = pd.DataFrame(rejected_questions)
        print(rejected_df[['Pair #', 'Target', 'Actual N-Q Sim', 'Haystack Sim', 'Question']].to_string(index=False))
    
    # Show closest pairs if no valid questions found
    if not valid_questions:
        print("\n" + "="*80)
        print("ANALYSIS: Closest pairs to target (regardless of haystack)")
        print("="*80)
        results_df['difference'] = abs(results_df['needle_question_sim'] - results_df['target'])
        closest_indices = results_df.nsmallest(10, 'difference').index
        closest_analysis = []
        for idx in closest_indices:
            row = results_df.loc[idx]
            pair = pairs_to_test[row['pair_id']]
            closest_analysis.append({
                "Pair #": row['pair_id'],
                "Target": row['target'],
                "N-Q Sim": round(row['needle_question_sim'], 4),
                "H Sim": round(row['haystack_sim'], 4),
                "Diff": round(abs(row['needle_question_sim'] - row['target']), 4),
                "Question": pair[1][:80] + "..." if len(pair[1]) > 80 else pair[1]
            })
        closest_df = pd.DataFrame(closest_analysis)
        print(closest_df.to_string(index=False))


# --- 5. Standard Python Entry Point ---
if __name__ == "__main__":
    main()