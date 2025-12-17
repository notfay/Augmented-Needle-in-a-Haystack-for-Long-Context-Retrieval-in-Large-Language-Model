# run_combined_test.py
import os
import re
import json
import time
import csv
import numpy as np
from openai import OpenAI
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
import atexit
import signal

# --- KONFIGURASI & SETUP ---
load_dotenv()

# ==================================================================
# =================== BAGIAN YANG HILANG & DIPERBAIKI =================
# ==================================================================
# Class ini hilang di file Anda, menyebabkan NameError
class Position(Enum):
    START = "start"
    MIDDLE = "middle"
    END = "end"
    RANDOM = "random"
# ==================================================================

@dataclass
class NeedleConfig:
    text: str
    custom_position_percent: Optional[float] = None
    # Tambahkan ID dan similarity untuk pencatatan
    needle_id: int = 0
    haystack_similarity: float = 0.0

@dataclass
class QuestionConfig:
    text: str
    question_similarity: float = 0.0

# ... (Class AugmentedNeedleHaystack dan GeminiProvider tetap sama) ...
class AugmentedNeedleHaystack:
    def __init__(self, haystack_text: str):
        self.haystack_text = haystack_text
        self.haystack_sentences = self._split_into_sentences(haystack_text)

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    # Sekarang 'Position' dikenali karena class-nya sudah ditambahkan di atas
    def _calculate_position_index(self, position: Position = None, custom_position_percent: float = None) -> int:
        total_sentences = len(self.haystack_sentences)
        if custom_position_percent is not None:
            percent = max(0.0, min(1.0, custom_position_percent))
            index = int(total_sentences * percent)
            return min(index, total_sentences - 1) if total_sentences > 0 else 0
        
        # Bagian ini tidak akan terpakai di 'main' Anda saat ini,
        # tapi kami biarkan agar tidak error jika Anda butuh lagi
        if position == Position.START:
            return random.randint(0, max(1, int(total_sentences * 0.20)))
        elif position == Position.MIDDLE:
            return random.randint(int(total_sentences * 0.40), int(total_sentences * 0.60))
        elif position == Position.END:
            return random.randint(int(total_sentences * 0.80), max(0, total_sentences - 1))
        elif position == Position.RANDOM:
            return random.randint(0, max(0, total_sentences - 1))
            
        return total_sentences // 2

    def create_context(self, needle_config: NeedleConfig) -> str:
        sentences = self.haystack_sentences.copy()
        needle_pos = self._calculate_position_index(
            custom_position_percent=needle_config.custom_position_percent
        )
        sentences.insert(needle_pos, needle_config.text)
        return ' '.join(sentences)

class OpenAIProvider:
    def __init__(self, model_name: str = "qwen/qwen3-next-80b-a3b-instruct", base_url: str | None = None):
        self.model_name = model_name
        # Expect the NVIDIA-style base_url and API key in env
        self.base_url = base_url or os.getenv("NV_API_BASE_URL") or os.getenv("NV_BASE_URL") or "https://integrate.api.nvidia.com/v1"
        self.api_key = os.getenv("NV_API_KEY") or os.getenv("NVIDIA_API_KEY") or os.getenv("NVAPI_KEY") or ""
        if not self.api_key:
            raise ValueError("NVIDIA API key not found in environment (NV_API_KEY / NVIDIA_API_KEY / NVAPI_KEY).")

        # Create OpenAI client configured for the provided base_url
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def evaluate(self, prompt: str, max_retries: int = 3, retry_delay: int = 5) -> str:
        for attempt in range(max_retries):
            try:
                # The prompt already includes instructions for short, direct answers
                messages = [
                    {"role": "user", "content": prompt}
                ]

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    top_p=0.95,
                    max_tokens=4096,
                    stream=False
                )

                # For non-streaming, access the content directly
                if hasattr(completion, 'choices') and completion.choices:
                    message = completion.choices[0].message
                    if hasattr(message, 'content') and message.content:
                        return message.content

                return ""
            except Exception as e:
                error_msg = str(e)
                print(f"Attempt {attempt + 1} failed: {error_msg}")
                
                # Check if it's a rate limit or quota error that should be retried
                is_retryable_error = any(keyword in error_msg.lower() for keyword in [
                    'rate limit', '429', 'quota', 'exhausted', 'timeout', 'connection',
                    '503', '500', '502', '504'
                ])
                
                if attempt < max_retries - 1 and is_retryable_error:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Exponential backoff
                    retry_delay *= 2
                else:
                    # Final attempt or non-retryable error
                    print(f"Final error or max retries reached: {e}")
                    return "Error: Could not retrieve response from model."

# --- RUNNER DIPERBARUI UNTUK MENANGANI QUESTION YANG BERUBAH-UBAH ---
class ExperimentRunner:
    def __init__(self, provider: object, haystack_text: str, delay_seconds: int = 0, checkpoint_file: str = "checkpoint.json", emergency_file: str = "emergency_save.json"):
        self.provider = provider
        self.haystack_text = haystack_text
        self.delay_seconds = delay_seconds
        self.results = []
        self.checkpoint_file = checkpoint_file
        self.emergency_file = emergency_file
        self.current_experiment_index = 0
        self.total_experiments = 0
        
        # Register emergency save handlers
        atexit.register(self._emergency_save)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def run_experiment(self, experiment_config: Dict):
        # Ambil Question dari config, bukan dari __init__
        needle_config = experiment_config['needle']
        question_config = experiment_config['question']
        
        print(f"\n--- Running Experiment: {experiment_config['name']} ---")
        
        niah = AugmentedNeedleHaystack(self.haystack_text)
        augmented_context = niah.create_context(needle_config=needle_config)
        
        prompt = f"You are a helpful AI bot that answers questions for a user. Keep your response short and direct.\n\nDOCUMENT:\n{augmented_context}\n\nQUESTION:\n{question_config.text}\n\nDon't give information from outside the document / context given or repeat your findings. Explain the answers, Dont give a yes/no answers."
        
        # DISABLED: Token counting doubles API usage by sending the full haystack twice
        # token_count = self.provider.count_tokens(prompt)
        # print(f"Verified prompt token count: {token_count}") 
        
        start_time = time.time()
        response_text = self.provider.evaluate(prompt)
        end_time = time.time()
        
        latency = end_time - start_time
        print(f"Response: {response_text[:100]}...")
        print(f"Latency: {latency:.2f} seconds")

        result_data = {
            "experiment_name": experiment_config['name'],
            "config": {
                "needle": asdict(needle_config),
                "question": asdict(question_config) 
            },
            "response": response_text,
            "latency_seconds": latency,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result_data)

    def run_all(self, experiments: List[Dict], resume_from_checkpoint: bool = True):
        self.total_experiments = len(experiments)
        start_index = 0
        
        # Try to resume from checkpoint
        if resume_from_checkpoint:
            start_index = self._load_checkpoint()
            if start_index > 0:
                print(f"\n>>> RESUMING from checkpoint at experiment {start_index + 1}/{self.total_experiments}")
        
        for i in range(start_index, len(experiments)):
            self.current_experiment_index = i
            exp = experiments[i]
            print(f"\n>>> Progress: {i + 1}/{self.total_experiments}")
            
            # Save checkpoint BEFORE attempting the experiment
            # This ensures we can resume from the exact failed experiment
            self._save_checkpoint()
            
            try:
                self.run_experiment(exp)
            except Exception as e:
                error_msg = str(e)
                print(f"\n!!! ERROR in experiment {i + 1}: {error_msg}")
                
                # Check if it's an API exhaustion or quota error
                is_api_error = any(keyword in error_msg.lower() for keyword in [
                    'quota', 'exhausted', 'rate limit', '429', '403', '503', '500',
                    'resource exhausted', 'billing', 'insufficient funds'
                ])
                
                if is_api_error:
                    print(">>> API ERROR DETECTED (quota exhausted or rate limited)")
                    print(">>> Stopping execution to prevent further API calls")
                    print(">>> Progress saved. You can resume from this exact point later.")
                else:
                    print(">>> Unexpected error occurred")
                    print(">>> Saving emergency backup...")
                
                # Save emergency backup with error details
                self._emergency_save_with_error(error_msg, i, exp)
                
                # Don't re-raise - just exit gracefully
                print(f"\n>>> Execution stopped at experiment {i + 1}")
                print(">>> Run the script again to resume from this point.")
                exit(1)  # Exit with error code
            
            if i < len(experiments) - 1 and self.delay_seconds > 0:
                print(f"\nWaiting for {self.delay_seconds} seconds before the next API call...")
                time.sleep(self.delay_seconds)

    def save_results(self, output_file: str = "Qwen3.json"):
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4)
        print(f"\n✓ All results saved to: {output_path}")
        
        # Clean up checkpoint file after successful completion
        self._cleanup_checkpoint()
    
    def _save_checkpoint(self):
        """Save current progress to checkpoint file."""
        checkpoint_data = {
            "last_completed_index": self.current_experiment_index,
            "total_experiments": self.total_experiments,
            "results": self.results,
            "timestamp": datetime.now().isoformat()
        }
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
    
    def _load_checkpoint(self) -> int:
        """Load checkpoint and return the index to resume from."""
        checkpoint_path = Path(self.checkpoint_file)
        emergency_path = Path(self.emergency_file)
        
        # First check for emergency save (takes priority)
        if emergency_path.exists():
            try:
                with open(emergency_path, 'r', encoding='utf-8') as f:
                    emergency_data = json.load(f)
                
                if emergency_data.get('error_save', False):
                    # This is an error save - resume from the failed experiment
                    failed_index = emergency_data.get('failed_experiment_index', -1)
                    error_msg = emergency_data.get('error_message', 'unknown error')
                    failed_name = emergency_data.get('failed_experiment_name', 'unknown')
                    
                    print(f"\n>>> EMERGENCY SAVE FOUND (from error):")
                    print(f"   Failed experiment: {failed_name}")
                    print(f"   Error: {error_msg}")
                    print(f"   Will resume from experiment {failed_index + 1}")
                    
                    self.results = emergency_data.get('results', [])
                    return failed_index  # Resume from the failed experiment
                else:
                    # Regular emergency save
                    self.results = emergency_data.get('results', [])
                    last_index = emergency_data.get('last_experiment_index', -1)
                    print(f"\n>>> Emergency save found: {len(self.results)} experiments completed")
                    return last_index + 1
            except Exception as e:
                print(f"Warning: Could not load emergency save: {e}")
        
        # Check for regular checkpoint
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                self.results = checkpoint_data.get('results', [])
                last_completed = checkpoint_data.get('last_completed_index', -1)
                
                print(f"\n>>> Checkpoint found: {len(self.results)} experiments completed")
                print(f">>> Last completed: {checkpoint_data.get('timestamp', 'unknown')}")
                
                # Resume from next experiment
                return last_completed + 1
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                return 0
        
        return 0
    
    def _cleanup_checkpoint(self):
        """Remove checkpoint file after successful completion."""
        try:
            if Path(self.checkpoint_file).exists():
                os.remove(self.checkpoint_file)
                print(f"✓ Checkpoint file cleaned up")
        except Exception as e:
            print(f"Warning: Could not remove checkpoint file: {e}")
    
    def _emergency_save(self):
        """Emergency save in case of crash or interruption."""
        if not self.results:
            return
        
        emergency_data = {
            "emergency_save": True,
            "saved_at": datetime.now().isoformat(),
            "completed_experiments": len(self.results),
            "last_experiment_index": self.current_experiment_index,
            "total_experiments": self.total_experiments,
            "results": self.results
        }
        
        try:
            with open(self.emergency_file, 'w', encoding='utf-8') as f:
                json.dump(emergency_data, f, indent=4)
            print(f"\n⚠️  EMERGENCY SAVE: {len(self.results)} results saved to {self.emergency_file}")
        except Exception as e:
            print(f"\n!!! CRITICAL: Emergency save failed: {e}")
    
    def _emergency_save_with_error(self, error_message: str, failed_experiment_index: int, failed_experiment: Dict):
        """Emergency save with detailed error information."""
        emergency_data = {
            "emergency_save": True,
            "error_save": True,
            "saved_at": datetime.now().isoformat(),
            "error_message": error_message,
            "failed_experiment_index": failed_experiment_index,
            "failed_experiment_name": failed_experiment.get('name', 'unknown'),
            "completed_experiments": len(self.results),
            "total_experiments": self.total_experiments,
            "results": self.results,
            "resume_from_index": failed_experiment_index  # Resume from the failed experiment
        }
        
        try:
            with open(self.emergency_file, 'w', encoding='utf-8') as f:
                json.dump(emergency_data, f, indent=4)
            print(f"\n🚨 EMERGENCY ERROR SAVE: {len(self.results)} results + error details saved to {self.emergency_file}")
            print(f"   Failed at experiment {failed_experiment_index + 1}: {failed_experiment.get('name', 'unknown')}")
        except Exception as e:
            print(f"\n!!! CRITICAL: Emergency error save failed: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle interruption signals (Ctrl+C, etc.)."""
        print("\n\n>>> Interruption detected! Saving progress...")
        self._emergency_save()
        print(">>> Progress saved. You can resume by running the script again.")
        exit(0)

# --- FUNGSI BARU UNTUK MEMBACA FILE CSV ---
def load_needles_and_questions(filepath: str) -> List[Dict]:
    """Membaca file CSV dan mengembalikannya sebagai daftar dictionary."""
    data = []
    try:
        with open(filepath, mode='r', encoding='utf-8-sig') as csvfile: # 'utf-8-sig' untuk menangani BOM
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        print(f">>> Successfully loaded {len(data)} needle/question pairs from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Please create it.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return []

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # ==================================================================
    # =================== EXPERIMENT CONTROL PANEL =====================
    # ==================================================================
    HAYSTACK_FILE = "CognitiveBias4.txt"
    # Nama file yang berisi needle dan question
    NEEDLE_QUESTION_FILE = "final.csv" 
    DELAY_BETWEEN_CALLS_SECONDS = 62
    # ==================================================================
    
    print(">>> Generating experiment plan from control panel settings...")
    
    # 1. Muat data Needle dan Question dari file
    nq_data = load_needles_and_questions(NEEDLE_QUESTION_FILE)
    if not nq_data:
        exit() # Keluar jika file tidak ada atau kosong

    # 2. Tentukan posisi yang akan diuji
    needle_positions = np.linspace(0, 1, 11) # 0, 0.1, 0.2 ... 1.0

    # 3. Buat rencana eksperimen dengan loop bersarang
    experiments_to_run = []
    for percent in needle_positions:
        for item in nq_data:
            try:
                # Ambil data dari baris CSV
                h_sim = float(item['haystack_similarity'])
                q_sim = float(item['question_similarity'])
                n_id = int(item['needle_id'])

                # Buat nama eksperimen yang deskriptif
                exp_name = f"Pos: {percent*100:.0f}%, N-ID: {n_id}, H-Sim: {h_sim}, Q-Sim: {q_sim}"
                
                # Buat objek config
                needle_config = NeedleConfig(
                    text=item['needle_text'],
                    custom_position_percent=float(percent),
                    needle_id=n_id,
                    haystack_similarity=h_sim
                )
                question_config = QuestionConfig(
                    text=item['question_text'],
                    question_similarity=q_sim
                )
                
                experiments_to_run.append({
                    "name": exp_name,
                    "needle": needle_config,
                    "question": question_config
                })
            except KeyError as e:
                print(f"Error: Missing column {e} in your CSV file. Please check the header.")
                exit()
            except ValueError as e:
                print(f"Error: Could not convert value in CSV to number (float/int). Check {e}")
                exit()

    print(f">>> Plan generated. Total experiments to run: {len(experiments_to_run)}")
    # for i, exp in enumerate(experiments_to_run):
    #     print(f"  {i+1}. {exp['name']}") # Matikan ini agar tidak terlalu panjang

    try:
        with open(HAYSTACK_FILE, "r", encoding="utf-8") as f:
            base_haystack = f.read()
    except FileNotFoundError:
        print(f"Error: Haystack file '{HAYSTACK_FILE}' not found. Please create it.")
        exit()
        
    openai_provider = OpenAIProvider(model_name="qwen/qwen3-next-80b-a3b-instruct")
    runner = ExperimentRunner(
        provider=openai_provider,
        haystack_text=base_haystack,
        delay_seconds=DELAY_BETWEEN_CALLS_SECONDS
    )
    
    # Check if checkpoint or emergency save exists and ask user
    resume_choice = True  # Default to resume
    checkpoint_path = Path(runner.checkpoint_file)
    emergency_path = Path(runner.emergency_file)
    
    save_exists = checkpoint_path.exists() or emergency_path.exists()
    
    if save_exists:
        print("\n" + "="*60)
        print("⚠️  PROGRESS SAVE FOUND!")
        print("="*60)
        
        # Check emergency save first
        if emergency_path.exists():
            try:
                with open(emergency_path, 'r', encoding='utf-8') as f:
                    emergency_data = json.load(f)
                
                completed = len(emergency_data.get('results', []))
                total = emergency_data.get('total_experiments', 0)
                timestamp = emergency_data.get('saved_at', 'unknown')
                
                if emergency_data.get('error_save', False):
                    failed_exp = emergency_data.get('failed_experiment_name', 'unknown')
                    error_msg = emergency_data.get('error_message', 'unknown error')[:100] + "..."
                    print(f"Emergency save from ERROR: {completed}/{total} experiments completed")
                    print(f"Failed experiment: {failed_exp}")
                    print(f"Error: {error_msg}")
                else:
                    print(f"Emergency save: {completed}/{total} experiments completed")
                
                print(f"Last saved: {timestamp}")
            except:
                print("Emergency save file exists but couldn't read details")
        
        # Check regular checkpoint
        elif checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                completed = len(checkpoint_data.get('results', []))
                total = checkpoint_data.get('total_experiments', 0)
                timestamp = checkpoint_data.get('timestamp', 'unknown')
                print(f"Checkpoint: {completed}/{total} experiments completed")
                print(f"Last saved: {timestamp}")
            except:
                print("Checkpoint file exists but couldn't read details")
        
        print("\nOptions:")
        print("  [R] Resume from save (continue where you left off)")
        print("  [S] Start over (delete saves and begin fresh)")
        print("  [Q] Quit")
        
        while True:
            choice = input("\nYour choice (R/S/Q): ").strip().upper()
            if choice == 'R':
                resume_choice = True
                print("\n>>> Resuming from save...")
                break
            elif choice == 'S':
                resume_choice = False
                # Delete both files
                for path in [checkpoint_path, emergency_path]:
                    try:
                        if path.exists():
                            os.remove(path)
                    except:
                        pass
                print("\n>>> Save files deleted. Starting fresh...")
                break
            elif choice == 'Q':
                print("\n>>> Exiting...")
                exit(0)
            else:
                print("Invalid choice. Please enter R, S, or Q.")
    else:
        print("\n>>> No save files found. Starting fresh...")
    
    runner.run_all(experiments_to_run, resume_from_checkpoint=resume_choice)
    runner.save_results()