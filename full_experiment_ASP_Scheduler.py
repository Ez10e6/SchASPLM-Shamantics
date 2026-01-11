# We illustrate the generation of a scheduling problem using a local model  (using chain-of-thought and few-shot prompting)
# Notice you have to have your api-key under .env

# Doing all the imports in this cell because each local GPU run needs a kernel restart. 
# Without a kernel restart, CPU is used instead of GPU after loading a local model once.
from LLM import bots
from ASP_Scheduler.problem_descriptions import all_problems
from ASP_Scheduler import scheduler
import os
import torch
from datetime import datetime
from utils import logger

###########################################################
#                        SETTINGS                         #
###########################################################

# GENERAL SETTINGS
RUN_LOCAL_MODEL = True         # Set to True to run a local model, False to run a remote model via OpenAI API
PRINT = True                    # Set to True to print intermediate outputs
PROGRAM_FOLDER = 'Results/experiment'     # Folder to save programs (set to None to disable saving)
METRICS_LOG_FILE = 'metrics/metrics.csv'

# REMOTE SETTINGS
# REMOTE_PIPE = None # None defaults to meta-llama/Meta-Llama-3-8B-Instruct
# REMOTE_PIPE = 'deepseek' # 'deepseek' model on OpenAI API
# REMOTE_PIPE_SEMANTICS = None # None defaults to meta-llama/Meta-Llama-3-8B-Instruct
# REMOTE_PIPE_SEMANTICS = 'deepseek' # 'deepseek' model on OpenAI API

# LOCAL SETTINGS
# CHECKPOINT, CHECKPOINT_SHORT_NAME = "meta-llama/Meta-Llama-3-8B-Instruct", "Llama-3-8B-Instruct" 
CHECKPOINT, CHECKPOINT_SHORT_NAME = "Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"
# CHECKPOINT, CHECKPOINT_SHORT_NAME = "ft_llama_mlx", "ft_llama_mlx"
# CHECKPOINT, CHECKPOINT_SHORT_NAME = "ft_qwen_mlx", "ft_qwen_mlx"
# CHECKPOINT_SEMANTICS, CHECKPOINT_SHORT_NAME_SEMANTICS = "meta-llama/Meta-Llama-3-8B-Instruct", "Llama-3-8B-Instruct" 
CHECKPOINT_SEMANTICS, CHECKPOINT_SHORT_NAME_SEMANTICS = "Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"

# DYNAMIC HARDWARE CONFIGURATION
if torch.cuda.is_available():
    # Settings for NVIDIA GPUs
    print("CUDA detected. Using NVIDIA configuration.")
    os.environ["BNB_CUDA_VERSION"] = "123"  # Force bnb version for Windows/Cuda if needed
    QUANTIZATION_CONFIG = '4bit'          # '4bit', '8bit' supported on CUDA
else:
    # Settings for Mac (MPS) or CPU
    print("CUDA not detected. Using MPS/CPU configuration (Quantization disabled).")
    QUANTIZATION_CONFIG = None            # bitsandbytes quantization is not supported on MPS yet

# SAMPLING / REPRODUCIBILITY SETTINGS
# - Set SEED = -1 to disable fixed seeding (non-deterministic runs). Set to an integer for reproducible runs.
SEED = 42
MAX_NEW_TOKENS = 512 #Max tokens for response. Should be in balance with the model's context size

# TEMPERATURE = 0.2  # Less deterministic
# TOP_P = 0.9        # Less deterministic
TEMPERATURE = 0.01 # More deterministic (same as local settings of original experiments)
TOP_P = 0          # More deterministic (same as local settings of original experiments)

# PROBLEM SETTINGS
# PROBLEM_NAMES = ['sports scheduling']
# PROBLEM_NAMES = ['nurse_scheduling']
# PROBLEM_NAMES = ['nurse_scheduling', 'sports scheduling']
# PROBLEM_NAMES = ['post_enrollment_based_course_time_tabling', 'examination_timetabling']
PROBLEM_NAMES = list(all_problems.keys())  # To run the program for ALL available problem names
MAX_SYNTAX_REPAIRS = 3  # Maximum number of repair attempts per statement block for syntax errors
MAX_SEMANTIC_REPAIRS = 3  # Maximum number of repair attempts per statement block for semantic errors
RUNS_PER_PROBLEM = 1  # Number of runs per problem for averaging results

if RUN_LOCAL_MODEL:
    # To work locally, we need to manually load the pipeline 
    PIPE = bots.load_pipe(model_checkpoint=CHECKPOINT, local_dir="./local_models", quantization_config=QUANTIZATION_CONFIG, save=True)
    SEMANTICS_PIPE = bots.load_pipe(model_checkpoint=CHECKPOINT_SEMANTICS, local_dir="./local_models", quantization_config=QUANTIZATION_CONFIG, save=True)  
else:
    # For remote models, we set pipe to a string with the model name
    PIPE = REMOTE_PIPE
    SEMANTICS_PIPE = REMOTE_PIPE_SEMANTICS

# Run the LLM scheduler per problem
for problem_name in PROBLEM_NAMES:
    for run_id in range(RUNS_PER_PROBLEM):        
        # Initialize the metrics logger
        # Build a model identifier string for the logfile (include LOCAL/REMOTE)
        model_id = (f"{CHECKPOINT} (LOCAL, QUANTIZATION: {QUANTIZATION_CONFIG})" if RUN_LOCAL_MODEL else (f"{REMOTE_PIPE} (REMOTE)" if REMOTE_PIPE is not None else "Meta-Llama-3-8B-Instruct"))
        semantics_model_id = (f"{CHECKPOINT_SEMANTICS} (LOCAL, QUANTIZATION: {QUANTIZATION_CONFIG})" if RUN_LOCAL_MODEL else (f"{REMOTE_PIPE_SEMANTICS} (REMOTE)" if REMOTE_PIPE_SEMANTICS is not None else "Meta-Llama-3-8B-Instruct"))
        logger.init_logger(filename=METRICS_LOG_FILE,
                           problem_ID=problem_name,
                           max_fix_attempts=MAX_SYNTAX_REPAIRS,
                           model=model_id,
                           semantics_model=semantics_model_id,
                           temperature=TEMPERATURE,
                           top_p=TOP_P,
                           seed=SEED)

        full_program = scheduler.full_ASP_program(
            all_problems[problem_name],    # Input problem specifications for examination timetabling
            pipe=PIPE,                     # Input the PIPEline object for the LLM
            semantic_validation_pipe=SEMANTICS_PIPE, # Input the PIPEline object for the semantics validation LLM
            printer=PRINT,                 # Set to True to print intermediate outputs
            k=MAX_SYNTAX_REPAIRS,                 # Max repairs
            n=MAX_SEMANTIC_REPAIRS,                 # Max repairs
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=(None if SEED == -1 else SEED),
            max_new_tokens=MAX_NEW_TOKENS)
                            

        if PROGRAM_FOLDER is not None:
            # Save the full program to a file
            os.makedirs(PROGRAM_FOLDER, exist_ok=True)
            timestamp = logger.time_stamp()
            if RUN_LOCAL_MODEL:
                model_string = CHECKPOINT_SHORT_NAME
                if QUANTIZATION_CONFIG is not None:
                    # Append quantization info like " (quant 4bit)" or " (quant 8bit)"
                    model_string = f"{model_string} (quant {QUANTIZATION_CONFIG})"
            else:
                model_string = REMOTE_PIPE if REMOTE_PIPE is not None else "Meta-Llama-3-8B-Instruct"
            max_repairs_string = f"_k={MAX_SYNTAX_REPAIRS}" if MAX_SYNTAX_REPAIRS is not None else ""
            max_sematics_repairs_string = f"_n={MAX_SEMANTIC_REPAIRS}" if MAX_SEMANTIC_REPAIRS is not None else ""
            program_filename = os.path.join(PROGRAM_FOLDER, f"{problem_name}_{model_string}{max_repairs_string}{max_sematics_repairs_string}_{timestamp}.lp")
            with open(program_filename, 'w', encoding='utf-8') as f:
                f.write(full_program)
            if PRINT:
                print(f"Full program saved to {program_filename}")
        else:
            # Print the full program as it is returned by the scheduler
            print('----------------------------FULL PROGRAM----------------------------')
            print(full_program)
