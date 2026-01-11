from sched import scheduler
from LLM import bots
import time
import os
import re
import utils.utils as utils
from utils import logger
import json

BASE_DIR = os.path.dirname(__file__)

def read_system_prompt(file_path):
    ''' Read a system prompt from a file.

    Args:
        file_path (str): The path to the file containing the system prompt.

    Returns:
        str: The system prompt as a string.
    '''
    file_path = os.path.join(BASE_DIR, file_path)
    with open(file_path, 'r') as file:
        system_prompt = file.read()
    return system_prompt


def sleep_if_using_remote_clients(pipe, seconds=10):
    """Sleep for `seconds` only if the provided pipe indicates a remote client.

    Remote clients are represented by `pipe is None` (use HF API) or the
    string `'deepseek'` (the Deepseek provider).
    """
    if pipe is None or pipe == 'deepseek':
        time.sleep(seconds)

def get_hard_constraints(hard_constraint_descriptions, problem_description, instance_template, generator, pipe=None, semantic_validation_pipe=None, printer=False, k=0, n=0, temperature=None, top_p=None, seed=None, max_new_tokens=512):
    ''' Get hard constraints based on their descriptions. Uses different prompts based on the type of constraint.

    Args:
        hard_constraint_descriptions (list): A list of descriptions for each hard constraint.
        problem_description (str): The overall problem description.
        instance_template (str): The instance template generated from the instance description.
        generator (str): The generator generated from the generator description.
        pipe (optional): The pipeline to use for the LLM. Defaults to None.
        semantic_validation_pipe (optional): The pipeline to use for semantic validation. Defaults to None.
        printer (bool, optional): Whether to print intermediate results. Defaults to False.
        k (int, optional): The number of retries to get a syntactically correct response. Defaults to 0 (no retries).
        n (int, optional): The number of retries to get a semantically valid response. Defaults to 0 (no retries). Semantic repair takes place after syntax repair.
        temperature (float, optional): The temperature setting for the LLM. Defaults to None.
        top_p (float, optional): The top_p setting for the LLM. Defaults to None.
        seed (int, optional): The seed for the LLM. Defaults to None.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.

    Returns:
        list: A list of hard constraints as strings.
    '''
    # If there are no hard constraints we return None
    if hard_constraint_descriptions is None:
        return None
    
    hard_constraints = []
    print('\n\nHard Constraints\n') if printer else None

    # For every hard constraint description
    for constraint_description in hard_constraint_descriptions:
        # Use the correct hard constraint prompt based on the type of constraint
        if 'type: count' in constraint_description.lower():
            # Remove type: count from the prompt
            constraint_description = constraint_description.replace('type: count', '')
            hard_constraint = get_partial_program(
                system_prompt_path='system_prompts/count_hard_constraints.txt',
                prompt=constraint_description,
                system_prompt_variables={
                    'problem_description': problem_description,
                    'instance_template': instance_template,
                    'generator': generator
                },
                pipe=pipe,
                semantic_validation_pipe=semantic_validation_pipe,
                k=k,
                n=n,
                printer=printer,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_new_tokens=max_new_tokens
            )
            sleep_if_using_remote_clients(pipe)
        elif 'type: sum' in constraint_description.lower():
            # Remove type: sum from the prompt
            constraint_description = constraint_description.replace('type: sum', '')
            hard_constraint = get_partial_program(
                system_prompt_path='system_prompts/sum_hard_constraints.txt',
                prompt=constraint_description,
                system_prompt_variables={
                    'problem_description': problem_description,
                    'instance_template': instance_template,
                    'generator': generator
                },
                pipe=pipe,
                semantic_validation_pipe=semantic_validation_pipe,
                k=k,
                n=n,
                printer=printer,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_new_tokens=max_new_tokens
            )
            sleep_if_using_remote_clients(pipe)
        else:
            hard_constraint = get_partial_program(
                system_prompt_path='system_prompts/regular_hard_constraints.txt',
                prompt=constraint_description,
                system_prompt_variables={
                    'problem_description': problem_description,
                    'instance_template': instance_template,
                    'generator': generator
                },
                pipe=pipe,
                semantic_validation_pipe=semantic_validation_pipe,
                k=k,
                n=n,
                printer=printer,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_new_tokens=max_new_tokens
            )
            sleep_if_using_remote_clients(pipe)
        
        # Append the generated hard constraint to the list
        hard_constraints.append(hard_constraint)

        print(constraint_description + '\n' + hard_constraints[-1]+ '\n') if printer else None
    
    return hard_constraints

# Get Soft Constraints
def get_soft_constraints(soft_constraint_descriptions, problem_description, instance_template, generator, pipe=None, semantic_validation_pipe=None, printer=False, k=0, n=0, temperature=None, top_p=None, seed=None, max_new_tokens=512):
    ''' Get soft constraints based on their descriptions. Uses different prompts based on the type of constraint.

    Args:
        soft_constraint_descriptions (list): A list of descriptions for each soft constraint.
        problem_description (str): The overall problem description.
        instance_template (str): The instance template generated from the instance description.
        generator (str): The generator generated from the generator description.
        pipe (optional): The pipeline to use for the LLM. Defaults to None.
        semantic_validation_pipe (optional): The pipeline to use for semantic validation. Defaults to None.
        printer (bool, optional): Whether to print intermediate results. Defaults to False.
        k (int, optional): The number of retries to get a syntactically correct response. Defaults to 0 (no retries).
        n (int, optional): The number of retries to get a semantically valid response. Defaults to 0 (no retries). Semantic repair takes place after syntax repair.
        temperature (float, optional): The temperature setting for the LLM. Defaults to None.
        top_p (float, optional): The top_p setting for the LLM. Defaults to None.
        seed (int, optional): The seed for the LLM. Defaults to None.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.

    Returns:
        list: A list of soft constraints as strings.
    '''
    # If there are no soft constraints we return None
    if soft_constraint_descriptions is None:
        return None
    
    soft_constraints = []
    print('\nSoft Constraints:\n') if printer else None

    # For every hard constraint description
    for constraint_description in soft_constraint_descriptions:
        # Use the correct soft constraint prompt based on the type of constraint
        if 'type: count' in constraint_description.lower():
            # Remove type: count from the prompt
            constraint_description = constraint_description.replace('type: count', '')
            soft_constraint = get_partial_program(
                system_prompt_path='system_prompts/count_soft_constraints.txt',
                prompt=constraint_description,
                system_prompt_variables={
                    'problem_description': problem_description,
                    'instance_template': instance_template,
                    'generator': generator
                },
                pipe=pipe,
                semantic_validation_pipe=semantic_validation_pipe,
                k=k,
                n=n,
                printer=printer,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_new_tokens=max_new_tokens
            )
            sleep_if_using_remote_clients(pipe)
        elif 'type: sum' in constraint_description.lower():
            # Remove type: sum from the prompt
            constraint_description = constraint_description.replace('type: sum', '')
            soft_constraint = get_partial_program(
                system_prompt_path='system_prompts/sum_soft_constraints.txt',
                prompt=constraint_description,
                system_prompt_variables={
                    'problem_description': problem_description,
                    'instance_template': instance_template,
                    'generator': generator
                },
                pipe=pipe,
                semantic_validation_pipe=semantic_validation_pipe,
                k=k,
                n=n,
                printer=printer,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_new_tokens=max_new_tokens
            )
            sleep_if_using_remote_clients(pipe)
        else:
            soft_constraint = get_partial_program(
                system_prompt_path='system_prompts/regular_soft_constraints.txt',
                prompt=constraint_description,
                system_prompt_variables={
                    'problem_description': problem_description,
                    'instance_template': instance_template,
                    'generator': generator
                },
                pipe=pipe,
                semantic_validation_pipe=semantic_validation_pipe,
                k=k,
                n=n,
                printer=printer,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_new_tokens=max_new_tokens
            )
            sleep_if_using_remote_clients(pipe)

        # Append the generated soft constraint to the list
        soft_constraints.append(soft_constraint)

        print(constraint_description + '\n' + soft_constraints[-1]+ '\n') if printer else None
    
    return soft_constraints

def extract_constraints(descriptions, constraints):
    ''' Extract ASP constraints from the LLM output, removing markdown and comments. Also add the description as a comment before each constraint.

    Args:
        descriptions (list): A list of descriptions for each constraint.
        constraints (list): A list of constraints as returned by the LLM.

    Returns:
        str: A string containing all the extracted constraints with descriptions as comments.
    '''
    program = ''
    for i in range(len(descriptions)):
        description = descriptions[i]
        description = description.splitlines()
        description = description[0]
        description = description[2:]
        output = constraints[i]

        output_lines = output.splitlines()
        asp = ''
        for line in output_lines:
            if '```' in line:
                continue
            if '%' in line:
                continue
            else:
                asp += line + '\n'
        
        program += f'''% {description}\n{asp}\n\n'''
    
    return program

def extract_bullet_points(text):
    ''' Extract bullet points from a text block, where main points start with '- ' and can span multiple lines.

    Args:
        text (str): The input text containing bullet points.

    Returns:
        list: A list of strings, each representing a main point with its sub-points.
    '''
    lines = text.split('\n')
    result = []
    current_point = ""

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('- '):  # Main point
            if current_point:  # Save the previous point
                result.append(current_point.strip())
            current_point = stripped_line  # Start a new point
        else:  # Sub-point
            current_point += f"\n {stripped_line}"  # Append to the current point

    if current_point:  # Append the last point
        result.append(current_point.strip())
    
    return result

def extract_descriptions(problem):
    ''' Extract all descriptions from a full problem description.

    Args:
        problem (dict): A dictionary containing the problem descriptions.

    Returns:
        tuple: A tuple containing the extracted descriptions.
    '''

    problem_description = problem['problem_description']
    instance_description = problem['instance_description']
    generator_description = problem['generator_description']
    hard_constraint_descriptions = extract_bullet_points(problem['hard_constraint_descriptions'])
    soft_constraint_descriptions = extract_bullet_points(problem['soft_constraint_descriptions'])
   
    return problem_description, instance_description, generator_description, hard_constraint_descriptions, soft_constraint_descriptions


def check_and_repair_statement_blocks(generated_code, prompt, syntax_corrector_bot, k, generation_type, printer=False):
    '''Check syntax for each statement block and attempt to repair using the provided syntax_corrector_bot.

    Args:
        generated_code (str): The generated ASP code as a single string.
        prompt (str): The original user prompt describing intended semantics (used in repair prompts).
        syntax_corrector_bot (object): Bot to use for repair.
        k (int): Number of retries per statement block.
        generation_type (str): Type of generation for logging purposes.
        printer (bool): Whether to print debug information.

    Returns:
        tuple: (updated_statement_blocks, total_errors)
    '''

    # Split the response into separate statement blocks, so each can be syntax checked individually
    statement_blocks = utils.split_ASP_code_into_statement_blocks([generated_code])

    total_errors = 0

    for idx, stmt in enumerate(statement_blocks):
        # Check heuristically if the block is not written text (e.g., comments, explanations, LLM mumbo jumbo or empty lines)
        # This is crucial to prevent a lot of LLM repair calls on non-ASP text (wasting tokens and time, as learned the hard way).
        if not utils.check_if_ASP(stmt):
            statement_blocks[idx] = ""  # Replace non-ASP blocks with empty string to avoid syntax errors
            continue

        syntax_error = utils.check_syntax_of_one_string(stmt)
        retries = k  # Number of syntax repair retries left

        if syntax_error:
            # Try to repair the syntax k times
            if printer:
                print("================================================================================")
                print(f'Initial response with syntax error:\n{stmt}\n\nError: {syntax_error}\n')
                print(f" Starting syntax repair attempts...")
                print("================================================================================")

            while retries > 0 and syntax_error and syntax_corrector_bot is not None:
                retries -= 1

                # Create a prompt for repairing the syntax
                repair_prompt = f"Intended semantics:\n{prompt}\n\nErroneous ASP code:\n{stmt}\n\nClingo error message:\n{syntax_error}"
                stmt = syntax_corrector_bot.prompt(repair_prompt)

                if printer:
                    print("--------------------------------------------------------------------------------")
                    print(f'Correction attempt {k - retries}:\n{stmt}\n')

                # Failsafe to correct the bot if it returned multiple statements
                while len(utils.split_ASP_code_into_statement_blocks(stmt)) > 1 and retries > 0:
                    retries -= 1
                    repair_prompt = "The previous response contained multiple statements, which is not allowed. Please provide only one corrected ASP code without any extra explanations."
                    stmt = syntax_corrector_bot.prompt(repair_prompt)

                    if printer:
                        print("--------------------------------------------------------------------------------")
                        print(f'Multiple statement blocks returned by LLM - Correction attempt {k - retries}:\n{stmt}\n')

                # Check the syntax again
                syntax_error = utils.check_syntax_of_one_string(stmt)

                if printer:
                    print("--------------------------------------------------------------------------------")
                    if syntax_error:
                        print(f'Syntax error still present: {syntax_error}\n')
                    else:
                        print(f'Syntax corrected successfully!\n')

        # Replace the original statement with the (possibly) corrected one
        statement_blocks[idx] = stmt

        # Collect metrics (per statement block)
        fix_success = not syntax_error
        if not fix_success:
            total_errors += 1
        # attempts_made = k - retries  # kept locally if later needed

        # Log metrics to logfile, one log line for each statement block.
        # ONLY log if the block is a program statement (not a comment or empty) - for correct metrics.
        if utils.check_if_block_is_program_statement(stmt):
                        # generation_type is now required by the logger API and is provided by the caller
            logger.log(generation_type, fix_attempt_count_syntax=k - retries, correct_syntax=fix_success)

    repaired_code = '\n'.join(statement_blocks)

    return repaired_code, total_errors

def check_semantics(generated_code, prompt, semantics_bot, printer=False):
    ''' Check the semantics of the generated code against the prompt using the semantics_bot. Assumes that the code is syntactically correct.

    Args:
        generated_code (str): The generated ASP code as a single string.
        prompt (str): The original user prompt describing intended semantics.
        semantics_bot (object): Bot to use for semantic extraction.
        printer (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        tuple: (semantics_correct (bool), reason if incorrect (str))
    '''
    
    # First extract the semantics from the generated code
    semantics_prompt = f"Generated ASP code:\n{generated_code}\n\nIntended semantics:{prompt}"
    result = semantics_bot.prompt(semantics_prompt)

    if printer:
        print("================================================================================")
        print(f'Send prompt for semantic check:\n{semantics_prompt}\n')
        print(f'Semantic check result:\n{result}\n')
        print("================================================================================")

    # Deserialize the result (as it is expected to be JSON)
    try:
        semantics_correct_response = json.loads(result)
    except json.JSONDecodeError:
        if printer:
            print("================================================================================")
            print(f'Failed to parse JSON from semantic check result. Returning False.\n')
            print("================================================================================")
        return False, "Failed to parse JSON from semantic check result."
    
    # Covert 'match' to boolean
    match = True if semantics_correct_response.get('match', "false").lower() == "true" else False
    
    return match, semantics_correct_response.get('reason', '')

def repair_semantics(generated_code, prompt, semantics_bot, repair_bot, n, printer=False):
    ''' Attempt to repair the semantics of the generated code using the semantics_bot.

    Args:
        generated_code (str): The generated ASP code as a single string.
        prompt (str): The original user prompt describing intended semantics.
        semantics_bot (object): Bot to use for semantic extraction and validation.
        repair_bot (object): Bot to use for repairing the code.
        n (int): Number of repair attempts.
        printer (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        tuple: (repaired_code (str), semantics_correct (bool), syntax_correct (bool), retries_left (int))
    '''
    retries = n
    repaired_code = generated_code

    syntax_correct = True  # Assume syntax is correct when entering semantic repair

    while retries > 0:
        retries -= 1

        # Check semantics of the current code
        semantics_correct, reason = check_semantics(
            generated_code=repaired_code,
            prompt=prompt,
            semantics_bot=semantics_bot,
            printer=printer
        )

        # Exit if semantics are correct
        if semantics_correct:
            if printer:
                print("================================================================================")
                print("Semantics are now correct according to LLM. Exiting semantic repair loop.\n")
                print("================================================================================")
            break  # Exit the loop if semantics are correct
        
        # Semantics are incorrect, print the current code and index of the attempt
        if printer:
            print("--------------------------------------------------------------------------------")
            print(f'Semantic repair attempt {n - retries}/{n}:\n{repaired_code}\n')
            print("--------------------------------------------------------------------------------")

        # Repair the code using the repair bot
        repair_prompt = f"The ASP code below does not match the intended semantics.\n\nIntended semantics:\n{prompt}\n\nGenerated ASP code:\n{repaired_code}\n\nReason for mismatch:\n{reason}\n\nPlease provide a corrected version of the ASP code that matches the intended semantics."
        repaired_code = repair_bot.prompt(repair_prompt)

        # Check syntax of the repaired code
        syntax_error = utils.check_syntax_of_one_string(repaired_code)
        if syntax_error:
            syntax_correct = False
            if printer:
                print("================================================================================")
                print(f'Syntax error found in repaired code:\n{syntax_error}\n. Exiting semantic repair loop.\n')
                print("================================================================================")
            break  # Exit the loop if syntax error is found after repair as we need to go back to syntax repair first

    return repaired_code, semantics_correct, syntax_correct, retries

def get_partial_program(system_prompt_path, prompt, system_prompt_variables={}, pipe=None, semantic_validation_pipe=None, k=0, n=0, printer=False, temperature=None, top_p=None, seed=None, max_new_tokens=512):
    ''' Generate a partial ASP program based on a system prompt and variables.

    Args:
        system_prompt_path (str): The path to the system prompt file.
        prompt (str): The user prompt to send to the LLM for the specific partial program.
        system_prompt_variables (dict): A dictionary containing variables to replace in the system prompt.
        pipe (optional): The pipeline to use for the LLM. Defaults to None.
        semantic_validation_pipe (optional): The pipeline to use for semantic validation. Defaults to None.
        k (int, optional): The number of retries to get a syntactically correct response. Defaults to 0 (no retries).
        n (int, optional): The number of retries to get a semantically valid response. Defaults to 0 (no retries). Semantic repair takes place after syntax repair.
        printer (bool, optional): Whether to print intermediate results. Defaults to False.
        temperature (float, optional): The temperature setting for the LLM. Defaults to None.
        top_p (float, optional): The top_p setting for the LLM. Defaults to None.
        seed (int, optional): The seed for the LLM. Defaults to None.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.

    Returns:
        str: The generated partial ASP program as a string.
    '''
    # Total errors for metrics
    total_errors = 0

    # Copy initial n for logging later
    initial_n = n

    # Read the system prompt and replace variables
    system_prompt = read_system_prompt(system_prompt_path)
    
    # Replace variables in the system prompt
    for key, value in system_prompt_variables.items():
        system_prompt = system_prompt.replace(f'<<{key}>>', value)

    # Load the bot and get the response
    asp_generator_bot = bots.load_bot(system_prompt, pipe, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=seed)
    generated_code = asp_generator_bot.prompt(prompt)
    generated_code = utils.remove_backtick_lines(generated_code)    
    generated_code = utils.sanitize_asp_code(generated_code)
    initial_generated_code = generated_code  # Keep a copy of the initial response for printing later

    # Create a repair prompt if k > 0
    if k > 0:
        repair_prompt = read_system_prompt('system_prompts/syntax_corrector.txt')

        # Replace variables in the system prompt
        for key, value in system_prompt_variables.items():
            repair_prompt = repair_prompt.replace(f'<<{key}>>', value)

        # Replace remaining variables with None using regex
        repair_prompt = re.sub(r'<<[^<>]*>>', 'None', repair_prompt)

        # Create a new bot for repairing the syntax
        syntax_corrector_bot = bots.load_bot(repair_prompt, pipe, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=seed)

    # We always need a semantics bot for semantic checking even if n=0 (since we need to know if semantics are correct for logging)
    semantics_prompt = read_system_prompt('system_prompts/semantics.txt')

    # Replace variables in the system prompt
    for key, value in system_prompt_variables.items():
        semantics_prompt = semantics_prompt.replace(f'<<{key}>>', value)
    
    # Replace remaining variables with None using regex
    semantics_prompt = re.sub(r'<<[^<>]*>>', 'None', semantics_prompt)

    # Create a new bot for extracting semantics
    semantics_bot = bots.load_bot(semantics_prompt, semantic_validation_pipe, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=seed)
    
    # region Determine generation_type for logging based on the system prompt file used. A bit hacky but works for now.
    if 'instance' in system_prompt_path:
        gen_type = 'instance'
    elif 'generator' in system_prompt_path:
        gen_type = 'generator'
    elif 'hard_constraints' in system_prompt_path or 'hard constraints' in system_prompt_path:
        gen_type = 'hard constraints'
    elif 'soft_constraints' in system_prompt_path or 'soft constraints' in system_prompt_path:
        gen_type = 'soft constraints'
    else:
        gen_type = 'unknown'
    # endregion

    continue_semantics_loop = True # We use this variable to simulate a do-while loop as Python does not have native do-while loops
    while continue_semantics_loop:
        # STEP 1 CHECK SYNTAX AND REPAIR IF NEEDED AND K > 0
        # Check syntax of each statement block individually and attempt to fix error
        if k > 0:
            # Use the previously created syntax_corrector_bot to repair statement blocks
            generated_code, total_errors = check_and_repair_statement_blocks(
                generated_code=generated_code,
                prompt=prompt,
                syntax_corrector_bot=syntax_corrector_bot,
                k=k,
                generation_type=gen_type,
                printer=printer
            )

            syntax_correct = (total_errors == 0)
        else:
            # When no repairs are requested, still run a basic syntax check to count errors
            syntax_error = utils.check_syntax_of_one_string(generated_code)
            if syntax_error:
                total_errors += 1
                syntax_correct = False
            else:
                syntax_correct = True

        # If there are still syntax errors, we cannot proceed to semantic checking
        if total_errors > 0:
            if printer:
                print("================================================================================")
                print(f'Syntax errors remain after repair attempts. Cannot proceed to semantic checking.\nTotal syntax errors: {total_errors}\n')
                print("================================================================================")
            semantics_correct = False # Set semantics_correct to False as we cannot check semantics if syntax is incorrect but we still need it for logging later
            logger.log(f"{gen_type} semantics", correct_semantics_val=-2)  # Log the failed syntax check
            break  # Exit the while loop if syntax errors remain. The semantics cannot be checked if syntax is incorrect.

        # STEP 2 CHECK SEMANTICS AND REPAIR IF N > 0
        semantics_correct, _ = check_semantics(
            generated_code=generated_code,
            prompt=prompt,
            semantics_bot=semantics_bot,
            printer=printer
        )

        if not semantics_correct and n > 0:
            if printer:
                print("================================================================================")
                print(f'Semantics are incorrect. Starting semantic repair attempts (n={n} left)...')
                print("================================================================================")
            generated_code, semantics_correct, syntax_correct, n = repair_semantics(
                generated_code=generated_code,
                prompt=prompt,
                semantics_bot=semantics_bot,
                repair_bot=asp_generator_bot,
                n=n,
                printer=printer
            )

            # If syntax became incorrect during semantic repair, we need to go back to syntax repair
            if not syntax_correct:
                total_errors += 1  # Count as a syntax error for metrics

                if printer:
                    print("================================================================================")
                    print(f'Syntax error found after semantic repair. Returning to syntax repair loop. Semantic repair has {n} attempts left.\n')
                    print("================================================================================")

        logger.log(f"{gen_type} semantics", fix_attempt_count_semantics=initial_n - n, correct_semantics_val=int(semantics_correct), correct_syntax=syntax_correct)  # Log semantics correctness after repair

        if semantics_correct or n == 0:
            continue_semantics_loop = False  # Exit the while loop if semantics are correct or no repair attempts are left

    if printer:
        if(generated_code != initial_generated_code):
            print("####################################################################################")
            print(f"PROGRAM PART WAS REPAIRED {', BUT UN' if total_errors > 0 else ''}SUCCESSFULLY")
            print(f'INITIAL RESPONSE:\n{initial_generated_code}\n\nREPAIRED RESPONSE:\n{generated_code}\n')
            print(f'Total syntax errors remaining after repair attempts: {total_errors}\n')
            print(f"Semantics correct (according to LLM): {semantics_correct}\n")
            print("####################################################################################")
        else:
            print("####################################################################################")
            print("PROGRAM PART DID NOT NEED REPAIR!")
            print(f'RESPONSE:\n{generated_code}\n')
            print("####################################################################################")

    return(generated_code)

def full_ASP_program(problem, printer=False, pipe=None, semantic_validation_pipe=None, k=0, n=0, temperature=None, top_p=None, seed=None, max_new_tokens=512):
    ''' Generate a full ASP program based on the problem description.

    Args:
        problem (dict): A dictionary containing the problem descriptions.
        printer (bool, optional): Whether to print intermediate results. Defaults to False.
        pipe (optional): The pipeline to use for the LLM. Defaults to None.
        semantic_validation_pipe (optional): The pipeline to use for semantic validation. Defaults to None.
        k (int, optional): The number of retries to get a syntactically correct response. Defaults to 0 (no retries).
        n (int, optional): The number of retries to get a semantically valid response. Defaults to 0 (no retries). Semantic repair takes place after syntax repair.
        temperature (float, optional): The temperature setting for the LLM. Defaults to None.
        top_p (float, optional): The top_p setting for the LLM. Defaults to None.
        seed (int, optional): The seed for the LLM. Defaults to None.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.

    Returns:
        str: The full ASP program as a string.
    '''

    problem_description, instance_description, generator_description, hard_constraint_descriptions, soft_constraint_descriptions = extract_descriptions(problem)

    # Generate an instance template based on instance description (we still provide problem description for the repair prompt)
    instance_template = get_partial_program(
        system_prompt_path='system_prompts/instance.txt',
        prompt=instance_description,
        system_prompt_variables={
            'problem_description': problem_description
        },
        pipe=pipe,
        semantic_validation_pipe=semantic_validation_pipe,
        k=k,
        n=n,
        printer=printer,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_new_tokens=max_new_tokens
    )
    print('Instance Template:\n' + instance_template) if printer else None
    
    # Generate a generator based on generator description and instance template (we still provide problem description for the repair prompt)
    generator = get_partial_program(
        system_prompt_path='system_prompts/generator.txt',
        prompt=generator_description,
        system_prompt_variables={
            'instance_template': instance_template,
            'problem_description': problem_description
        },
        pipe=pipe,
        semantic_validation_pipe=semantic_validation_pipe,
        k=k,
        n=n,
        printer=printer,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_new_tokens=max_new_tokens
    )
    print('\n\nGenerator\n' + generator) if printer else None

    # Generate hard constraints based on hard constraint descriptions, problem description, instance template and generator
    hard_constraints = get_hard_constraints(hard_constraint_descriptions, problem_description, instance_template, generator, pipe=pipe, semantic_validation_pipe=semantic_validation_pipe, printer=printer, k=k, n=n, temperature=temperature, top_p=top_p, seed=seed, max_new_tokens=max_new_tokens)

    # Generate soft constraints based on soft constraint descriptions, problem description, instance template and generator
    soft_constraints = get_soft_constraints(soft_constraint_descriptions, problem_description, instance_template, generator, pipe=pipe, semantic_validation_pipe=semantic_validation_pipe, printer=printer, k=k, n=n, temperature=temperature, top_p=top_p, seed=seed, max_new_tokens=max_new_tokens)

    # Create a string that contains all hard and soft constraints with descriptions as comments
    hard_constraints_str = extract_constraints(hard_constraint_descriptions, hard_constraints)
    soft_constraints_str = extract_constraints(soft_constraint_descriptions, soft_constraints)
    
    # Combine everything into a full ASP program
    full_program = f'''
{instance_template}

% Generator

{generator}\n

% Hard Constraints

{hard_constraints_str}

% Soft Constraints

{soft_constraints_str}

% Objective function
#minimize {{ Penalty,Reason,SoftConstraint : penalty(SoftConstraint,Reason,Penalty) }}.
'''

    return full_program