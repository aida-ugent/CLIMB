import pickle
import json
from pathlib import Path
from tqdm import tqdm
import openai
from google.genai import types
import re


def robust_json_parser(text: str):
    """
    Extracts a JSON object from a string that might be wrapped in markdown code fences
    or other text, and parses it.

    Args:
        text: The input string containing a JSON object.

    Returns:
        The parsed Python dictionary or list if successful, otherwise None.
    """
    # This regex finds a string that starts with '{' and ends with '}',
    # and captures everything in between. re.DOTALL makes '.' match newlines.
    match = re.search(r'\{.*\}', text, re.DOTALL)
    
    if not match:
        # If no JSON object is found, try to parse the whole string
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print("Error: The string does not contain a valid JSON object.")
            return None

    json_str = match.group(0)
    
    try:
        # Parse the extracted string
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        # The extracted string is not valid JSON
        return None


def annotate_occupations(prompts, save_path, client, model="gpt-4o-mini", temperature=0., max_tokens=4000, provider="openai"):
    """
    Annotate prompts using OpenAI API and save results to a pickle file.
    
    Args:
        prompts (dict): Dictionary of prompts with node_ids as keys
        save_path (str or Path): Path to save the annotation responses
        client (openai.OpenAI): OpenAI client instance
        model (str): OpenAI model to use
        temperature (float): Temperature parameter for generation
        max_tokens (int): Maximum tokens for completion
    
    Returns:
        dict: Dictionary containing all annotations
    """
    save_path = Path(save_path)
    
    # Load existing annotations if they exist
    if save_path.exists():
        with open(save_path, 'rb') as f:
            annotation_response = pickle.load(f)
    else:
        annotation_response = {}
    
    # Process prompts
    for node_id, values in tqdm(prompts.items(), total=len(prompts)):

        if node_id in annotation_response:
            continue
        
        annotation_response[node_id] = {}
        
        if provider == 'openai':
            response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": values['prompt']}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            )
            annotation_response[node_id]['annotation'] = response.choices[0].message.content
        elif provider == 'google':
            if '2.5' in model:
                response = client.models.generate_content(
                model=model,
                # system_instruction="", # TODO: add system instruction   
                contents=[values['prompt']],
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    # response_mime_type='application/json',
                    thinking_config=types.ThinkingConfig(thinking_budget=512)
                )
                )
            else:
                response = client.models.generate_content(
                model=model,
                # system_instruction="", # TODO: add system instruction   
                contents=[values['prompt']],
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    response_mime_type='application/json'# TODO: drop this when instruction is added?
                )
                )
            try:
                annotation_response[node_id]['annotation'] = robust_json_parser(response.text)
            except:
                print(node_id, response)
                annotation_response[node_id]['annotation'] = response.text

        annotation_response[node_id]['sampled_job_titles'] = values['sampled_job_titles']
        annotation_response[node_id]['sampled_job_idx'] = values['sampled_job_idx']
        
        # Save after each annotation
        with open(save_path, 'wb') as f:
            pickle.dump(annotation_response, f)
    
    return annotation_response