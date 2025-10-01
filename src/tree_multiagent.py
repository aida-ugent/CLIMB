import json
import pickle
import os
import re
from datetime import datetime
from google import genai
from google.genai import types

from dotenv import load_dotenv

load_dotenv()


import nltk # For lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet # For POS tagging



# 
# --- Configuration ---

MOCK_LLM_CALLS = False

# --- LLM Interaction (Mocked or Real) ---
def call_llm(prompt, client=None, model_name="gemini-2.0-flash", temperature=0.2):
    if MOCK_LLM_CALLS:
        print(f"\n--- MOCK LLM Call ---")
        print(f"Prompt (first 500 chars):\n{prompt[:500]}...")

        # Default mock JSON response: an empty list
        mock_response_content_str = "[]"

        if "PREVIOUS ATTEMPT ISSUES" in prompt and "Missing child items" in prompt:
            print("Simulating Generator LLM response *with correction* after feedback...")
            # Assuming the small example input of 6 items
            # Corrected L1 for the 6-item example
            mock_response_content_str = json.dumps([
                {
                    "title": "Specialized Registered Nurses",
                    "description": "Registered nurses with advanced specialization in specific patient populations or critical care environments, requiring specialized skills and knowledge.",
                    "kids": [
                        {
                            "title": "Pediatric Registered Nurse",
                            "description": "A specialized registered nurse focused on providing medical care to infants, children, and adolescents. Responsibilities include patient assessment, administering treatments, and educating families on child health and disease prevention. Works in various settings like hospitals, clinics, and schools."
                        },
                        {
                            "title": "Critical Care Registered Nurse (ICU)",
                            "description": "A registered nurse who provides intensive care to critically ill or unstable patients in intensive care units (ICUs). Duties involve close monitoring, advanced life support, and managing complex medical equipment. Requires quick decision-making and specialized skills."
                        },
                        {
                            "title": "Neonatal Intensive Care Unit (NICU) Nurse",
                            "description": "Provides highly specialized nursing care to premature and critically ill newborn infants in a Neonatal Intensive Care Unit. Involves managing ventilators, incubators, and complex feeding and medication regimens."
                        }
                    ]
                },
                {
                    "title": "Web Developers",
                    "description": "Software developers focused on building and maintaining web applications, encompassing both frontend user interface design and backend server-side logic.",
                    "kids": [
                        {
                            "title": "Frontend Web Developer (React & Vue)",
                            "description": "A software developer specializing in creating user interfaces and user experiences for web applications using JavaScript frameworks like React and Vue.js. Focuses on visual design, interactivity, and ensuring responsive design across devices."
                        },
                        {
                            "title": "Backend Web Developer (Node.js & Python)",
                            "description": "A software developer responsible for server-side logic, database management, and API development for web applications. Works with technologies such as Node.js, Python (Django/Flask), and various database systems to ensure application functionality and performance."
                        }
                    ]
                },
                {
                    "title": "Clinical Research Professionals",
                    "description": "Professionals involved in the management and coordination of clinical trials and research studies.",
                    "kids": [
                         {
                            "title": "Clinical Research Coordinator",
                            "description": "Manages and coordinates clinical trial activities, ensuring compliance with protocols and regulatory requirements. Responsibilities include patient recruitment, data collection, and maintaining trial documentation. Liaises with investigators, sponsors, and ethics committees."
                        }
                    ]
                }
            ])
        elif "cluster the following list of occupation items" in prompt.lower(): # Generator prompt
            if "Pediatric Registered Nurse" in prompt: # Initial L1 generation for the 6-item example
                print("Simulating Generator LLM response for L1 generation (initial attempt - may have errors)...")
                # Flawed response for the 6-item example
                mock_response_content_str = json.dumps([
                    {
                        "title": "Nursing Professionals", # Good parent
                        "description": "Nurses providing care in various settings.",
                        "kids": [
                            { # Original child item object
                                "title": "Pediatric Registered Nurse",
                                "description": "A specialized registered nurse focused on providing medical care to infants, children, and adolescents. Responsibilities include patient assessment, administering treatments, and educating families on child health and disease prevention. Works in various settings like hospitals, clinics, and schools."
                            },
                            { # Original child item object
                                "title": "Critical Care Registered Nurse (ICU)",
                                "description": "A registered nurse who provides intensive care to critically ill or unstable patients in intensive care units (ICUs). Duties involve close monitoring, advanced life support, and managing complex medical equipment. Requires quick decision-making and specialized skills."
                            }
                            # Missing "Neonatal Intensive Care Unit (NICU) Nurse"
                        ]
                    },
                    {
                        "title": "Software Developers", # Good parent
                        "description": "Developers creating software applications.",
                        "kids": [
                             { # Original child item object
                                "title": "Frontend Web Developer (React & Vue)",
                                "description": "A software developer specializing in creating user interfaces and user experiences for web applications using JavaScript frameworks like React and Vue.js. Focuses on visual design, interactivity, and ensuring responsive design across devices."
                            },
                            # Missing "Backend Web Developer (Node.js & Python)"
                            {
                                "title": "Bogus Developer", # Extra, non-original child
                                "description": "A developer that shouldn't be here."
                            }
                        ]
                    }
                    # Missing "Clinical Research Coordinator" and its parent
                ])
            elif "Specialized Registered Nurses" in prompt: # L2 generation based on corrected L1
                print("Simulating Generator LLM response for L2 generation...")
                mock_response_content_str = json.dumps([
                    {
                        "title": "Healthcare Professionals",
                        "description": "A broad category encompassing various roles in medical care and research.",
                        "kids": [
                            {
                                "title": "Specialized Registered Nurses",
                                "description": "Registered nurses with advanced specialization in specific patient populations or critical care environments, requiring specialized skills and knowledge."
                            },
                            {
                                "title": "Clinical Research Professionals",
                                "description": "Professionals involved in the management and coordination of clinical trials and research studies."
                            }
                        ]
                    },
                    {
                        "title": "Technology Professionals",
                        "description": "Experts in software and web development.",
                        "kids": [
                            {
                                "title": "Web Developers",
                                "description": "Software developers focused on building and maintaining web applications, encompassing both frontend user interface design and backend server-side logic."
                            }
                        ]
                    }
                ])
            else:
                print("Unmatched clustering prompt in mock for new format. Returning empty list.")
                mock_response_content_str = "[]"
        else:
            print("Unrecognized prompt type in mock. Returning non-JSON error string.")
            return "Error: Unrecognized prompt for JSON list generation."

        return mock_response_content_str

    # Real LLM call
    else:
        try:
            response = client.models.generate_content(
            model=model_name,
            # system_instruction="", # TODO: add system instruction   
            contents=[prompt],
            config=types.GenerateContentConfig(
                # max_output_tokens=max_tokens,
                temperature=temperature,
                # response_mime_type='application/json'
            )
            )

            response_content = response.text
            json.loads(response_content) # Validate
            return response_content
        except json.JSONDecodeError as e:
            print(f"LLM response was not valid JSON: {e}")
            print(f"LLM raw response: {response_content}") # This was response_str before, fixed
            return response_content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            raise

# --- Generator Agent ---
class GeneratorAgent:
    def __init__(self, client, llm_model="gemini-2.5-pro-preview-05-06", output_dir="."):
        self.client = client
        self.llm_model = llm_model
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def _extract_and_clean_json_list_string(self, text_response):
        """
        Attempts to extract a JSON list string from a larger text response.
        Handles common markdown code fences.
        """
        if not text_response or not isinstance(text_response, str):
            return None
        text_response = text_response.strip()

        # Try to find JSON within markdown code blocks first
        patterns = [
            r"```json\s*(\[.*?\])\s*```",  # ```json [...] ```
            r"```\s*(\[.*?\])\s*```"      # ``` [...] ```
        ]
        for pattern in patterns:
            match = re.search(pattern, text_response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no code blocks, try to find the first '[' and last ']' heuristically
        # This is a fallback and might not always be correct if there's nested JSON outside the main list
        first_bracket_idx = text_response.find('[')
        last_bracket_idx = text_response.rfind(']')

        if first_bracket_idx != -1 and last_bracket_idx != -1 and first_bracket_idx < last_bracket_idx:
            potential_json = text_response[first_bracket_idx : last_bracket_idx + 1]
            # A simple check: if it seems to contain objects, it might be our list
            if potential_json.count('{') > 0 and potential_json.count('}') > 0:
                # Try to parse this substring to see if it's valid in isolation
                try:
                    json.loads(potential_json)
                    return potential_json.strip()
                except json.JSONDecodeError:
                    pass # If this substring isn't valid, fall through

        # If the original string itself looks like a list
        if text_response.startswith('[') and text_response.endswith(']'):
            return text_response

        return None # Could not reliably extract a JSON list string

    def _attempt_local_json_fixes(self, malformed_json_string):
        """
        Attempts common local fixes for malformed JSON strings.
        Returns a potentially fixed string or None if unfixable.
        This is a simplified fixer; more advanced libraries could be used.
        """
        if not malformed_json_string:
            return None
        
        s = malformed_json_string
        
        # 1. Try to fix trailing commas in objects/arrays (simple version)
        #    e.g. {"a":1,} -> {"a":1} or [1,2,] -> [1,2]
        s = re.sub(r',\s*([\}\]])', r'\1', s)

        # 2. Try to add missing commas between object elements (if easily detectable)
        #    e.g. {"a":1 "b":2} -> {"a":1, "b":2} (this is hard to do reliably with regex)
        #    A common error is "}"\s*" (missing comma between objects in a list)
        s = re.sub(r'\}\s*\"', r'},"', s) # after object, before next property in another object (often wrong context)
        s = re.sub(r'\}\s*\{', r'},{', s) # Missing comma between objects in a list
        s = re.sub(r'\]\s*\[', r'],[', s) # Missing comma between arrays in a list
        s = re.sub(r'\"\s*\"', r'","', s) # Missing comma between string values

        # 3. Ensure strings are properly quoted (very basic attempt)
        # This is complex; a full linter/fixer is needed for robust string quoting.
        # Example: 'key': "value" -> "key": "value" (Python dict style single quotes)
        # s = s.replace("'", '"') # Too broad, might break correctly single-quoted strings within double-quoted strings

        try:
            json.loads(s)
            print("Local JSON fixer: Applied some fixes, now parsable.")
            return s # It's parsable now
        except json.JSONDecodeError:
            # print("Local JSON fixer: Could not fix with simple rules.")
            return malformed_json_string # Return original if simple fixes didn't work, for LLM to retry


    def generate_parent_level(self, 
                                child_items_list_of_dicts,  # 1 (excluding self)
                                current_level_name,         # 2
                                min_parents,                # 3
                                max_parents,                # 4
                                previous_attempt_issues=None, # 5 (default)
                                previous_flawed_taxonomy_str=None, # 6 (default)
                                is_retry_for_json_syntax=False): # 7 (default)
        
        child_items_prompt_str = "[\n"
        for item in child_items_list_of_dicts:
            child_items_prompt_str += f"  {json.dumps(item, indent=2)},\n"
        child_items_prompt_str = child_items_prompt_str.rstrip(",\n") + "\n]"

        main_task_instruction = f"Your task is to cluster these {len(child_items_list_of_dicts)} items into broader parent categories."
        feedback_prompt_segment = ""

        if previous_attempt_issues:
            if is_retry_for_json_syntax:
                # ... (JSON syntax feedback as before) ...
                feedback_prompt_segment = f"""
                IMPORTANT CORRECTION FOR JSON SYNTAX:
                Your previous response was not valid JSON.
                Please ensure your entire response is ONLY a valid JSON list of objects,
                starting with '[' and ending with ']'. Do not include any other text or markdown.
                The required structure for each parent object is:
                {{ "title": "string", "description": "string", "kids": ["child_title_1", "child_title_2", ...] }}
                Adhere strictly to this JSON format.
                """
            elif previous_flawed_taxonomy_str: # Semantic errors AND we have the previous flawed output
                main_task_instruction = "YOUR TASK IS TO REVISE THE PREVIOUS OUTPUT (shown below) to correct the specified issues. Only make necessary changes."
                flawed_output_segment = f"""
                YOUR PREVIOUS (FLAWED) OUTPUT TO REVISE:
                ```json
                {previous_flawed_taxonomy_str}
                ```
                """
                issues_str = "\n".join([f"- {issue}" for issue in previous_attempt_issues])
                feedback_prompt_segment = f"""
                {flawed_output_segment}
                This previous output had the following semantic issues:
                --- PREVIOUS ATTEMPT ISSUES ---
                {issues_str}
                --- END OF PREVIOUS ATTEMPT ISSUES ---

                INSTRUCTIONS FOR REVISION:
                1.  **Review your previous output (above) and the listed issues carefully.**
                2.  **Modify your previous output to address EACH issue.**
                    -   For **Missing Items**: Integrate the missing child titles into appropriate 'kids' arrays. You might need to adjust existing parent categories or create a new one if a missing item doesn't fit well.
                    -   For **Extra Items**: REMOVE these specific titles from any 'kids' arrays.
                    -   For **Duplicate Items**: Ensure each duplicated title appears in the 'kids' array of only ONE parent. Decide which parent is the most suitable and remove it from others.
                    -   For **Parent-Child Title Collision**: If a parent's title is the same as one of its children's, you MUST change the parent's title to be a more general, distinct category name.
                3.  **Maintain Correct Parts:** Try to keep parent categories and their assignments that were NOT part of the issues, unless resolving an issue requires re-clustering some items.
                4.  **Ensure all original child items are assigned once.**
                5.  **The "kids" array should contain only the *titles* (strings) of the original child items.**
                6.  Your entire response MUST BE ONLY the revised JSON list, starting with '[' and ending with ']'.
                """
            else: # Semantic issues but no previous_flawed_taxonomy_str (should be rare if logic is correct)
                # ... (fallback actionable feedback as in the previous version) ...
                missing_items_feedback, extra_items_feedback, duplicate_items_feedback = "", "", ""
                parent_collision_feedback = ""
                parent_child_collision_feedback = ""

                for issue in previous_attempt_issues:
                    if "Missing child item titles" in issue: missing_items_feedback = f"  - Missing Items instructions...\n" # (Full instructions from prev. version)
                    elif "Extra child item titles" in issue: extra_items_feedback = f"  - Extra Items instructions...\n"
                    elif "Duplicate child item titles" in issue: duplicate_items_feedback = f"  - Duplicate Items instructions...\n"
                    elif "LLM generated multiple parent titles that normalize to the same form" in issue:
                        # Extract conflicting titles: "The conflicting titles were: ['Title Cased', 'title cased']"
                        match = re.search(r"conflicting titles were: (\[.*?\])", issue)
                        if match:
                            colliding_parent_titles_str = match.group(1)
                            parent_collision_feedback = (
                                f"  - **Parent Title Collision:** You generated multiple parent titles that are too similar "
                                f"(e.g., differ only by case or minor variations) and effectively represent the same category. "
                                f"The conflicting titles were: {colliding_parent_titles_str}. "
                                f"Each parent category you define must have a meaningfully distinct title. "
                                f"Please revise these to ensure uniqueness OR merge them if they truly represent one concept.\n"
                            )
                    elif "A parent cannot be its own child" in issue:
                        parent_child_collision_feedback = (
                            f"  - **Parent-Child Title Collision:** You created a parent category with a title that is the same as one of its children. "
                            f"A parent cannot have the same title as an item within it. The issue was: '{issue}'. "
                            f"Please change the parent's title to a more general, distinct category name.\n"
                        )
                if missing_items_feedback or extra_items_feedback or duplicate_items_feedback or parent_collision_feedback or parent_child_collision_feedback:
                    feedback_prompt_segment = f"IMPORTANT CORRECTIONS...\n{missing_items_feedback}{extra_items_feedback}{duplicate_items_feedback}{parent_collision_feedback}{parent_child_collision_feedback}"


        prompt = f"""
        You are an expert in creating hierarchical taxonomies for occupations.
        ...
        Given the following list of '{current_level_name}' occupation items each with a title and a description (these titles are canonical representations, each with a unique normalized form):
        --- START OF {current_level_name} CANONICAL ITEMS ---
        {child_items_prompt_str} 
        --- END OF {current_level_name} CANONICAL ITEMS ---

        {main_task_instruction} 

        For each parent category you create (or revise), you must provide:
        1. A "title" (string) for the parent category.
        2. A "description" (string) for the parent category.
        3. A "kids" field, which must be a JSON array of STRINGS. Each string in this array must be one of the *exact canonical 'title's* from the input list of items you received.

        Aim for a total number of parent categories between {min_parents} and {max_parents}.

        {feedback_prompt_segment}

        Output Format Constraints:
        Your *entire response* MUST be a single, valid JSON list of parent objects,
        beginning *directly* with `[` and ending *directly* with `]`.
        No explanations, comments, or Markdown.

        Parent object structure:
        {{
        "title": "string",
        "description": "string",
        "kids": [ "child_title_1", "child_title_2", /* ... */ ]
        }}

        Key Constraints:
        - EVERY original child item's 'title' MUST be in exactly ONE parent's "kids" array.
        - Use only original child item titles.
        - Parent titles must be distinct.
        - A parent's title CANNOT be the same as any of its child's titles.

        Now, generate (or revise) the JSON list of parent objects.
        """
        if previous_attempt_issues:
            print(f"GeneratorAgent: Re-generating parents for {len(child_items_list_of_dicts)} '{current_level_name}' items {'FOR JSON SYNTAX' if is_retry_for_json_syntax else 'WITH SEMANTIC FEEDBACK'}.")
        else:
            print(f"GeneratorAgent: Generating parents for {len(child_items_list_of_dicts)} '{current_level_name}' items.")

        
        raw_llm_response_str = call_llm(prompt, client=self.client, model_name=self.llm_model, temperature=0.05 if is_retry_for_json_syntax else (1. if not previous_attempt_issues else 0)) # Lower temp for correction

        if raw_llm_response_str is None:
            print("GeneratorAgent: LLM response was None. Returning None.")
            return None, "api_error"
        
        json_list_str = self._extract_and_clean_json_list_string(raw_llm_response_str)
        if json_list_str is None:
            print("GeneratorAgent: Failed to extract and clean JSON list string from LLM response. Returning None.")
            print(f"GeneratorAgent: Raw LLM response: {raw_llm_response_str[:500]}")
            # save the raw response to a file
            # add timestamp to the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_filename = f"error_raw_llm_response_{timestamp}.txt"
            output_path = os.path.join(self.output_dir, error_filename) if self.output_dir else error_filename
            with open(output_path, "w") as f:
                f.write(raw_llm_response_str)
            return None, "extract_error"
        
        try:
            proposed_taxonomy_list = json.loads(json_list_str)
            if not isinstance(proposed_taxonomy_list, list):
                print("GeneratorAgent: LLM response was valid JSON but not a list as expected. Type: {type(proposed_taxonomy_list)}")
                return None, "not a list"
            return proposed_taxonomy_list, json_list_str # Success after local fix
        
        except json.JSONDecodeError as e:
            print(f"GeneratorAgent: Failed to decode JSON list string initially: {e}")
            print(f"Attempting to fix locally: {json_list_str[:500]}")

            # 3. Attempt local fixes if initial parse fails
            fixed_json_str = self._attempt_local_json_fixes(json_list_str)
            if fixed_json_str != json_list_str: # If fixes were applied (or attempted)
                try:
                    proposed_taxonomy_list = json.loads(fixed_json_str)
                    if not isinstance(proposed_taxonomy_list, list):
                        print(f"GeneratorAgent: Locally fixed content was JSON but not a list.")
                        return None, "not_a_list_after_fix"
                    print("GeneratorAgent: Successfully parsed JSON after local fixes.")
                    return proposed_taxonomy_list, fixed_json_str # Success after local fix
                except json.JSONDecodeError as e_after_fix:
                    print(f"GeneratorAgent: Failed to decode JSON even after local fixes: {e_after_fix}")
                    print(f"String after local fix attempts: {fixed_json_str[:500]}")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # save the fixed string to a file
                    error_filename = f"error_fixed_json_str_{timestamp}.txt"
                    output_path = os.path.join(self.output_dir, error_filename) if self.output_dir else error_filename
                    with open(output_path, "w") as f:
                        f.write(fixed_json_str)
                    return None, "json_decode_error_after_fix" # Still a JSON error
            else: # No fixes applied or fixes didn't change the string, and it's still invalid
                return None, "json_decode_error_no_fix"


# --- Examiner Agent with Tolerant Matching ---
class ExaminerAgent:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.camel_case_splitter = re.compile(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')
        self.punctuation_remover = re.compile(r'[^\w\s/-]')  # Keeps words, spaces, hyphens, and slashes        
        self.multiple_space_reducer = re.compile(r'\s+')

    def _get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def normalize_title(self, title):
        if not title or not isinstance(title, str): return ""
        title_spaced = self.camel_case_splitter.sub(r' ', title)
        title_lower = title_spaced.lower()
        title_no_possessive = title_lower.replace("'s", "")
        title_no_punct = self.punctuation_remover.sub('', title_no_possessive)
        tokens = word_tokenize(title_no_punct)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token, self._get_wordnet_pos(token)) for token in tokens if token.strip()]
        normalized = ' '.join(lemmatized_tokens)
        normalized = self.multiple_space_reducer.sub(' ', normalized).strip()
        return normalized

    def examine_taxonomy(self, 
                         # This is the list of CANONICAL items LLM received as input for this level
                         canonical_input_items_for_llm_this_level, 
                         # This is the LLM's direct output (list of parent dicts)
                         llm_direct_output_taxonomy, 
                         min_parents_expected, max_parents_expected):
        issues = []
        if llm_direct_output_taxonomy is None:
            issues.append("Validation Error: Generator failed to produce a parsable taxonomy list.")
            return False, issues, None 

        # --- Part 1: Prepare sets based on CANONICAL input LLM received ---
        # These are the exact titles the LLM was *supposed* to use in its 'kids' arrays.
        # Normalization is used here to build a reference set for tolerant matching.
        canonical_input_titles_exact_set = {item['title'] for item in canonical_input_items_for_llm_this_level}
        # Map of normalized canonical input title -> exact canonical input title
        normalized_canonical_input_to_exact_map = {
            self.normalize_title(item['title']): item['title'] 
            for item in canonical_input_items_for_llm_this_level
        }
        normalized_canonical_input_set = set(normalized_canonical_input_to_exact_map.keys())


        if not isinstance(llm_direct_output_taxonomy, list):
            issues.append(f"Validation Error: Proposed taxonomy is not a list.")
            return False, issues, None

        # This will hold the final version of the taxonomy for this level,
        # with LLM's parent titles/desc, and kid titles mapped back to the
        # exact canonical form they matched from the input.
        validated_and_corrected_taxonomy = json.loads(json.dumps(llm_direct_output_taxonomy)) # Deep copy

        # --- Part 2: Collect and normalize what LLM *actually* put in 'kids' arrays ---
        # List of (normalized_llm_kid_title, exact_llm_kid_title_as_outputted)
        all_normalized_and_exact_llm_kids_tuples = [] 
        parent_titles_set = set()
        num_parents_generated = len(llm_direct_output_taxonomy)

        # --- NEW: Check for LLM-generated parent title collisions ---
        generated_parent_exact_titles_seen = set()
        generated_parent_normalized_titles_to_exact_map = {} # normalized_parent_title -> list of exact parent_titles from LLM that normalize to it

        num_parents_generated = len(llm_direct_output_taxonomy)
        # ... (parent count check) ...
        if not (min_parents_expected <= num_parents_generated <= max_parents_expected): # Example:
             if len(canonical_input_items_for_llm_this_level) > max_parents_expected and num_parents_generated > 0 :
                issues.append(f"Warning: Parent count ({num_parents_generated}) outside range.")
        if num_parents_generated == 0 and len(canonical_input_items_for_llm_this_level) > 0:
            issues.append("Validation Error: No parent categories generated.")


        for i, parent_obj_from_llm in enumerate(llm_direct_output_taxonomy):
            parent_obj_for_storage = validated_and_corrected_taxonomy[i]

            if not isinstance(parent_obj_from_llm, dict):
                issues.append(f"Validation Error: Parent item at index {i} is not a dictionary."); continue
            
            parent_title_llm = parent_obj_from_llm.get("title")
            normalized_parent_title_llm = "" # Defined here to ensure scope
            
            if not parent_title_llm or not isinstance(parent_title_llm, str) or not parent_title_llm.strip():
                issues.append(f"Validation Error: Parent at index {i} has missing, non-string, or empty title.")
                # Cannot proceed with collision checks for this parent if title is bad
            else:
                # Check for exact duplicate parent titles from LLM
                if parent_title_llm in generated_parent_exact_titles_seen:
                    issues.append(f"Validation Error: LLM generated duplicate exact parent title: '{parent_title_llm}'.")
                else:
                    generated_parent_exact_titles_seen.add(parent_title_llm)

                # Check for normalized duplicate parent titles from LLM
                normalized_parent_title_llm = self.normalize_title(parent_title_llm)
                if normalized_parent_title_llm not in generated_parent_normalized_titles_to_exact_map:
                    generated_parent_normalized_titles_to_exact_map[normalized_parent_title_llm] = []
                generated_parent_normalized_titles_to_exact_map[normalized_parent_title_llm].append(parent_title_llm)
            
            parent_description = parent_obj_from_llm.get("description")
            if not parent_description or not isinstance(parent_description, str) or not parent_description.strip():
                issues.append(f"Warning: Parent '{parent_title_llm or f'idx {i}'}' has invalid description.")

            kid_titles_list_from_llm = parent_obj_from_llm.get("kids", [])
            if not isinstance(kid_titles_list_from_llm, list):
                issues.append(f"Validation Error: 'kids' for parent '{parent_title_llm or f'idx {i}'}' is not a list."); continue
            
            # ... (empty kids list warning) ...
            if not kid_titles_list_from_llm and len(canonical_input_items_for_llm_this_level) > 0:
                 issues.append(f"Warning: Parent '{parent_title_llm or f'idx {i}'}' has an empty 'kids' list.")

            
            canonical_kids_for_this_parent_in_storage = [] # For the validated_and_corrected_taxonomy

            for kid_idx, exact_llm_kid_title in enumerate(kid_titles_list_from_llm):
                if not isinstance(exact_llm_kid_title, str) or not exact_llm_kid_title.strip():
                    issues.append(f"Validation Error: Kid at {kid_idx} for '{parent_title_llm}' not a non-empty string. Got: '{exact_llm_kid_title}'"); continue
                
                normalized_llm_kid_title = self.normalize_title(exact_llm_kid_title)

                # NEW: Check for parent-child title collision
                if normalized_parent_title_llm and normalized_llm_kid_title == normalized_parent_title_llm:
                    issues.append(
                        f"Validation Error: Parent title '{parent_title_llm}' is identical (or normalizes to be identical) "
                        f"to one of its child titles: '{exact_llm_kid_title}'. A parent cannot be its own child."
                    )
                
                all_normalized_and_exact_llm_kids_tuples.append((normalized_llm_kid_title, exact_llm_kid_title))

                # If this normalized LLM kid title matches a normalized *canonical input* title,
                # then for storage, we use the *exact canonical input title*.
                if normalized_llm_kid_title in normalized_canonical_input_to_exact_map:
                    matched_exact_canonical_input_title = normalized_canonical_input_to_exact_map[normalized_llm_kid_title]
                    canonical_kids_for_this_parent_in_storage.append(matched_exact_canonical_input_title)
                else:
                    # This kid title from LLM (even normalized) doesn't match any canonical input.
                    # It's an "extra". For the purpose of constructing the 'validated_and_corrected_taxonomy',
                    # we'll initially put what the LLM gave. The "extra" check later will flag it as an error.
                    canonical_kids_for_this_parent_in_storage.append(exact_llm_kid_title) 
            
            parent_obj_for_storage["kids"] = canonical_kids_for_this_parent_in_storage

        # After iterating all parents, check for normalized parent title collisions
        for norm_parent_title, exact_llm_parent_titles_list in generated_parent_normalized_titles_to_exact_map.items():
            if len(exact_llm_parent_titles_list) > 1:
                issues.append(f"Validation Error: LLM generated multiple parent titles that normalize to the same form ('{norm_parent_title}'). "
                              f"The conflicting titles were: {exact_llm_parent_titles_list}. Parent titles must be distinct even after normalization.")
        
        # --- Part 3: Validate assignments based on normalized titles ---
        normalized_llm_kid_titles_list = [t[0] for t in all_normalized_and_exact_llm_kids_tuples]
        normalized_llm_kid_titles_set = set(normalized_llm_kid_titles_list)

        # Duplicates: Check if any *normalized LLM kid title* appears more than once across all parent 'kids' lists
        if len(normalized_llm_kid_titles_list) != len(normalized_llm_kid_titles_set):
            temp_counts = {}
            for norm_title in normalized_llm_kid_titles_list:
                temp_counts[norm_title] = temp_counts.get(norm_title, 0) + 1
            
            # For feedback, report the *exact canonical input title* that was duplicated, if the duplicate matched one.
            # Or report the LLM's exact output if the duplicate was an "extra" item.
            duplicate_titles_for_feedback = set()
            for norm_title, count in temp_counts.items():
                if count > 1:
                    if norm_title in normalized_canonical_input_to_exact_map:
                        # This is a canonical input title that LLM duplicated
                        duplicate_titles_for_feedback.add(f"'{normalized_canonical_input_to_exact_map[norm_title]}' (which you received as input)")
                    else:
                        # This is an "extra" title (not in canonical input) that LLM duplicated
                        # Find one of the exact forms LLM used for this normalized duplicate
                        llm_exact_form_of_extra_duplicate = "unknown (extra item)"
                        for nt, et in all_normalized_and_exact_llm_kids_tuples:
                            if nt == norm_title:
                                llm_exact_form_of_extra_duplicate = et
                                break
                        duplicate_titles_for_feedback.add(f"'{llm_exact_form_of_extra_duplicate}' (which was an extra item you generated)")
            if duplicate_titles_for_feedback:
                issues.append(f"Validation Error: Duplicate child assignments. The following item titles (from your input or generated by you) were assigned to multiple parents: {duplicate_titles_for_feedback}")


        # Missing: Compare set of *normalized canonical input titles* with set of *normalized LLM kid titles*
        missing_normalized_canonical_titles = normalized_canonical_input_set - normalized_llm_kid_titles_set
        if missing_normalized_canonical_titles:
            # Report missing items using their *exact canonical input titles*
            missing_exact_canonical_titles_for_report = {
                normalized_canonical_input_to_exact_map[norm_title] 
                for norm_title in missing_normalized_canonical_titles
            }
            issues.append(f"Validation Error: Missing child item titles that were in your input: {missing_exact_canonical_titles_for_report}")

        # Extra: Compare set of *normalized LLM kid titles* with set of *normalized canonical input titles*
        extra_normalized_llm_titles = normalized_llm_kid_titles_set - normalized_canonical_input_set
        if extra_normalized_llm_titles:
            # Report extra items using the *LLM's exact output title*
            extra_exact_llm_titles_for_report = set()
            for norm_extra_title in extra_normalized_llm_titles:
                found_exact = False
                for nt, et in all_normalized_and_exact_llm_kids_tuples: # find an exact form LLM used for this normalized extra
                    if nt == norm_extra_title:
                        extra_exact_llm_titles_for_report.add(et)
                        found_exact = True
                        break
                if not found_exact: extra_exact_llm_titles_for_report.add(norm_extra_title + " (normalized form)")

            issues.append(f"Validation Error: Extra child item titles in your 'kids' lists (not in your input): {extra_exact_llm_titles_for_report}")

        hard_errors_present = any("Validation Error:" in issue for issue in issues)
        
        if not hard_errors_present:
            return True, issues, validated_and_corrected_taxonomy
        else:
            return False, issues, None
        

# --- Workflow Manager ---
class WorkflowManager:
    def __init__(self, initial_leaf_items_file_json, output_dir, client=None, llm_model='gemini-2.5-pro-preview-05-06'):
        self.leaf_items_file = initial_leaf_items_file_json
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.generator = GeneratorAgent(client, llm_model, self.output_dir)
        self.examiner = ExaminerAgent() # Make sure examiner has its own normalize_title
        self.full_taxonomy_by_level = {}
        self.normalized_to_originals_map_per_level = {} # Store this mapping for each level
        self.original_to_canonical_title_map_per_level = {}


    def _load_and_prepare_initial_items(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                original_items = json.load(f)
            # ... (validation of original_items structure) ...
            if not isinstance(original_items, list): return None # Basic check

            print(f"Loaded {len(original_items)} original items from {filepath}.")
            
            # Use the standalone normalize_title from Examiner for consistency
            # Or make _normalize_title_for_preprocessing a static method or top-level function
            self.examiner_normalizer = self.examiner.normalize_title 

            canonical_items_list, normalized_to_originals, orig_to_canon_map = \
                self._create_canonical_item_list_and_mapping(original_items, self.examiner_normalizer)
            
            self.normalized_to_originals_map_per_level[0] = normalized_to_originals
            self.original_to_canonical_title_map_per_level[0] = orig_to_canon_map
            return canonical_items_list # This list goes to the LLM for level 1
        except Exception as e:
            print(f"Error loading/preparing initial items: {e}")
            return None

    def _create_canonical_item_list_and_mapping(self, original_items_list_of_dicts, normalizer_func):
        normalized_to_originals_map = {} 
        canonical_items_list = []      
        normalized_seen_for_canonical = set()
        original_to_canonical_title_map = {} 

        print(f"Processing {len(original_items_list_of_dicts)} original items for canonicalization...")

        for item in original_items_list_of_dicts:
            original_title = item['title']
            normalized_title = normalizer_func(original_title)
            
            if normalized_title not in normalized_to_originals_map:
                normalized_to_originals_map[normalized_title] = {
                    "canonical_chosen_item": item, # First one becomes canonical by default
                    "all_originals": [item]
                }
                canonical_items_list.append(item)
                normalized_seen_for_canonical.add(normalized_title)
                original_to_canonical_title_map[original_title] = item['title']
            else:
                # It's a collision with an already seen normalized form. Add to its list of originals.
                normalized_to_originals_map[normalized_title]["all_originals"].append(item)
                # Map this original title to the already chosen canonical title for this norm_form
                original_to_canonical_title_map[original_title] = normalized_to_originals_map[normalized_title]["canonical_chosen_item"]['title']
                print(f"  Normalization Collision: '{original_title}' also normalizes to '{normalized_title}'. "
                      f"Canonical for this group is '{normalized_to_originals_map[normalized_title]['canonical_chosen_item']['title']}'.")
        
        print(f"Result: {len(canonical_items_list)} canonical items to be processed by LLM.")
        return canonical_items_list, normalized_to_originals_map, original_to_canonical_title_map

    def _load_items_from_json_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                items = json.load(f)
                if not isinstance(items, list):
                    print(f"Error: Content of {filepath} is not a JSON list.")
                    return None
                for item in items:
                    if not (isinstance(item, dict) and "title" in item and "description" in item):
                        print(f"Error: Invalid item structure in {filepath}. Expected {{'title': ..., 'description': ...}} Got: {item}")
                        return None
                return items
        except FileNotFoundError:
            print(f"Error: File not found - {filepath}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON from {filepath} - {e}")
            return None


    def run_taxonomy_generation(self, max_levels=5, initial_level_name="root_items", expand_kids_to_all_variants=True, retry_attempts=3, min_parents_to_continue_threshold=10): # New flag
        # current_items_for_llm are the CANONICAL items for the current level
        current_items_for_llm = self._load_and_prepare_initial_items(self.leaf_items_file)
        if current_items_for_llm is None: return None
        
        # Store the original full items for Level 0 display, using original titles as keys for their descriptions
        original_leaf_items_objects = self._load_items_from_json_file(self.leaf_items_file) # Load again for full list
        self.full_taxonomy_by_level[0] = {initial_level_name: original_leaf_items_objects}


        current_level_context_file_name = self.leaf_items_file # Initial context

        for level_num in range(1, max_levels + 1):
            print(f"\n--- Generating Level {level_num} (using {len(current_items_for_llm)} canonical items) ---")
            if not current_items_for_llm or len(current_items_for_llm) <= 1: break

            # ... (min_parents, max_parents calculation for current_items_for_llm) ...
            num_current_items = len(current_items_for_llm); # ... (min_p, max_p logic)
            min_parents = max(1, int(num_current_items * 0.2)); max_parents = max(min_parents + 1, int(num_current_items * 0.7))
            max_parents = min(max_parents, num_current_items -1 if num_current_items > 1 else 1); min_parents = min(min_parents, max_parents) if max_parents > 0 else 1
            print(f"Targeting {min_parents}-{max_parents} parents for {num_current_items} canonical items.")

            last_successfully_parsed_llm_output_str = None 
            current_issues_for_feedback = None
            retry_attempts = retry_attempts
            level_successfully_generated = False
            is_json_syntax_retry = False
            generated_parent_list_for_this_level = None

            for attempt in range(retry_attempts):
                # ... (retry loop setup as before) ...
                print(f"Attempt {attempt + 1} for Level {level_num}...")
                flawed_tax_for_prompt_str = None
                if not is_json_syntax_retry and current_issues_for_feedback and last_successfully_parsed_llm_output_str:
                    flawed_tax_for_prompt_str = last_successfully_parsed_llm_output_str
                
                # LLM receives canonical items for clustering
                llm_output_list, parsed_output_str_from_generator = self.generator.generate_parent_level(
                    current_items_for_llm,                 # Arg 1
                    f"Level {level_num-1} Items",          # Arg 2
                    min_parents, max_parents,              # Args 3, 4
                    current_issues_for_feedback,           # Arg 5
                    flawed_tax_for_prompt_str,             # Arg 6 <--- This is previous_flawed_taxonomy_str
                    is_json_syntax_retry                   # Arg 7 <--- This is is_retry_for_json_syntax
                )
                # ... (handling of llm_output_list, parsed_output_str_from_generator as before) ...
                is_json_syntax_retry = False 
                if parsed_output_str_from_generator: last_successfully_parsed_llm_output_str = parsed_output_str_from_generator
                else: last_successfully_parsed_llm_output_str = None
                if llm_output_list is None: 
                    print(f"GeneratorAgent failed this attempt."); current_issues_for_feedback = [f"Generator Error."]; is_json_syntax_retry = True 
                    if attempt == retry_attempts - 1: print("Max retries for generator failure."); return self.full_taxonomy_by_level
                    continue


                # Examiner validates LLM's output (which should contain canonical kid titles)
                # against the canonical items list for this level.
                is_semantically_valid, semantic_issues, taxonomy_with_original_kid_titles_if_valid = \
                    self.examiner.examine_taxonomy( # This now takes the current level's canonical items
                        current_items_for_llm,    # The distinct items LLM was asked to cluster
                        llm_output_list,          # LLM's output (parents with kid titles)
                        min_parents, max_parents
                    )
                current_issues_for_feedback = semantic_issues
                generated_parent_list_for_this_level = taxonomy_with_original_kid_titles_if_valid

                if is_semantically_valid:
                    # taxonomy_with_original_kid_titles_if_valid has parent.kids as CANONICAL titles
                    # (because examine_taxonomy's first arg was canonical_items)
                    
                    # NOW, expand kid titles back to all original variants if desired for storage
                    final_taxonomy_for_this_level_storage = []
                    map_norm_to_orig_this_level = self.normalized_to_originals_map_per_level.get(level_num - 1) # From previous level or initial load

                    for parent_obj_llm_canon_kids in taxonomy_with_original_kid_titles_if_valid: # This has canonical kids
                        expanded_kids = []
                        for canonical_kid_title in parent_obj_llm_canon_kids.get("kids", []):
                            # Find the normalized form of this canonical_kid_title
                            normalized_canonical_kid = self.examiner_normalizer(canonical_kid_title)
                            
                            if map_norm_to_orig_this_level and normalized_canonical_kid in map_norm_to_orig_this_level:
                                if expand_kids_to_all_variants:
                                    # Add all original titles that map to this normalized canonical kid
                                    for original_item_obj in map_norm_to_orig_this_level[normalized_canonical_kid]["all_originals"]:
                                        expanded_kids.append(original_item_obj['title'])
                                else:
                                    # Just add the chosen canonical title (which is what LLM used)
                                    expanded_kids.append(canonical_kid_title)
                            else: # Should not happen if logic is correct, implies canonical_kid_title wasn't properly mapped
                                expanded_kids.append(canonical_kid_title) # Fallback
                        
                        # Create a new parent object for storage with expanded (or canonical) kids
                        stored_parent_obj = {
                            "title": parent_obj_llm_canon_kids["title"],
                            "description": parent_obj_llm_canon_kids["description"],
                            "kids": sorted(list(set(expanded_kids))) # Ensure unique and sorted
                        }
                        final_taxonomy_for_this_level_storage.append(stored_parent_obj)

                    self.full_taxonomy_by_level[level_num] = final_taxonomy_for_this_level_storage
                    
                    # Prepare for next level: new_canonical_parents become the new current_items_for_llm
                    parents_for_next_level_input = [
                        {"title": p_obj["title"], "description": p_obj["description"]}
                        for p_obj in final_taxonomy_for_this_level_storage # Use the stored parents
                    ]
                    
                    current_items_for_llm, norm_to_orig_next, orig_to_canon_next = \
                        self._create_canonical_item_list_and_mapping(parents_for_next_level_input, self.examiner_normalizer)
                    
                    self.normalized_to_originals_map_per_level[level_num] = norm_to_orig_next
                    self.original_to_canonical_title_map_per_level[level_num] = orig_to_canon_next

                    current_level_context_file_name = f"level_{level_num}_canonical_parents_with_desc.json" # Store canonical for next LLM
                    output_path = os.path.join(self.output_dir, current_level_context_file_name)
                    with open(output_path, 'w', encoding='utf-8') as f_out: 
                        json.dump(current_items_for_llm, f_out, indent=2)
                    level_successfully_generated = True
                    break
                else: # Semantic errors
                    # ... (handling retries as before) ...
                    print(f"Level {level_num} FAILED SEMANTIC VALIDATION (Attempt {attempt + 1}):")
                    for issue in semantic_issues: print(f"  - {issue}")
                    if attempt == retry_attempts - 1: print("Max retries for semantic errors.")
                    else: print("Retrying with semantic feedback and previous (flawed but parsed) output...")

            if not level_successfully_generated: print(f"Failed Level {level_num}."); break
            # Use the 'generated_parent_list_for_this_level' which holds the output from the last successful
            # call to examiner (which is taxonomy_with_canonical_kid_titles)
            if generated_parent_list_for_this_level is not None and \
               len(generated_parent_list_for_this_level) < min_parents_to_continue_threshold:
                print(f"Stopping: Number of parents generated for Level {level_num} ({len(generated_parent_list_for_this_level)}) "
                      f"is less than threshold ({min_parents_to_continue_threshold}).")
                break

            # Existing stopping condition (now checks canonical list for next iteration)
            if not current_items_for_llm or len(current_items_for_llm) <= 1: 
                 print(f"Stopping: Only {len(current_items_for_llm)} canonical item(s) prepared for the next level.")
                 break
            if not current_items_for_llm or len(current_items_for_llm) <= 1: break # Now check canonical list
        print("\n--- Taxonomy Generation Complete ---")
        return self.full_taxonomy_by_level



output_dir = "botswana_taxonomy_output"
input_file_name = 'botswana_init_input_test_full_cannonical.json'
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
workflow = WorkflowManager(initial_leaf_items_file_json=input_file_name, output_dir=output_dir, client=client, llm_model='gemini-2.5-pro-preview-05-06')
# Run for fewer levels with this small dataset
final_taxonomy = workflow.run_taxonomy_generation(max_levels=5, initial_level_name="Leaf Items", retry_attempts=6, expand_kids_to_all_variants=False, min_parents_to_continue_threshold=10)


 
# save the final taxonomy
output_path = os.path.join(output_dir, 'full_generated_taxonomy_with_desc.json')
with open(output_path, 'w', encoding='utf-8') as f_out:
    json.dump(final_taxonomy, f_out, indent=2)


