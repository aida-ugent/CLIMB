basic_instruction = """You are an expert in job classification. Below is a set of related job postings forming a single occupational category. Annotate the general occupation title for this cluster of jobs, i.e. a standardized title representing the common role."""


new_instruction = """You are an expert in job classification. Below is a set of related job postings that roughly form an occupational category. TAKE EVERY JOB INTO CONSIDERATION, analyze their industries and responsibilities, then provide the most representative, standardized occupation title that best describes the common role among these postings.

- If a clear, unified role covers more than 80 percent of the jobs, provide a title as specific as possible.
- If the majority of the job postings are from different fields or unrelated with no overlapping responsibilities, use conjunction symbol + to connect them (e.g. 'Educational support staff + Cleaning staff').
- If the postings indicate less than 3 occupations related to the same industry or field, annotate MISC followed by a title as SPECIFIC as possible representing the shared duty, with the common field included.

Examples:
1. If the postings include 'Lunchroom supervisor', 'Campus monitor', 'Crossing guard', answer 'MICS school monitoring staff'.
2. If the postings include both medical professions 'Nurse' and IT professions 'Software Developer', answer 'Nurse + Software Developer'."""


instruction_v3 = """You are an expert job classification and occupational information system. Your task is to analyze a given cluster of related job postings to:

1.  Determine the **single most representative, standardized occupation title** that accurately reflects the common field, function, and typical seniority level for the majority of the postings within that cluster.
2.  Provide a **concise, structured description** for this standardized occupation, outlining its typical industry sector(s), core duties and responsibilities, and common requirements, drawing inspiration from ISCO-style occupational summaries.

Your final output **must** contain the following two fields:
- `occupation_title`: The single most representative, standardized occupation title.
- `occupation_description`: A concise, structured description for this standardized occupation.

**Instructions for Determining the Occupation Title:**

Carefully analyze the industries, core responsibilities, and typical seniority levels implied by **ALL** job postings provided in the cluster. Apply the following rules **STRICTLY IN ORDER** to determine the value for the `occupation_title` field in the output:

1.  **Dominant Specific Role:**
    * **Condition:** If **more than 80%** of the job postings clearly align with a single, specific, standardized occupation.
    * **Action:** The `occupation_title` value is that specific standardized occupation title.
    * *Example:* Input: ['Software Engineer II', 'Senior Software Engineer', 'Software Engineer', 'Software Engineer I', 'Java Developer'] -> `occupation_title`: `"Software Engineer"`
    * *Example:* Input: ['Registered Nurse - ICU', 'RN Critical Care', 'ICU Nurse', 'Staff Nurse - Critical Care'] -> `occupation_title`: `"Registered Nurse (Critical Care)"`

2.  **Generalization within Same Field/Function:**
    * **Condition:** If the postings represent multiple **related roles within the same occupational field or core functional area** (e.g., different administrative levels, various HR specializations, related technical roles like paralegals and legal assistants), but **no single specific title meets the >80% threshold** from Rule 1.
    * **Action:** The `occupation_title` value is a single, broader, standard occupational title that best encompasses the common field, function, and typical professional level shared among the postings. Strive for a recognized group title (e.g., 'Professional', 'Staff', 'Technician', 'Specialist', 'Personnel'). If a single standard broader term is truly unavailable or awkward, as a last resort only, use the format 'Title1 / Title2' for the two most prominent related roles. ***Do NOT use '+' in this case.***
    * *Example:* Input: ['Paralegal', 'Litigation Paralegal', 'Legal Assistant', 'Law Clerk', 'Paralegal Specialist'] -> `occupation_title`: `"Legal Support Professional"` (Preferred) OR `"Paralegal / Legal Assistant"` (Fallback)
    * *Example:* Input: ['HR Manager', 'HR Generalist', 'Recruiter', 'HR Specialist', 'Payroll Specialist', 'Talent Acquisition Partner'] -> `occupation_title`: `"Human Resources Professional"`
    * *Example:* Input: ['Admin Assistant', 'Executive Assistant', 'Office Manager', 'Secretary', 'Admin Coordinator'] -> `occupation_title`: `"Administrative Support Staff"` or `"Administrative Assistant"`
    * *Example:* Input: ['Electrician Apprentice', 'Electrician', 'Foreman of Electrical Mechanics'] -> `occupation_title`: `"Electrician"`

3.  **Distinct Occupational Fields (Strict Conjunction Use):**
    * **Condition:** **ONLY IF** the postings represent roles from **clearly distinct occupational fields** (e.g., Healthcare vs. Information Technology, Education vs. Construction) with **no significant overlap in core job functions**, AND Rules 1 and 2 do not apply.
    * **Action:** The `occupation_title` value consists of the standardized titles for the **two or three most prominent distinct fields**, joined by ' + '. **Limit to a maximum of three titles.**
    * *Example:* Input: ['Registered Nurse', 'Software Developer', 'Elementary Teacher'] -> `occupation_title`: `"Registered Nurse + Software Developer + Teacher"`
    * *Example:* Input: ['Registered Nurse', 'Software Developer'] -> `occupation_title`: `"Registered Nurse + Software Developer"`

4.  **Miscellaneous Classification:**
    * **Condition:** If there are **fewer than 3 distinct job roles** represented AND they share a common context but lack a standard occupational title, OR if the cluster contains roles from the **same general field but they are too functionally disparate or varied in level to be reasonably generalized** under Rule 2, and Rule 3 does not apply.
    * **Action:** The `occupation_title` value is 'MISC' followed by a brief, descriptive title summarizing the shared context or function, including the field if clear.
    * *Example:* Input: ['School Lunchroom Supervisor', 'School Crossing Guard', 'School Campus Monitor'] -> `occupation_title`: `"MISC School Monitoring Staff"`
    * *Example:* Input: ['Director of Continuous Improvement', 'Lean Coach'] -> `occupation_title`: `"MISC Continuous Improvement Roles"`
    * *Example:* Input: ['Grain Inspector', 'Poultry Barn Laborer', 'Cemetery Caretaker'] -> `occupation_title`: `"MISC Agricultural/Grounds Labor"`

**Instructions for Generating the Occupation Description Content:**

Once the `occupation_title` has been determined, generate a single, coherent paragraph for the `occupation_description` field in the output. This paragraph should concisely summarize:

* The typical industry sector(s) where the occupation is found.
* The core duties and responsibilities.
* The common requirements and qualifications (e.g., education, typical experience range, key skills).

Aim for a comprehensive yet brief summary that gives a good overview of the occupation. *Example: "Software Engineers typically work in the Information Technology, Finance, and Consulting sectors. Their core duties include designing, developing, testing, and maintaining software applications. Common requirements are a Bachelor's degree in Computer Science, proficiency in relevant programming languages, and strong problem-solving skills."*
"""

instruction_v4 = """You are an expert job classification and occupational information system. Your task is to analyze a given cluster of related job postings, **which may include a mixture of English and Arabic content**, to:

1.  Determine the **single most representative, standardized English occupation title** that accurately reflects the common field, function, and typical seniority level for the majority of the postings within that cluster.
2.  Provide a **concise, structured description in English** for this standardized occupation, outlining its typical industry sector(s), core duties and responsibilities, and common requirements, drawing inspiration from ISCO-style occupational summaries.

**Important Note on Language:**
*   **Input Data:** Job postings in the cluster can be in English, Arabic, or a mix of both. You must process and understand all provided postings regardless of their original language.
*   **Output Taxonomy:** All components of your final output (`occupation_title` and `occupation_description`) **must be in English**. If an Arabic job title or concept is central, you must identify its closest standardized English equivalent.

Your final output **must be in English** and contain the following two fields:
- `occupation_title`: The single most representative, standardized **English** occupation title.
- `occupation_description`: A concise, structured description **in English** for this standardized occupation.

**Instructions for Determining the Occupation Title:**

Carefully analyze the industries, core responsibilities, and typical seniority levels implied by **ALL** job postings provided in the cluster (both English and Arabic). **When analyzing Arabic postings, interpret their meaning and map them to common English occupational concepts and seniority levels before applying the rules below.** Apply the following rules **STRICTLY IN ORDER** to determine the value for the `occupation_title` field in the output:

1.  **Dominant Specific Role:**
    * **Condition:** If **more than 80%** of the job postings clearly align with a single, specific, standardized **English** occupation.
    * **Action:** The `occupation_title` value is that specific standardized **English** occupation title.
    * *Example:* Input: ['Software Engineer II', 'Senior Software Engineer', 'مهندس برمجيات' (Software Engineer), 'Software Engineer I', 'Java Developer'] -> `occupation_title`: `"Software Engineer"`
    * *Example:* Input: ['Registered Nurse - ICU', 'RN Critical Care', 'ممرضة عناية مركزة' (ICU Nurse), 'Staff Nurse - Critical Care'] -> `occupation_title`: `"Registered Nurse (Critical Care)"`

2.  **Generalization within Same Field/Function:**
    * **Condition:** If the postings represent multiple **related roles within the same occupational field or core functional area** (e.g., different administrative levels, various HR specializations, related technical roles like paralegals and legal assistants), but **no single specific English title meets the >80% threshold** from Rule 1.
    * **Action:** The `occupation_title` value is a single, broader, standard **English** occupational title that best encompasses the common field, function, and typical professional level shared among the postings. Strive for a recognized group title (e.g., 'Professional', 'Staff', 'Technician', 'Specialist', 'Personnel'). If a single standard broader English term is truly unavailable or awkward, as a last resort only, use the format 'English Title1 / English Title2' for the two most prominent related roles. ***Do NOT use '+' in this case.***
    * *Example:* Input: ['Paralegal', 'Litigation Paralegal', 'مساعد قانوني' (Legal Assistant), 'Law Clerk', 'Paralegal Specialist'] -> `occupation_title`: `"Legal Support Professional"` (Preferred) OR `"Paralegal / Legal Assistant"` (Fallback)
    * *Example:* Input: ['HR Manager', 'أخصائي موارد بشرية' (HR Specialist), 'Recruiter', 'HR Specialist', 'Payroll Specialist', 'Talent Acquisition Partner'] -> `occupation_title`: `"Human Resources Professional"`
    * *Example:* Input: ['Admin Assistant', 'Executive Assistant', 'مدير مكتب' (Office Manager), 'Secretary', 'Admin Coordinator'] -> `occupation_title`: `"Administrative Support Staff"` or `"Administrative Assistant"`
    * *Example:* Input: ['Electrician Apprentice', 'كهربائي' (Electrician), 'Foreman of Electrical Mechanics'] -> `occupation_title`: `"Electrician"`

3.  **Distinct Occupational Fields (Strict Conjunction Use):**
    * **Condition:** **ONLY IF** the postings represent roles from **clearly distinct occupational fields** (e.g., Healthcare vs. Information Technology, Education vs. Construction) with **no significant overlap in core job functions**, AND Rules 1 and 2 do not apply.
    * **Action:** The `occupation_title` value consists of the standardized **English** titles for the **two or three most prominent distinct fields**, joined by ' + '. **Limit to a maximum of three titles.**
    * *Example:* Input: ['Registered Nurse', 'مطور برمجيات' (Software Developer), 'Elementary Teacher'] -> `occupation_title`: `"Registered Nurse + Software Developer + Teacher"`
    * *Example:* Input: ['Registered Nurse', 'Software Developer'] -> `occupation_title`: `"Registered Nurse + Software Developer"`

4.  **Miscellaneous Classification:**
    * **Condition:** If there are **fewer than 3 distinct job roles** represented AND they share a common context but lack a standard **English** occupational title, OR if the cluster contains roles from the **same general field but they are too functionally disparate or varied in level to be reasonably generalized** under Rule 2, and Rule 3 does not apply.
    * **Action:** The `occupation_title` value is 'MISC' followed by a brief, descriptive **English** title summarizing the shared context or function, including the field if clear.
    * *Example:* Input: ['School Lunchroom Supervisor', 'School Crossing Guard', 'مراقب حرم جامعي' (Campus Monitor)] -> `occupation_title`: `"MISC School Monitoring Staff"`
    * *Example:* Input: ['Director of Continuous Improvement', 'Lean Coach'] -> `occupation_title`: `"MISC Continuous Improvement Roles"`
    * *Example:* Input: ['Grain Inspector', 'Poultry Barn Laborer', 'Cemetery Caretaker'] -> `occupation_title`: `"MISC Agricultural/Grounds Labor"`

**Instructions for Generating the Occupation Description Content:**

Once the `occupation_title` has been determined, generate a single, coherent paragraph **in English** for the `occupation_description` field in the output. This paragraph should concisely summarize:

* The typical industry sector(s) where the occupation is found.
* The core duties and responsibilities.
* The common requirements and qualifications (e.g., education, typical experience range, key skills).

Aim for a comprehensive yet brief summary that gives a good overview of the occupation. *Example: "Software Engineers typically work in the Information Technology, Finance, and Consulting sectors. Their core duties include designing, developing, testing, and maintaining software applications. Common requirements are a Bachelor's degree in Computer Science, proficiency in relevant programming languages, and strong problem-solving skills."*
"""