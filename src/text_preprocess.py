import re
from bs4 import BeautifulSoup
import unicodedata
import warnings
from typing import Dict, Optional, Union
import pandas as pd
from bs4 import MarkupResemblesLocatorWarning
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import pickle



# Suppress BeautifulSoup warning
warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)

class TextChunker:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        """Initialize the TextChunker with a specified sentence transformer model."""
        self.model = SentenceTransformer(model_name)

    def process_file(self, file_path, context_window=1, percentile_threshold=95, 
                    min_chunk_size=3, max_chunk_size=10):
        """
        Process a text file and split it into semantically meaningful chunks.
        
        Args:
            file_path: Path to the text file
            context_window: Number of sentences to consider on either side for context
            percentile_threshold: Percentile threshold for identifying breakpoints
            min_chunk_size: Minimum number of sentences in a chunk
            max_chunk_size: Maximum number of sentences in a chunk
            
        Returns:
            list: Semantically coherent text chunks
        """
        # Process the text file
        sentences = self._load_text(file_path)
        contextualized = self._add_context(sentences, context_window)
        embeddings = self.model.encode(contextualized)
        
        # Create and refine chunks
        distances = self._calculate_distances(embeddings)
        breakpoints = self._identify_breakpoints(distances, percentile_threshold)
        initial_chunks = self._create_chunks(sentences, breakpoints, max_chunk_size)
        
        # Merge small chunks for better coherence
        chunk_embeddings = self.model.encode(initial_chunks)
        final_chunks = self._merge_small_chunks(initial_chunks, chunk_embeddings, 
                                              min_size=min_chunk_size,
                                              max_size=max_chunk_size)
        return final_chunks
    

    def _load_text(self, file_path):
        """Load and tokenize text from a file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return sent_tokenize(text)

    def _add_context(self, sentences, window_size):
        """Combine sentences with their neighbors for better context."""
        contextualized = []
        for i in range(len(sentences)):
            start = max(0, i - window_size)
            end = min(len(sentences), i + window_size + 1)
            context = ' '.join(sentences[start:end])
            contextualized.append(context)
        return contextualized

    def _calculate_distances(self, embeddings):
        """Calculate cosine distances between consecutive embeddings."""
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distance = 1 - similarity
            distances.append(distance)
        return distances

    def _identify_breakpoints(self, distances, threshold_percentile):
        """Find natural breaking points in the text based on semantic distances."""
        threshold = np.percentile(distances, threshold_percentile)
        return [i for i, dist in enumerate(distances) if dist > threshold]

    def _create_chunks(self, sentences, breakpoints, max_size):
        """Create initial text chunks based on identified breakpoints."""
        chunks = []
        start_idx = 0
        
        for breakpoint in breakpoints:
            # Check if current chunk would exceed max_size
            if breakpoint + 1 - start_idx > max_size:
                # Split into smaller chunks
                for i in range(start_idx, breakpoint + 1, max_size):
                    end_idx = min(i + max_size, breakpoint + 1)
                    chunk = ' '.join(sentences[i:end_idx])
                    chunks.append(chunk)
                start_idx = breakpoint + 1
            else:
                chunk = ' '.join(sentences[start_idx:breakpoint + 1])
                chunks.append(chunk)
                start_idx = breakpoint + 1
            
        # Handle the final chunk
        remaining_sentences = sentences[start_idx:]
        for i in range(0, len(remaining_sentences), max_size):
            chunk = ' '.join(remaining_sentences[i:i + max_size])
            chunks.append(chunk)
        
        return chunks

    def _merge_small_chunks(self, chunks, embeddings, min_size, max_size):
        """Merge small chunks with their most similar neighbor."""
        final_chunks = [chunks[0]]
        merged_embeddings = [embeddings[0]]
        
        for i in range(1, len(chunks) - 1):
            current_chunk_size = len(chunks[i].split('. '))
            
            if current_chunk_size < min_size:
                # Check if merging would exceed max_size
                prev_size = len(final_chunks[-1].split('. '))
                next_size = len(chunks[i + 1].split('. '))
                
                # Calculate similarities
                prev_similarity = cosine_similarity([embeddings[i]], [merged_embeddings[-1]])[0][0]
                next_similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                
                # Merge with previous chunk if it won't exceed max_size and has higher similarity
                if prev_similarity > next_similarity and prev_size + current_chunk_size <= max_size:
                    final_chunks[-1] = f"{final_chunks[-1]} {chunks[i]}"
                    merged_embeddings[-1] = (merged_embeddings[-1] + embeddings[i]) / 2
                # Merge with next chunk if it won't exceed max_size
                elif current_chunk_size + next_size <= max_size:
                    chunks[i + 1] = f"{chunks[i]} {chunks[i + 1]}"
                    embeddings[i + 1] = (embeddings[i] + embeddings[i + 1]) / 2
                else:
                    # If can't merge without exceeding max_size, keep as separate chunk
                    final_chunks.append(chunks[i])
                    merged_embeddings.append(embeddings[i])
            else:
                final_chunks.append(chunks[i])
                merged_embeddings.append(embeddings[i])
        
        final_chunks.append(chunks[-1])
        return final_chunks
    

# class TextCleaner:
#     """A class to clean and normalize text data with configurable options."""
    
#     def __init__(self, options: Optional[Dict[str, bool]] = None):
#         self.default_options = {
#             'remove_html': True,
#             'remove_urls': True,
#             'remove_emails': True,
#             'remove_phone_numbers': True,
#             'remove_special_chars': False,
#             'remove_numbers': False,
#             'remove_extra_whitespace': True,
#             'remove_punctuation': False,
#             'convert_bullets': True,
#             'fix_sentence_spacing': True,
#             'lowercase': False,
#             'normalize_unicode': True
#         }
#         self.options = {**self.default_options, **(options or {})}
        
#         # Compile regex patterns
#         self.patterns = {
#             'urls': re.compile(r'http\S+|www\.\S+'),
#             'emails': re.compile(r'\S+@\S+\.\S+'),
#             'phone_numbers': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
#             'bullets': re.compile(r'[•●■◆▪️►▶︎\-*]'),
#             'newlines': re.compile(r'[\r\n]+'),
#             'extra_spaces': re.compile(r'\s+'),
#             'multiple_periods': re.compile(r'\.{2,}'),
#             'space_before_punct': re.compile(r'\s+([.,!?;:])'),
#             'missing_space_after_period': re.compile(r'\.(?=[A-Za-z])')
#         }

#     def clean_text(self, text: str) -> str:
#         if not isinstance(text, str) or not text.strip():
#             return ""
            
#         if self.options['remove_html']:
#             try:
#                 text = BeautifulSoup(text, "lxml").get_text()
#             except:
#                 text = BeautifulSoup(text, "html.parser").get_text()
        
#         if self.options['normalize_unicode']:
#             text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        
#         if self.options['remove_urls']:
#             text = self.patterns['urls'].sub(' ', text)
        
#         if self.options['remove_emails']:
#             text = self.patterns['emails'].sub(' ', text)
        
#         if self.options['remove_phone_numbers']:
#             text = self.patterns['phone_numbers'].sub(' ', text)
        
#         if self.options['convert_bullets']:
#             text = self.patterns['bullets'].sub('.', text)
        
#         text = self.patterns['newlines'].sub('. ', text)
        
#         if self.options['remove_special_chars']:
#             text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
        
#         if self.options['remove_numbers']:
#             text = re.sub(r'\d+', ' ', text)
        
#         if self.options['fix_sentence_spacing']:
#             text = self.patterns['multiple_periods'].sub('.', text)
#             text = self.patterns['space_before_punct'].sub(r'\1', text)
#             text = self.patterns['missing_space_after_period'].sub('. ', text)
        
#         if self.options['remove_extra_whitespace']:
#             text = self.patterns['extra_spaces'].sub(' ', text)
#             text = text.strip()
        
#         if text and text[-1] not in '.!?':
#             text += '.'
            
#         return text
# #-----------------------------------------------------------------------------------    

import re
import unicodedata
from typing import Optional, Dict, Any

# Optional imports with fallbacks
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore
    print("Warning: BeautifulSoup not installed. HTML removal will be skipped if enabled.")

try:
    from lingua import Language, LanguageDetectorBuilder, IsoCode639_1
    LINGUA_AVAILABLE = True
except ImportError:
    LINGUA_AVAILABLE = False
    Language = None # type: ignore
    LanguageDetectorBuilder = None # type: ignore
    IsoCode639_1 = None # type: ignore
    print("Warning: Lingua library not found. Language detection will be disabled if enabled.")
    raise ImportError("Lingua library not found. Please install it with 'pip install lingua-language-detector'.")

# from pyarabic.araby import strip_tashkeel, strip_tatweel, normalize_ligature, \
#                             normalize_hamza, normalize_alef, normalize_teh_marbuta

try:
    from pyarabic.araby import (
        strip_tashkeel,
        strip_tatweel,
        normalize_alef,
        normalize_ligature,
        normalize_hamza
    )
    from pyarabic.normalize import  normalize_searchtext, normalize_spellerrors
    PYARABIC_AVAILABLE = True
except ImportError:
    PYARABIC_AVAILABLE = False
    # Define dummy functions if pyarabic is not available
    def strip_tashkeel(text: str) -> str: print("Warning: pyarabic not found, strip_tashkeel is a no-op."); return text
    def strip_tatweel(text: str) -> str: print("Warning: pyarabic not found, strip_tatweel is a no-op."); return text
    def normalize_ligature(text: str) -> str: print("Warning: pyarabic not found, normalize_ligature is a no-op."); return text
    def normalize_hamza(text: str) -> str: print("Warning: pyarabic not found, normalize_hamza is a no-op."); return text
    def normalize_alef(text: str) -> str: print("Warning: pyarabic not found, normalize_alef is a no-op."); return text
    def normalize_teh_marbuta(text: str) -> str: print("Warning: pyarabic not found, normalize_teh_marbuta is a no-op."); return text
    if PYARABIC_AVAILABLE is False: # To avoid multiple prints if only some are missing.
        print("Warning: pyarabic library not found. Some Arabic normalization features will be unavailable.")
    raise ImportError("pyarabic library not found. Please install it with 'pip install pyarabic'.")

try:
    from camel_tools.utils.charmap import CharMapper
    from camel_tools.utils.normalize import normalize_alef_maksura_ar, normalize_teh_marbuta_ar, \
                                            normalize_alef_ar, normalize_unicode as ct_normalize_unicode
    CAMELTOOLS_AVAILABLE = True
except ImportError:
    CAMELTOOLS_AVAILABLE = False
    CharMapper = None # type: ignore
    # Define dummy functions if camel_tools is not available
    def normalize_alef_maksura_ar(text: str) -> str: print("Warning: camel_tools not found, normalize_alef_maksura_ar is a no-op."); return text
    def normalize_teh_marbuta_ar(text: str) -> str: print("Warning: camel_tools not found, normalize_teh_marbuta_ar is a no-op."); return text
    def normalize_alef_ar(text: str) -> str: print("Warning: camel_tools not found, normalize_alef_ar is a no-op."); return text
    def ct_normalize_unicode(text: str) -> str: print("Warning: camel_tools not found, normalize_unicode is a no-op."); return text
    if CAMELTOOLS_AVAILABLE is False:
        print("Warning: camel_tools library not found. Some Arabic normalization features will be unavailable or use fallbacks.")
    # raise ImportError("camel_tools library not found. Please install it with 'pip install camel-tools'.")

# Define Arabic punctuation
ARABIC_PUNCTUATION = r'\u060C\u061B\u061F\u066A\u066B\u066C\u066D' # ،؛؟٪٫٬٭
# Define Arabic letters and numbers
ARABIC_CHARS_NUMS = r'\u0600-\u06FF\u0750-\u077F\u0660-\u0669'


class TextCleaner:
    """
    A class to clean and normalize text data with configurable options,
    supporting both English and Arabic.
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None,
                 default_lang: str = 'en'):
        self.default_options = {
            # General options
            'remove_html': True,
            'remove_urls': True,
            'remove_emails': True,
            'remove_phone_numbers': True,
            'remove_special_chars': True,
            'remove_numbers': False,
            'remove_extra_whitespace': True,
            'remove_punctuation': False,
            'convert_bullets': True,
            'fix_sentence_spacing': True,
            'lowercase': False,
            # 'normalize_unicode': True, 
            'normalize_unicode_form': 'NFKC',
            'normalize_unicode_to_ascii': False,

            # Arabic-specific options
            'normalize_arabic_chars': True,
            'remove_tashkeel': True,
            'remove_tatweel': True,
            'use_camel_tools_norm': True,

            # Language detection
            'detect_language': True if LINGUA_AVAILABLE else False,
        }
        self.options = {**self.default_options, **(options or {})}
        self.default_lang = default_lang.lower() if default_lang else 'en'

        if not BeautifulSoup and self.options['remove_html']:
            # Warning already printed at import time if BeautifulSoup is None
            self.options['remove_html'] = False

        self.detector = None
        if LINGUA_AVAILABLE and self.options['detect_language']:
            try:
                self.detector = LanguageDetectorBuilder.from_languages(
                    Language.ARABIC, Language.ENGLISH
                ).build()
            except Exception as e:
                print(f"Warning: Could not initialize LanguageDetector: {e}. Disabling detection.")
                self.options['detect_language'] = False
        elif not LINGUA_AVAILABLE and self.options['detect_language']:
            # Warning already printed at import time if LINGUA_AVAILABLE is False
            self.options['detect_language'] = False

        self.ct_ar_normalizer = None
        if CAMELTOOLS_AVAILABLE and self.options.get('use_camel_tools_norm', True):
            try:
                self.ct_ar_normalizer = CharMapper.builtin_mapper('arclean')
            except Exception as e:
                print(f"Warning: Could not initialize Camel Tools CharMapper: {e}. Camel Tools normalization might be affected.")
                self.ct_ar_normalizer = None

        self.patterns = {
            'urls': re.compile(r'http\S+|www\.\S+'),
            'emails': re.compile(r'\S+@\S+\.\S+'),
            'phone_numbers': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'bullets': re.compile(r'[•●■◆▪️►▶︎\-*]'),
            'newlines': re.compile(r'[\r\n]+'),
            'extra_spaces': re.compile(r'\s+'),
            'multiple_periods': re.compile(r'\.{2,}'),

            # 'emails': re.compile(r'\S+@\S+\.\S+'),
            # 'phone_numbers': re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            # 'bullets': re.compile(r'[•●■◆▪️►▶︎\-*]'),
            # 'newlines': re.compile(r'[\r\n]+'),
            # 'extra_spaces': re.compile(r'\s+'),
            # 'multiple_periods': re.compile(r'\.{2,}'),
            'space_before_punct_en': re.compile(r'\s+([.,!?;:])'),
            'missing_space_after_period_en': re.compile(r'\.(?=[A-Za-z])'),
            'space_before_punct_ar': re.compile(fr'\s+([{ARABIC_PUNCTUATION}؟،؛.:])'),
            'missing_space_after_punct_ar': re.compile(fr'([{ARABIC_PUNCTUATION}؟،؛.])(?=[{ARABIC_CHARS_NUMS}])'),
            'en_special_chars_keep': re.compile(r'[^a-zA-Z0-9\s.,!?;:\'"()\-\–\—\/]'),
            'ar_special_chars_keep': re.compile(fr'[^{ARABIC_CHARS_NUMS}a-zA-Z0-9\s.,!?;:\'"(){ARABIC_PUNCTUATION}\-\–\—\/]'),
            'en_punctuation': re.compile(r'[.,!?;:\'"()\-\–\—]'),
            'ar_punctuation': re.compile(fr'[{ARABIC_PUNCTUATION}؟،؛.:\'"()\-\–\—]'),
            'numbers': re.compile(r'\d+'),
        }

    def _detect_lang(self, text: str) -> str:
        if self.detector and text.strip(): # Ensure detector exists and text is not empty
            try:
                lang_obj = self.detector.detect_language_of(text)
                if lang_obj:
                    if lang_obj.iso_code_639_1 == IsoCode639_1.AR:
                        return 'ar'
                    elif lang_obj.iso_code_639_1 == IsoCode639_1.EN:
                        return 'en'
            except Exception as e:
                # print(f"Language detection failed for a segment: {e}") # Optional: log error
                pass # Fall through to default_lang
        return self.default_lang


    def clean_text(self, text: str, lang: Optional[str] = None) -> str:
        if not isinstance(text, str) or not text.strip(): return ""

        current_lang = lang.lower() if lang else self.default_lang
        if not lang and self.options['detect_language']:
            current_lang = self._detect_lang(text)

        if self.options['remove_html'] and BeautifulSoup:
            try: text = BeautifulSoup(text, "lxml").get_text()
            except Exception:
                try: text = BeautifulSoup(text, "html.parser").get_text()
                except Exception: pass

        norm_form = self.options['normalize_unicode_form']
        if norm_form and norm_form in ['NFC', 'NFD', 'NFKC', 'NFKD']:
            text = unicodedata.normalize(norm_form, text)

        if current_lang == 'ar':
            if self.options['normalize_arabic_chars']:
                if self.ct_ar_normalizer and self.options['use_camel_tools_norm']: text = self.ct_ar_normalizer(text)
                elif CAMELTOOLS_AVAILABLE and self.options['use_camel_tools_norm']:
                    text = ct_normalize_unicode(text)
                    text = normalize_alef_maksura_ar(text)
                    text = normalize_teh_marbuta_ar(text)
                    text = normalize_alef_ar(text)
                elif PYARABIC_AVAILABLE:
                    text = normalize_alef(text); text = normalize_hamza(text); text = normalize_ligature(text); text = normalize_teh_marbuta(text)
            if self.options['remove_tashkeel'] and PYARABIC_AVAILABLE: text = strip_tashkeel(text)
            if self.options['remove_tatweel'] and PYARABIC_AVAILABLE: text = strip_tatweel(text)

        if current_lang == 'en':
            if self.options['lowercase']: text = text.lower()
            if self.options['normalize_unicode_to_ascii']:
                text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        
        # --- DEBUG SECTION FOR URLS ---
        # print(f"DEBUG Initial text for URL removal: '{text[:150]}'")
        # print(f"DEBUG: self.options['remove_urls'] = {self.options['remove_urls']}")
        if self.options['remove_urls']:
            text_before_url_removal = text # Store for comparison
            text = self.patterns['urls'].sub(' ', text)
            # if text_before_url_removal == text and "http" in text_before_url_removal: # Check if http was present and text didn't change
                # print(f"DEBUG: URL regex did not change text containing 'http'. Original: '{text_before_url_removal[:150]}'")
        # print(f"DEBUG: Text after URL sub attempt: '{text[:150]}'")
        # --- END DEBUG SECTION ---

        if self.options['remove_emails']: text = self.patterns['emails'].sub(' ', text)
        if self.options['remove_phone_numbers']: text = self.patterns['phone_numbers'].sub(' ', text)
        if self.options['convert_bullets']: text = self.patterns['bullets'].sub('. ', text)
        text = self.patterns['newlines'].sub('. ', text)
        if self.options['remove_numbers']: text = self.patterns['numbers'].sub(' ', text)

        if self.options['remove_punctuation']:
            if current_lang == 'ar': text = self.patterns['ar_punctuation'].sub(' ', text)
            else: text = self.patterns['en_punctuation'].sub(' ', text)
        elif self.options['remove_special_chars']:
            if current_lang == 'ar': text = self.patterns['ar_special_chars_keep'].sub(' ', text)
            else: text = self.patterns['en_special_chars_keep'].sub(' ', text)

        if self.options['fix_sentence_spacing']:
            text = self.patterns['multiple_periods'].sub('.', text)
            if current_lang == 'ar':
                text = self.patterns['space_before_punct_ar'].sub(r'\1', text)
                text = self.patterns['missing_space_after_punct_ar'].sub(r'\1 ', text)
            else:
                # print(f"DEBUG before missing_space_after_period_en: '{text[:150]}'")
                text = self.patterns['space_before_punct_en'].sub(r'\1', text)
                text = self.patterns['missing_space_after_period_en'].sub('. ', text) # Culprit for altering survived URLs
                # print(f"DEBUG after missing_space_after_period_en: '{text[:150]}'")


        if self.options['remove_extra_whitespace']:
            text = self.patterns['extra_spaces'].sub(' ', text)
            text = text.strip()

        if text and text[-1] not in '.!?؟':
             if current_lang == 'ar' and any(c in ARABIC_CHARS_NUMS for c in text): pass
             elif current_lang == 'en': text += '.'
        return text

if __name__ == '__main__':
    print("--- Testing with 'remove_special_chars' as default True ---")

    # This cleaner will use remove_special_chars=True by default
    default_cleaner = TextCleaner(options=custom_options) 

    test_text_default_behavior = "<p>URL: http://site.com/page. Text: a/b c#d. Email: x@y.com</p>"
    print(f"\nOriginal for default behavior test:\n{test_text_default_behavior}")
    cleaned_default_behavior = default_cleaner.clean_text(test_text_default_behavior, lang='en')
    print(f"Cleaned with default options (remove_special_chars=True):\n{cleaned_default_behavior}")
    # Expected output (approximate, exact spacing might vary slightly before final strip):
    # "URL: Text: a/b c d. Email:."
    # - URL and email removed.
    # - In "a/b", '/' is kept.
    # - In "c#d", '#' is removed.
    # - HTML tags removed.
    # - Final period added.

    print("\n--- English Cleaning (Defaults: remove_special_chars=True, remove_urls=True) ---")
    sample_en_for_url_slash_test = """
    <p>Hello World! Visit http://example.com/path or email test@example.com.
    Say hello/world. Call 123-456-7890. Item 1: • Bullet point. Special @#$/%.
    </p>""" # Added closing </p> for well-formed HTML
    print(f"Original EN for URL/slash test:\n{sample_en_for_url_slash_test}")
    cleaned_en_url_slash_test = default_cleaner.clean_text(sample_en_for_url_slash_test, lang='en')
    print(f"Cleaned EN for URL/slash test:\n{cleaned_en_url_slash_test}\n")
    # Expected: "Hello World! Visit or email. Say hello/world. Call. Item 1:. Bullet point. Special /."

    print("\n--- Arabic Cleaning (Defaults: remove_special_chars=True, remove_urls=True) ---")
    sample_ar_for_url_slash_test = """
    <p>مَرْحَبًا! اذهب إلى http://example.com/صفحة أو راسل test@example.com.
    قل أهلاً/بالعالم. اتصل ٠١٢٣٤٥٦٧٨٩. بند ١: • نقطة. خاص @#$/٪.
    </p>"""
    print(f"Original AR for URL/slash test:\n{sample_ar_for_url_slash_test}")
    if LINGUA_AVAILABLE and PYARABIC_AVAILABLE and CAMELTOOLS_AVAILABLE:
        cleaned_ar_url_slash_test = default_cleaner.clean_text(sample_ar_for_url_slash_test, lang='ar')
        print(f"Cleaned AR for URL/slash test:\n{cleaned_ar_url_slash_test}\n")
        # Expected (approximate, depends on Arabic normalization details):
        # "مرحبا! اذهب الى او راسل. قل اهلا/بالعالم. اتصل. بند ١:. نقطه. خاص /٪."
        # (Assuming ٪ is part of ARABIC_PUNCTUATION or basic Latin punctuation kept.
        # My current ARABIC_PUNCTUATION = r'\u060C\u061B\u061F\u066A\u066B\u066C\u066D' (،؛؟٪٫٬٭) includes ٪ (U+066A ARABIC PERCENT SIGN)
        # The ar_special_chars_keep regex includes ARABIC_PUNCTUATION so ٪ should be kept.
        # So expected: "مرحبا! اذهب الى او راسل. قل اهلا/بالعالم. اتصل. بند ١:. نقطه. خاص /٪."
    else:
        print("Skipping full Arabic test due to missing dependencies. Basic cleaning might still apply.")

    print("\n--- Overriding default to turn OFF remove_special_chars ---")
    cleaner_no_special = TextCleaner(options={'remove_special_chars': False})
    test_text_no_special = "Keep these: a/b c#d. And this url http://example.com"
    print(f"Original for no_special_chars test:\n{test_text_no_special}")
    cleaned_no_special = cleaner_no_special.clean_text(test_text_no_special, lang='en')
    print(f"Cleaned with remove_special_chars=False (URL still removed by default):\n{cleaned_no_special}")
    # Expected: "Keep these: a/b c#d. And this url." (URL removed, but # and / from text stay because remove_special_chars is off)