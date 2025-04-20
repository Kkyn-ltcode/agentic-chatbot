"""
Language detection for documents.
"""
from typing import Dict, List, Any, Optional
from langchain.schema import Document
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("language_detector")

try:
    import langdetect
    from langdetect import detect, DetectorFactory
    # Set seed for consistent results
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not installed. Using fallback language detection.")

# Language name mapping
LANGUAGE_NAMES = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'nl': 'Dutch',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'pa': 'Punjabi',
    'te': 'Telugu',
    'mr': 'Marathi',
    'ta': 'Tamil',
    'ur': 'Urdu',
    'fa': 'Persian',
    'tr': 'Turkish',
    'vi': 'Vietnamese',
    'th': 'Thai',
    'id': 'Indonesian',
    'ms': 'Malay',
    'sv': 'Swedish',
    'no': 'Norwegian',
    'da': 'Danish',
    'fi': 'Finnish',
    'pl': 'Polish',
    'cs': 'Czech',
    'sk': 'Slovak',
    'hu': 'Hungarian',
    'ro': 'Romanian',
    'bg': 'Bulgarian',
    'el': 'Greek',
    'he': 'Hebrew',
    'uk': 'Ukrainian',
    'ca': 'Catalan',
    'eu': 'Basque',
    'gl': 'Galician',
    'et': 'Estonian',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'sl': 'Slovenian',
    'hr': 'Croatian',
    'sr': 'Serbian',
    'mk': 'Macedonian',
    'sq': 'Albanian',
    'hy': 'Armenian',
    'ka': 'Georgian',
    'mt': 'Maltese',
    'cy': 'Welsh',
    'is': 'Icelandic',
    'af': 'Afrikaans',
    'sw': 'Swahili',
    'tl': 'Tagalog',
    'la': 'Latin',
    'gd': 'Scottish Gaelic',
    'ga': 'Irish',
    'mn': 'Mongolian',
    'yi': 'Yiddish',
    'kk': 'Kazakh',
    'uz': 'Uzbek',
    'az': 'Azerbaijani',
    'be': 'Belarusian',
    'km': 'Khmer',
    'lo': 'Lao',
    'my': 'Burmese',
    'ne': 'Nepali',
    'si': 'Sinhala',
    'ml': 'Malayalam',
    'kn': 'Kannada',
    'gu': 'Gujarati',
    'or': 'Odia',
    'as': 'Assamese',
    'bo': 'Tibetan',
    'ug': 'Uyghur',
    'jv': 'Javanese',
    'su': 'Sundanese',
    'mg': 'Malagasy',
    'so': 'Somali',
    'am': 'Amharic',
    'ti': 'Tigrinya',
    'zu': 'Zulu',
    'xh': 'Xhosa',
    'st': 'Sesotho',
    'tn': 'Tswana',
    'sn': 'Shona',
    'yo': 'Yoruba',
    'ig': 'Igbo',
    'ha': 'Hausa',
    'lb': 'Luxembourgish',
    'fy': 'Frisian',
    'oc': 'Occitan',
    'br': 'Breton',
    'rm': 'Romansh',
    'co': 'Corsican',
    'ht': 'Haitian Creole',
    'ku': 'Kurdish',
    'ky': 'Kyrgyz',
    'tk': 'Turkmen',
    'tt': 'Tatar',
    'ps': 'Pashto',
    'sd': 'Sindhi',
    'dv': 'Dhivehi',
    'tg': 'Tajik',
}

class LanguageDetector:
    """Detect language of documents."""
    
    def __init__(self, 
                 min_length: int = 50,
                 default_language: str = "en",
                 confidence_threshold: float = 0.5):
        """
        Initialize the language detector.
        
        Args:
            min_length: Minimum text length for reliable detection
            default_language: Default language code to use if detection fails
            confidence_threshold: Minimum confidence threshold for detection
        """
        self.min_length = min_length
        self.default_language = default_language
        self.confidence_threshold = confidence_threshold
    
    def detect_language(self, documents: List[Document]) -> List[Document]:
        """
        Detect language of documents.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of documents with language metadata
        """
        processed_docs = []
        
        for doc in documents:
            # Create a copy of the metadata
            metadata = doc.metadata.copy()
            
            # Skip if language is already detected
            if "language" in metadata and "language_code" in metadata:
                processed_docs.append(doc)
                continue
            
            # Get text content
            text = doc.page_content
            
            # Detect language
            lang_code, confidence = self._detect_language(text)
            
            # Add language metadata
            metadata["language_code"] = lang_code
            metadata["language"] = LANGUAGE_NAMES.get(lang_code, "Unknown")
            metadata["language_confidence"] = confidence
            
            # Create a new document with language metadata
            processed_docs.append(Document(
                page_content=doc.page_content,
                metadata=metadata
            ))
        
        return processed_docs
    
    def _detect_language(self, text: str) -> tuple:
        """
        Detect language of text.
        
        Args:
            text: Text to detect language
            
        Returns:
            Tuple of (language_code, confidence)
        """
        # Check if text is long enough for reliable detection
        if len(text.strip()) < self.min_length:
            return self.default_language, 0.0
        
        # Use langdetect if available
        if LANGDETECT_AVAILABLE:
            try:
                # Get language probabilities
                lang_probs = langdetect.detect_langs(text)
                
                if lang_probs:
                    # Get the most probable language
                    lang_code = lang_probs[0].lang
                    confidence = lang_probs[0].prob
                    
                    # Check confidence threshold
                    if confidence >= self.confidence_threshold:
                        return lang_code, confidence
                    else:
                        return self.default_language, confidence
                
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
        
        # Fallback to simple heuristics
        return self._fallback_detection(text)
    
    def _fallback_detection(self, text: str) -> tuple:
        """
        Fallback language detection using simple heuristics.
        
        Args:
            text: Text to detect language
            
        Returns:
            Tuple of (language_code, confidence)
        """
        # Simple character-based heuristics
        text = text.lower()
        
        # Check for Chinese characters
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh', 0.8
        
        # Check for Japanese characters
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return 'ja', 0.8
        
        # Check for Korean characters
        if re.search(r'[\uac00-\ud7af]', text):
            return 'ko', 0.8
        
        # Check for Arabic characters
        if re.search(r'[\u0600-\u06ff]', text):
            return 'ar', 0.8
        
        # Check for Cyrillic characters (Russian, etc.)
        if re.search(r'[\u0400-\u04ff]', text):
            return 'ru', 0.7
        
        # Check for common Spanish words
        spanish_words = ['el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'pero', 'porque', 'como', 'cuando', 'donde', 'quien']
        spanish_count = sum(1 for word in spanish_words if f' {word} ' in f' {text} ')
        
        # Check for common French words
        french_words = ['le', 'la', 'les', 'un', 'une', 'et', 'ou', 'mais', 'car', 'comme', 'quand', 'où', 'qui', 'ce', 'cette']
        french_count = sum(1 for word in french_words if f' {word} ' in f' {text} ')
        
        # Check for common German words
        german_words = ['der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'aber', 'weil', 'wie', 'wenn', 'wo', 'wer', 'was']
        german_count = sum(1 for word in german_words if f' {word} ' in f' {text} ')
        
        # Determine language based on word counts
        counts = {
            'es': spanish_count,
            'fr': french_count,
            'de': german_count
        }
        
        max_lang = max(counts, key=counts.get)
        max_count = counts[max_lang]
        
        # If we have a significant number of matches, return that language
        if max_count > 3:
            return max_lang, 0.6
        
        # Default to English
        return self.default_language, 0.5