from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ar": "Arabic",
    "ru": "Russian",
    "zh-cn": "Chinese",
}

def detect_language(text):
    try:
        code = detect(text)
        return LANGUAGES.get(code, code.upper())
    except Exception:
        return "Unknown"
