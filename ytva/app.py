import os
import re
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from googleapiclient.discovery import build
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime

import warnings
import sys

# Silence all the noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['SENTENCE_TRANSFORMERS_VERBOSITY'] = '0'
warnings.filterwarnings('ignore')

from topic_clustering import TopicClustering
from intent_prediction import IntentPredictor
from trend_topic_matching import TrendTopicMatcher
from report_generator import ReportGenerator

report_generator = ReportGenerator()

# Initialize trend topic matcher
trend_matcher = TrendTopicMatcher(similarity_threshold=0.25, use_google_trends=False)
print("✓ Trend topic matcher initialized")

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION AND PATHS
# ─────────────────────────────────────────────────────────────────────────────

# Get absolute paths to ensure files are found
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

BOT_TOKENIZER_PATH   = os.path.join(MODELS_DIR, 'bot_tokenizer.pickle')
TOXIC_TOKENIZER_PATH = os.path.join(MODELS_DIR, 'toxic_tokenizer.pickle')
BOT_MODEL_PATH       = os.path.join(MODELS_DIR, 'YouTube_Bot_Detection_Model.keras')
TOXIC_MODEL_PATH     = os.path.join(MODELS_DIR, 'YouTube_toxic_spam_Detection_Model.keras')

YOUTUBE_API_KEY     = 'AIzaSyBhz8PT4U0plmdrR5rOiwr46mFz4wgMIw4'
MAX_SEQUENCE_LENGTH = 100

# NLP signal keywords - Expanded for better detection
SPAM_PHRASES = [
    'subscribe', 'sub4sub', 'check out my', 'visit my channel', 'follow me',
    'click here', 'free gift', 'giveaway', 'win now', 'limited offer',
    'promo code', 'discount', 'bit.ly', 'tinyurl', 'earn money',
    'make money', 'work from home', 'dm me', 'link in bio', 'free followers',
    'free robux', 'free coins', 'giveaway', 'pinned comment', 'check my profile',
    'like here', '👇', '👆', 'sub back', 'sub4sub', 'sub for sub',
    'join my', 'telegram', 'whatsapp group', 'free money', 'win iphone',
]

TOXIC_PHRASES = [
    'idiot', 'stupid', 'moron', 'hate you', 'kill yourself', 'shut up',
    'loser', 'trash', 'garbage', 'dumb', 'ugly', 'retard', 'cancer',
    'die', 'kys', 'f*ck', 'fuck', 'bitch', 'ass', 'shit', 'pathetic',
    'disgusting', 'hopeless', 'waste of', 'fix your face', 'cry more',
    'nobody likes you', 'delete your channel', 'go jump', 'get punched',
    'lose your job', 'go broke', 'shut your mouth', 'you liar', 'fake tears',
    'racist snake', 'talentless', 'absolute waste', 'punch you', 'kill yourself',
]

# Bot-specific patterns for pattern matching
BOT_PHRASES = [
    # Promotional patterns
    'follow me', 'subscribe', 'check out my', 'visit my channel',
    'link in bio', 'free followers', 'free robux', 'free money',
    'giveaway', 'click here', 'dm me', 'telegram', 'instagram',
    'whatsapp group', 'join my', 'pinned comment', 'check my profile',
    
    # Sub4sub patterns
    'sub4sub', 'sub back', 'sub for sub', 'subscribe to me',
    'sub and sub back', 'sub pls', 'sub please',
    
    # Suspicious patterns
    'free gift', 'win now', 'limited offer', 'promo code',
    'earn money', 'make money', 'work from home',
]

BOT_USERNAME_PATTERNS = [
    r'user\d+', r'bot\d+', r'^[a-z]+\d{4,}$', r'^\d{5,}$',
    r'^[a-zA-Z]{1,2}\d{5,}$', r'^[A-Z]{4,}$', r'.*\d{4,}.*'
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD MODELS AND TOKENIZERS WITH VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

# Load Bot Tokenizer
try:
    with open(BOT_TOKENIZER_PATH, 'rb') as f:
        bot_tokenizer = pickle.load(f)
except Exception as e:
    print(f"✗ Bot tokenizer error: {e}")
    bot_tokenizer = None

# Load Toxic Tokenizer
try:
    with open(TOXIC_TOKENIZER_PATH, 'rb') as f:
        toxic_tokenizer = pickle.load(f)
except Exception as e:
    print(f"✗ Toxic tokenizer error: {e}")
    toxic_tokenizer = None

# Load Bot Model
try:
    if os.path.exists(BOT_MODEL_PATH):
        bot_model = keras.models.load_model(BOT_MODEL_PATH, compile=False)
        print("✓ Bot model loaded")
    else:
        bot_model = None
except:
    bot_model = None

# Load Toxic Model
try:
    toxic_model = keras.models.load_model(TOXIC_MODEL_PATH, compile=False)
    print("✓ Toxic model loaded")
except:
    toxic_model = None

# Initialize topic clustering and intent prediction AFTER models are loaded
topic_clustering = TopicClustering(use_bertopic=True)
intent_predictor = IntentPredictor()
print("✓ Topic clustering and intent prediction modules initialized")

# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def extract_video_id(url):
    """Extract YouTube video ID from various URL formats."""
    match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11}).*', url)
    if match:
        return match.group(1)
    match = re.search(r'youtu\.be/([0-9A-Za-z_-]{11})', url)
    if match:
        return match.group(1)
    return None

def preprocess_text(text):
    """Clean and normalize text for model input."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_signals(text):
    """Extract NLP-based signals from raw comment text."""
    lower = text.lower()
    has_url       = bool(re.search(r'https?://|www\.|bit\.ly|tinyurl', lower))
    spam_signals  = sum(1 for p in SPAM_PHRASES if p in lower)
    toxic_signals = sum(1 for p in TOXIC_PHRASES if p in lower)
    exclamations  = text.count('!')
    
    # Emoji detection
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
        flags=re.UNICODE
    )
    emoji_count   = len(emoji_pattern.findall(text))
    has_emoji_spam = emoji_count >= 3
    
    # ALL CAPS detection
    words = text.split()
    all_caps_words = len([w for w in words if w.isupper() and len(w) > 2])
    all_caps_ratio = all_caps_words / max(len(words), 1)

    return {
        'has_url':        has_url,
        'spam_signals':   spam_signals,
        'toxic_signals':  toxic_signals,
        'exclamations':   exclamations,
        'has_emoji_spam': has_emoji_spam,
        'all_caps':       all_caps_words,
        'all_caps_ratio': all_caps_ratio,
        'word_count':     len(words),
    }

def extract_bot_signals(author_name, comment_text):
    """
    Extract pattern-based signals for bot detection.
    """
    text_lower = comment_text.lower()
    author_lower = author_name.lower()
    
    # Count bot phrases
    bot_signals = sum(1 for p in BOT_PHRASES if p in text_lower)
    
    # Check for suspicious username patterns
    suspicious_username = any(re.search(p, author_name) for p in BOT_USERNAME_PATTERNS)
    
    # Check for URLs
    has_url = bool(re.search(r'https?://|www\.|bit\.ly|tinyurl', text_lower))
    
    # Check for excessive punctuation
    exclamations = comment_text.count('!')
    questions = comment_text.count('?')
    excessive_punctuation = exclamations > 2 or questions > 2
    
    # ALL CAPS detection
    words = comment_text.split()
    all_caps_words = len([w for w in words if w.isupper() and len(w) > 2])
    
    # Short comment
    is_short = len(words) < 4
    
    # Username characteristics
    username_digits = sum(c.isdigit() for c in author_name)
    digit_ratio = username_digits / max(len(author_name), 1)
    username_all_digits = username_digits == len(author_name) and len(author_name) > 2
    
    # Calculate strong indicators (weighted like toxic model)
    strong_indicators = 0
    
    # Each of these counts as a strong indicator
    if suspicious_username:
        strong_indicators += 1
    if has_url:
        strong_indicators += 1
    if bot_signals >= 2:
        strong_indicators += 1
    if username_all_digits:
        strong_indicators += 1
    if all_caps_words > 2:
        strong_indicators += 1
    if 'sub4sub' in text_lower or 'sub back' in text_lower:
        strong_indicators += 1
    if 'free followers' in text_lower or 'free money' in text_lower:
        strong_indicators += 1

    return {
        'bot_signals': bot_signals,
        'suspicious_username': suspicious_username,
        'has_url': has_url,
        'exclamations': exclamations,
        'questions': questions,
        'excessive_punctuation': excessive_punctuation,
        'all_caps': all_caps_words,
        'is_short': is_short,
        'username_digits': username_digits,
        'digit_ratio': round(digit_ratio, 2),
        'username_all_digits': username_all_digits,
        'strong_indicators': strong_indicators,
        'total_score': strong_indicators * 25 + bot_signals * 5
    }

def predict_toxic_spam(text):
    """Enhanced prediction with better error handling and model verification."""
    
    # Check if model is loaded
    if toxic_model is None or toxic_tokenizer is None:
        print("⚠ Toxic model/tokenizer not loaded, using rule-based fallback")
        return _rule_based_classify_enhanced(text)

    try:
        # Preprocess the text exactly like training
        clean = preprocess_text(text)
        
        # Convert to sequence
        seq = toxic_tokenizer.texts_to_sequences([clean])
        
        # Check if sequence is empty (all words unknown)
        if len(seq[0]) == 0:
            # print("⚠ No known words in text, using rule-based")
            return _rule_based_classify_enhanced(text)
        
        # Pad sequence
        pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        
        # Make prediction
        pred = toxic_model.predict(pad, verbose=0)[0]

        # Get NLP signals for enhancement
        signals = extract_signals(text)

        # Handle different model output shapes
        if len(pred) == 1:
            # Binary model (sigmoid output)
            toxic_prob = float(pred[0])
            
            # Determine label based on probability AND signals
            if toxic_prob > 0.5:
                if signals.get('toxic_signals', 0) > 0:
                    label = "toxic"
                elif signals.get('spam_signals', 0) > 0 or signals.get('has_url', False):
                    label = "spam"
                else:
                    label = "toxic"  # Default to toxic if model says so
            else:
                if signals.get('spam_signals', 0) > 0 or signals.get('has_url', False):
                    label = "spam"
                    toxic_prob = 0.7  # Override probability for spam
                elif signals.get('toxic_signals', 0) > 0:
                    label = "toxic"
                    toxic_prob = 0.6  # Override for toxic
                else:
                    label = "normal"
            
            # Calculate confidence with signal boost
            confidence = toxic_prob * 100
            if label == 'toxic' and signals.get('toxic_signals', 0) > 0:
                confidence = min(99, confidence + 10)
            elif label == 'spam' and (signals.get('spam_signals', 0) > 0 or signals.get('has_url', False)):
                confidence = min(99, confidence + 10)
            
            # Distribute probabilities
            if label == 'normal':
                return {
                    'label': 'normal',
                    'confidence': round(confidence, 1),
                    'probabilities': {
                        'normal': round(confidence, 1),
                        'spam': round((100 - confidence) * 0.6, 1),
                        'toxic': round((100 - confidence) * 0.4, 1)
                    }
                }
            elif label == 'spam':
                return {
                    'label': 'spam',
                    'confidence': round(confidence, 1),
                    'probabilities': {
                        'normal': round((100 - confidence) * 0.3, 1),
                        'spam': round(confidence, 1),
                        'toxic': round((100 - confidence) * 0.7, 1)
                    }
                }
            else:  # toxic
                return {
                    'label': 'toxic',
                    'confidence': round(confidence, 1),
                    'probabilities': {
                        'normal': round((100 - confidence) * 0.2, 1),
                        'spam': round((100 - confidence) * 0.3, 1),
                        'toxic': round(confidence, 1)
                    }
                }

        # If model outputs MULTI-CLASS (normal/spam/toxic)
        elif len(pred) >= 3:
            idx = int(np.argmax(pred))
            labels = ['normal', 'spam', 'toxic']
            conf = float(pred[idx]) * 100

            # Apply NLP signal boost for better accuracy
            signals = extract_signals(text)
            
            # Override if signals strongly indicate a different category
            if signals.get('toxic_signals', 0) >= 2 and labels[idx] != 'toxic':
                # Strong toxic signals should override
                toxic_idx = 2
                conf = max(conf, 85)
                return {
                    'label': 'toxic',
                    'confidence': round(conf, 1),
                    'probabilities': {
                        'normal': round(float(pred[0]) * 100 * 0.3, 1),
                        'spam': round(float(pred[1]) * 100 * 0.3, 1),
                        'toxic': round(conf, 1)
                    }
                }
            elif signals.get('spam_signals', 0) >= 2 and signals.get('has_url', False) and labels[idx] != 'spam':
                # Strong spam signals should override
                conf = max(conf, 85)
                return {
                    'label': 'spam',
                    'confidence': round(conf, 1),
                    'probabilities': {
                        'normal': round(float(pred[0]) * 100 * 0.3, 1),
                        'spam': round(conf, 1),
                        'toxic': round(float(pred[2]) * 100 * 0.3, 1)
                    }
                }
            
            # Boost confidence if signals match prediction
            if labels[idx] == 'toxic' and signals.get('toxic_signals', 0) > 0:
                conf = min(99, conf + 10)
            elif labels[idx] == 'spam' and (signals.get('spam_signals', 0) > 0 or signals.get('has_url', False)):
                conf = min(99, conf + 10)
            elif labels[idx] == 'normal' and signals.get('word_count', 0) > 10:
                conf = min(99, conf + 5)  # Longer comments more likely genuine
            
            return {
                'label': labels[idx],
                'confidence': round(conf, 1),
                'probabilities': {
                    'normal': round(float(pred[0]) * 100, 1),
                    'spam': round(float(pred[1]) * 100, 1),
                    'toxic': round(float(pred[2]) * 100, 1)
                }
            }
        
        else:
            print(f"⚠ Unexpected model output shape: {len(pred)}")
            return _rule_based_classify_enhanced(text)

    except Exception as e:
        print(f"❌ Toxic prediction error: {e}")
        import traceback
        traceback.print_exc()
        return _rule_based_classify_enhanced(text)

def _rule_based_classify_enhanced(text):
    """Enhanced fallback classifier with better pattern matching."""
    lower = text.lower()
    signals = extract_signals(text)
    
    # Check for toxic content first (priority)
    if signals['toxic_signals'] >= 1 or any(phrase in lower for phrase in [
        'lose your job', 'go broke', 'hopeless', 'pathetic', 'disgusting',
        'waste of', 'fix your face', 'cry more', 'shut your mouth',
        'nobody likes you', 'delete your', 'go jump', 'get punched',
        'you liar', 'fake tears', 'racist snake', 'talentless'
    ]):
        # Calculate confidence based on signal strength
        base_conf = 75
        conf = base_conf + (signals['toxic_signals'] * 5) + (signals['exclamations'] * 2)
        conf = min(98, conf)
        
        # Check if it's actually spam masquerading as toxic
        if signals['spam_signals'] > signals['toxic_signals']:
            return {
                'label': 'spam',
                'confidence': min(90, 65 + signals['spam_signals'] * 5),
                'probabilities': {
                    'normal': 10,
                    'spam': 85,
                    'toxic': 5
                }
            }
        
        return {
            'label': 'toxic',
            'confidence': round(conf, 1),
            'probabilities': {
                'normal': round(max(2, 100 - conf - 10), 1),
                'spam': 5,
                'toxic': round(conf, 1)
            }
        }
    
    # Check for spam
    elif (signals['spam_signals'] >= 1 or signals['has_url'] or 
          signals['all_caps_ratio'] > 0.3 or signals['exclamations'] > 3 or
          signals['has_emoji_spam']):
        
        conf = 65
        conf += signals['spam_signals'] * 6
        conf += 15 if signals['has_url'] else 0
        conf += 10 if signals['all_caps_ratio'] > 0.3 else 0
        conf += 5 if signals['exclamations'] > 3 else 0
        conf += 8 if signals['has_emoji_spam'] else 0
        conf = min(95, conf)
        
        return {
            'label': 'spam',
            'confidence': round(conf, 1),
            'probabilities': {
                'normal': round(max(5, 100 - conf - 10), 1),
                'spam': round(conf, 1),
                'toxic': round(max(2, 100 - conf - 5), 1)
            }
        }
    
    # Normal comment
    else:
        # Longer, more complex comments are more likely to be genuine
        if signals['word_count'] > 10:
            conf = 90
        elif signals['word_count'] > 5:
            conf = 85
        else:
            conf = 75
            
        return {
            'label': 'normal',
            'confidence': round(conf, 1),
            'probabilities': {
                'normal': round(conf, 1),
                'spam': round((100 - conf) * 0.6, 1),
                'toxic': round((100 - conf) * 0.4, 1)
            }
        }

def predict_bot(author_name, comment_text):
    """
    Predict if a comment is from a bot using trained model + comprehensive pattern matching.
    All logic happens here in the backend.
    """
    
    text_lower = comment_text.lower()
    author_lower = author_name.lower()
    
    # ===== INITIALIZE SCORE =====
    bot_score = 0
    reasons = []
    strong_indicators = 0
    
    # ===== STRONG BOT INDICATORS (70-90%) =====
    
    # Free followers pattern (YOUR EXAMPLE)
    if 'free followers' in text_lower:
        bot_score = max(bot_score, 85)
        reasons.append('free followers scam')
        strong_indicators += 2
    
    # Sub4sub patterns
    if 'sub4sub' in text_lower or 'sub back' in text_lower or 'sub for sub' in text_lower:
        bot_score = max(bot_score, 80)
        reasons.append('sub4sub pattern')
        strong_indicators += 2
    
    # Free money pattern
    if 'free money' in text_lower or 'earn money' in text_lower or 'make money' in text_lower:
        bot_score = max(bot_score, 85)
        reasons.append('money scam')
        strong_indicators += 2
    
    # Link in bio pattern
    if 'link in bio' in text_lower:
        bot_score = max(bot_score, 75)
        reasons.append('link in bio promotion')
        strong_indicators += 1
    
    # Giveaway patterns
    if 'giveaway' in text_lower or 'win now' in text_lower or 'free gift' in text_lower:
        bot_score = max(bot_score, 75)
        reasons.append('giveaway scam')
        strong_indicators += 1
    
    # ===== MEDIUM BOT INDICATORS (50-69%) =====
    
    # Multiple promotional phrases
    promo_count = 0
    if 'follow me' in text_lower: promo_count += 1
    if 'subscribe' in text_lower: promo_count += 1
    if 'check out' in text_lower: promo_count += 1
    if 'visit my' in text_lower: promo_count += 1
    if 'my channel' in text_lower: promo_count += 1
    
    if promo_count >= 3:
        bot_score = max(bot_score, 70)
        reasons.append(f'multiple promotions ({promo_count})')
        strong_indicators += 1
    elif promo_count >= 2:
        bot_score = max(bot_score, 60)
        reasons.append(f'promotional content ({promo_count})')
    
    # URL present
    if re.search(r'https?://|www\.|bit\.ly|tinyurl', text_lower):
        if bot_score < 65:
            bot_score = 60
        reasons.append('contains URL')
        strong_indicators += 1
    
    # Suspicious username patterns
    if re.search(r'user\d{4,}|bot\d{3,}', author_lower):
        bot_score = max(bot_score, 55)
        reasons.append('bot-like username')
        strong_indicators += 1
    elif len(author_name) < 3:
        bot_score = max(bot_score, 50)
        reasons.append('very short username')
    elif author_name.isdigit():
        bot_score = max(bot_score, 60)
        reasons.append('numeric username')
        strong_indicators += 1
    
    # Excessive punctuation
    if '!!!' in comment_text or '???' in comment_text:
        bot_score = max(bot_score, 45)
        reasons.append('excessive punctuation')
    
    # ALL CAPS words
    words = comment_text.split()
    all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
    if all_caps_words >= 2:
        bot_score = max(bot_score, 50)
        reasons.append(f'all caps ({all_caps_words} words)')
    
    # Very short comment (potential bot)
    if len(words) < 3 and bot_score < 40:
        bot_score = 35
        reasons.append('very short comment')
    
    # ===== TRY ML MODEL IF AVAILABLE =====
    ml_prob = 0
    if bot_model is not None and bot_tokenizer is not None:
        try:
            combined = preprocess_text(f"{author_name} {comment_text}")
            seq = bot_tokenizer.texts_to_sequences([combined])
            if len(seq[0]) > 0:
                pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
                prediction = bot_model.predict(pad, verbose=0)
                if len(prediction[0]) == 1:
                    ml_prob = float(prediction[0][0]) * 100
                # If ML gives high confidence, use it
                if ml_prob > 70:
                    bot_score = ml_prob
                    reasons.append(f'ML model: {ml_prob:.0f}%')
        except Exception as e:
            print(f"ML error: {e}")
    
    # ===== FINAL ADJUSTMENTS =====
    
    # If no indicators at all
    if bot_score == 0:
        bot_score = 10
        reasons.append('normal comment')
    
    # Ensure score is within bounds
    bot_score = min(99, bot_score)
    
    # Determine label based on score
    if bot_score >= 70:
        label = "BOT"
    elif bot_score >= 50:
        label = "SUSPICIOUS"
    elif bot_score >= 30:
        label = "MAYBE BOT"
    else:
        label = "HUMAN"
    
    return {
        'is_bot': bot_score > 50,
        'label': label,
        'bot_probability': round(bot_score),
        'ml_probability': round(ml_prob),
        'strong_indicators': strong_indicators,
        'reasons': reasons[:3],  # Top 3 reasons
        'promo_count': promo_count,
        'model_loaded': bot_model is not None
    }

def analyze_comment(comment_data):
    """Main function to analyze a single comment."""
    text = comment_data.get('text', '')
    author = comment_data.get('author', '')

    # Run ML models
    toxic_result = predict_toxic_spam(text)
    bot_result = predict_bot(author, text)

    # Extract NLP rule-based signals
    signals = extract_signals(text)

    # Get final label and confidence
    label = toxic_result.get('label', 'normal')
    confidence = toxic_result.get('confidence', 82.0)

    # Ensure confidence is reasonable
    if label == 'normal':
        confidence = max(confidence, 75)
    elif label == 'spam':
        confidence = max(confidence, 70)
    elif label == 'toxic':
        confidence = max(confidence, 75)

    return {
        'text': text,
        'author': author,
        'published_at': comment_data.get('published_at', ''),
        'likes': comment_data.get('likes', 0),
        'likeCount': comment_data.get('likes', 0),
        'label': label,
        'category': label,
        'confidence': confidence,
        'is_toxic': label == 'toxic',
        'is_spam': label == 'spam',
        'is_bot': bot_result.get('is_bot', False),
        'probabilities': toxic_result.get('probabilities', {}),
        'ml_probabilities': toxic_result.get('probabilities', {}),
        'bot_probability': bot_result.get('bot_probability', 0),
        'signals': signals
    }

def get_video_title(video_id):
    """Fetch video title from YouTube API"""
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.videos().list(
            part='snippet',
            id=video_id
        )
        response = request.execute()
        
        if response.get('items'):
            return response['items'][0]['snippet']['title']
        return f"Video {video_id}"
    except Exception as e:
        print(f"Error fetching title: {e}")
        return f"Video {video_id}"

# ─────────────────────────────────────────────────────────────────────────────
# 4. ROUTES — HTML PAGES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')

@app.route('/spam')
def spam_page():
    return render_template('toxic_spam.html')

@app.route('/bots')
def bots_page():
    return render_template('bots.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/topics')
def topics_page():
    return render_template('topics.html')

@app.route('/intents')
def intents_page():
    return render_template('intents.html')

@app.route('/trends')
def trends_page():
    return render_template('trends.html')

@app.route('/report')
def report_page():
    return render_template('report.html')

# ─────────────────────────────────────────────────────────────────────────────
# 5. API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Analyze all comments from a YouTube video."""
    try:
        data = request.get_json()
        video_url = data.get('video_url')

        if not video_url:
            return jsonify({'error': 'No video URL provided'}), 400

        video_id = extract_video_id(video_url)
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL format'}), 400
        
        video_title = get_video_title(video_id)
        
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        api_request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            order='relevance'
        )
        response = api_request.execute()

        results = []
        counts = {'normal': 0, 'spam': 0, 'toxic': 0, 'bots': 0}

        for item in response.get('items', []):
            snippet = item['snippet']['topLevelComment']['snippet']
            comment_data = {
                'text': snippet['textDisplay'],
                'author': snippet['authorDisplayName'],
                'published_at': snippet['publishedAt'],
                'likes': snippet.get('likeCount', 0),
            }
            analysis = analyze_comment(comment_data)
            results.append(analysis)
            counts[analysis['label']] += 1
            if analysis['is_bot']:
                counts['bots'] += 1

        total = len(results)
        bot_count = counts['bots']
        human_count = total - bot_count

        summary = {
            'total_comments': total,
            'normal_count': counts['normal'],
            'spam_count': counts['spam'],
            'toxic_count': counts['toxic'],
            'bot_count': bot_count,
            'human_count': human_count,
            'normal_pct': round(counts['normal'] / total * 100, 1) if total else 0,
            'spam_pct': round(counts['spam'] / total * 100, 1) if total else 0,
            'toxic_pct': round(counts['toxic'] / total * 100, 1) if total else 0,
            'bot_pct': round(bot_count / total * 100, 1) if total else 0,
            'human_pct': round(human_count / total * 100, 1) if total else 0,
        }

        return jsonify({
            'status': 'success',
            'video_id': video_id,
            'video_title': video_title,
            'summary': summary,
            'comments': results,
        })

    except Exception as e:
        print(f"Analyze error: {e}")
        return jsonify({'error': f'Failed to analyze video: {str(e)}'}), 500

@app.route('/api/classify_single', methods=['POST'])
def classify_single():
    """Live single-comment classifier used by Spam page."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        toxic_result = predict_toxic_spam(text)
        signals      = extract_signals(text)

        # Hybrid-aware label & confidence
        label = toxic_result.get('label', 'normal')
        confidence = toxic_result['probabilities'].get(label, 0)

        if signals.get('toxic_signals', 0) >= 1:
            label = 'toxic'
            confidence = max(confidence, 85)
        elif signals.get('spam_signals', 0) >= 1:
            label = 'spam'
            confidence = max(confidence, 80)
        if label == 'normal':
            confidence = max(confidence, 82)

        # Normalize probabilities so total = 100%
        remainder = 100 - confidence
        others = [k for k in ['normal','spam','toxic'] if k != label]
        ml_probabilities = {k: remainder/2 if k in others else confidence for k in ['normal','spam','toxic']}

        return jsonify({
            'label': label,
            'confidence': confidence,
            'ml_probabilities': ml_probabilities,
            'signals': signals,
        })

    except Exception as e:
        print(f"classify_single error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = {
        'bot_tokenizer': bot_tokenizer is not None,
        'toxic_tokenizer': toxic_tokenizer is not None,
        'bot_model': bot_model is not None,
        'toxic_model': toxic_model is not None,
    }
    return jsonify({
        'status': 'healthy' if all(status.values()) else 'degraded',
        'models': status,
    })

@app.route('/api/test_bot', methods=['POST'])
def test_bot():
    """Test the bot model directly."""
    try:
        data = request.get_json()
        author = data.get('author', '')
        text = data.get('text', '')
        
        if not author or not text:
            return jsonify({'error': 'Author and text required'}), 400
        
        result = predict_bot(author, text)
        
        return jsonify({
            'author': author,
            'text': text,
            'result': result,
            'model_loaded': bot_model is not None,
            'tokenizer_loaded': bot_tokenizer is not None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ping', methods=['GET'])
def ping():
    """Simple endpoint to test if server is running."""
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'bot_model_loaded': bot_model is not None,
        'toxic_model_loaded': toxic_model is not None
    })

@app.route('/api/debug_bot_model', methods=['POST'])
def debug_bot_model():
    """Debug endpoint to see raw bot model output."""
    try:
        data = request.get_json()
        author = data.get('author', 'TestUser')
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text required'}), 400
        
        # Get raw ML prediction
        ml_prob = 0
        raw_prediction = None
        
        if bot_model is not None and bot_tokenizer is not None:
            combined = preprocess_text(f"{author} {text}")
            seq = bot_tokenizer.texts_to_sequences([combined])
            if len(seq[0]) > 0:
                pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
                raw_prediction = bot_model.predict(pad, verbose=0).tolist()
                if len(raw_prediction[0]) == 1:
                    ml_prob = float(raw_prediction[0][0]) * 100
        
        # Get pattern-based result
        pattern_result = predict_bot(author, text)
        
        return jsonify({
            'text': text,
            'author': author,
            'raw_ml_prediction': raw_prediction,
            'ml_probability': ml_prob,
            'pattern_result': pattern_result,
            'tokenizer_vocab': len(bot_tokenizer.word_index) if bot_tokenizer else 0,
            'model_loaded': bot_model is not None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# ─────────────────────────────────────────────────────────────────────────────
# 5. TREND TOPIC MATCHING ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/trend_evaluate', methods=['POST'])
def trend_evaluate():
    """Evaluate growth potential for topics"""
    try:
        data = request.get_json()
        topics = data.get('topics', [])
        
        if not topics:
            return jsonify({'error': 'No topics provided'}), 400
        
        result = trend_matcher.evaluate_topic_growth(topics)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────────────────────────────────────────
# 6. NEW API ENDPOINTS - Topic Clustering & Intent Prediction
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/topic_clustering', methods=['POST'])
def analyze_topics():
    """Perform topic clustering on comments"""
    try:
        data = request.get_json()
        comments = data.get('comments', [])
        
        if not comments:
            return jsonify({'error': 'No comments provided'}), 400
        
        # Extract comment texts
        comment_texts = [c.get('text', '') for c in comments if c.get('text')]
        
        # Perform clustering
        result = topic_clustering.cluster_comments(
            comment_texts, 
            min_topic_size=max(3, len(comment_texts)//10)  # Dynamic min size
        )
        
        if not result['success']:
            return jsonify({'error': result.get('error', 'Clustering failed')}), 500
        
        # Get summary
        summary = topic_clustering.get_topic_summary(comment_texts, result['assignments'])
        
        # Add sample comments to topics for display
        topics_with_samples = []
        for topic in result['topics']:
            topic_id = str(topic['id'])
            sample_indices = [a['comment_idx'] for a in result['assignments'] 
                            if str(a['topic']) == topic_id][:3]
            samples = [comment_texts[idx] for idx in sample_indices if idx < len(comment_texts)]
            topic['sample_comments'] = samples
            topics_with_samples.append(topic)
        
        return jsonify({
            'success': True,
            'method': result['method'],
            'total_comments': result['total_comments'],
            'num_topics': result['num_topics'],
            'topics': topics_with_samples,
            'topic_counts': result['topic_counts'],
            'summary': summary
        })
        
    except Exception as e:
        print(f"Topic clustering error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/intent_prediction', methods=['POST'])
def predict_intents():
    """Predict intents for comments"""
    try:
        data = request.get_json()
        comments = data.get('comments', [])
        
        if not comments:
            return jsonify({'error': 'No comments provided'}), 400
        
        # Extract comment texts
        comment_texts = [c.get('text', '') for c in comments if c.get('text')]
        
        # Predict intents
        predictions = intent_predictor.predict_batch_intents(comment_texts)
        
        # Get summary
        summary = intent_predictor.get_intent_summary(predictions)
        
        # Add predictions to comments
        comments_with_intent = []
        for i, comment in enumerate(comments):
            if i < len(predictions):
                comment_copy = comment.copy()
                comment_copy['intent'] = predictions[i]
                comments_with_intent.append(comment_copy)
        
        return jsonify({
            'success': True,
            'total_comments': len(predictions),
            'summary': summary,
            'predictions': predictions,
            'comments': comments_with_intent
        })
        
    except Exception as e:
        print(f"Intent prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze_advanced', methods=['POST'])
def analyze_advanced():
    """Full advanced analysis: topics + intents on analyzed comments"""
    try:
        data = request.get_json()
        # Expect analyzed comments from previous analysis
        comments = data.get('comments', [])
        
        if not comments:
            return jsonify({'error': 'No comments provided'}), 400
        
        # Extract texts
        comment_texts = [c.get('text', '') for c in comments if c.get('text')]
        
        # Run topic clustering
        topic_result = topic_clustering.cluster_comments(
            comment_texts,
            min_topic_size=max(3, len(comment_texts)//10)
        )
        
        # Run intent prediction
        intent_results = intent_predictor.predict_batch_intents(comment_texts)
        
        # Combine results
        enhanced_comments = []
        for i, comment in enumerate(comments):
            if i < len(intent_results):
                comment_copy = comment.copy()
                comment_copy['intent'] = intent_results[i]
                
                # Add topic assignment if available
                if topic_result['success'] and i < len(topic_result.get('assignments', [])):
                    assignment = topic_result['assignments'][i]
                    comment_copy['topic'] = assignment['topic']
                    comment_copy['topic_probability'] = assignment.get('probability', 0)
                
                enhanced_comments.append(comment_copy)
        
        return jsonify({
            'success': True,
            'intent_summary': intent_predictor.get_intent_summary(intent_results) if intent_results else {},
            'topic_summary': topic_result if topic_result['success'] else None,
            'comments': enhanced_comments
        })
        
    except Exception as e:
        print(f"Advanced analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    """Generate comprehensive report of analysis"""
    try:
        data = request.get_json()
        analysis_data = data.get('analysis_data')
        report_type = data.get('report_type', 'html')  # html, csv, json, all
        
        if not analysis_data:
            return jsonify({'error': 'No analysis data provided'}), 400
        
        video_id = analysis_data.get('video_id', 'unknown')
        files = []
        
        if report_type == 'html' or report_type == 'all':
            html_path = report_generator.generate_html_report(analysis_data)
            files.append({'type': 'html', 'path': html_path, 'filename': os.path.basename(html_path)})
        
        if report_type == 'csv' or report_type == 'all':
            csv_path = report_generator.generate_csv_report(analysis_data)
            files.append({'type': 'csv', 'path': csv_path, 'filename': os.path.basename(csv_path)})
        
        if report_type == 'json' or report_type == 'all':
            json_path = report_generator.generate_json_report(analysis_data)
            files.append({'type': 'json', 'path': json_path, 'filename': os.path.basename(json_path)})
        
        return jsonify({
            'success': True,
            'message': f'Report(s) generated successfully',
            'files': files,
            'report_dir': 'reports/'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_report/<filename>', methods=['GET'])
def download_report(filename):
    """Download generated report file"""
    try:
        from flask import send_file
        filepath = os.path.join('reports', filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/list_reports', methods=['GET'])
def list_reports():
    """List all generated reports"""
    try:
        reports_dir = 'reports'
        if not os.path.exists(reports_dir):
            return jsonify({'reports': []})
        
        reports = []
        for file in os.listdir(reports_dir):
            filepath = os.path.join(reports_dir, file)
            reports.append({
                'filename': file,
                'size': os.path.getsize(filepath),
                'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
            })
        
        # Sort by modified date (newest first)
        reports.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({'reports': reports})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/match_trends', methods=['POST'])
def match_trends():
    """Match topics with trends for dashboard"""
    try:
        data = request.get_json()
        topics = data.get('topics', [])
        
        if not topics:
            return jsonify({'success': False, 'error': 'No topics'}), 400
        
        matches = []
        for topic in topics:
            matches.append({
                'topic_name': topic.get('name', ''),
                'match_score': topic.get('size', 0)
            })
        
        return jsonify({
            'success': True,
            'matches': matches
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────────────────────────────────────────
# 6. RUN SERVER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  YouTube Video Analyzer — Backend Server")
    print("  Dept. of Software Engineering, UAJK  |  FYP 2025-26")
    print("=" * 60)

    # Check all 6 modules
    models = [
        ("Bot Tokenizer", bot_tokenizer is not None),
        ("Toxic Tokenizer", toxic_tokenizer is not None),
        ("Bot Model", bot_model is not None),
        ("Toxic Model", toxic_model is not None),
        ("Topic Clustering", topic_clustering is not None),
        ("Intent Prediction", intent_predictor is not None),
        ("Trend Matching", trend_matcher is not None),
    ]
    
    print("\nModel Loading Status:")
    all_loaded = True
    for name, ok in models:
        status = "✓ LOADED" if ok else "✗ FAILED"
        # Format with consistent spacing
        print(f"  {name:20s}: {status}")
        if not ok:
            all_loaded = False

    # Count loaded modules
    loaded_count = sum(1 for _, ok in models if ok)
    total_count = len(models)

    if not all_loaded:
        print(f"\n⚠  WARNING: {total_count - loaded_count}/{total_count} models failed to load!")
        print("   Check the paths above and ensure model files exist.")
        print("   Using enhanced rule-based fallback system for failed modules.\n")
    else:
        print(f"\n✓ All models loaded successfully!")
        print("  System running at full capacity.\n")

    print("=" * 60)
    print("  → Server URL: http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(debug=True, port=5000, host='0.0.0.0')


