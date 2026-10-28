"""
Trend Topic Matching Module using Semantic Similarity
Evaluates growth potential by matching topics with trending themes using Sentence-BERT embeddings
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except ImportError:
    HAS_PYTRENDS = False
    print("⚠ pytrends not available - using simulated trend scores")


@dataclass
class TrendMatch:
    """Data class for a single trend match"""
    trend_name: str
    similarity_score: float
    trend_popularity: float
    combined_score: float


class TrendTopicMatcher:
    """
    Advanced Trend Topic Matcher using semantic embeddings
    
    Uses Sentence-BERT for intelligent trend matching beyond simple keyword matching.
    Optionally integrates with Google Trends for real popularity data.
    """
    
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 similarity_threshold: float = 0.3,
                 use_google_trends: bool = False,
                 region: str = 'US'):
        """
        Initialize the trend matcher
        
        Args:
            embedding_model: Sentence-BERT model name
            similarity_threshold: Minimum similarity to consider a match (0-1)
            use_google_trends: Whether to fetch real trends from Google
            region: Country code for Google Trends ('US', 'GB', 'IN', 'PK', etc.)
        """
        self.similarity_threshold = similarity_threshold
        self.region = region
        self.use_google_trends = use_google_trends
        
        # Load embedding model (pre-trained, no training needed)
        print(f"📚 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("✓ Embedding model loaded successfully")
        
        # Initialize Google Trends if requested
        self.pytrends = None
        if use_google_trends and HAS_PYTRENDS:
            try:
                self.pytrends = TrendReq(hl='en-US', tz=360)
                print("✓ Google Trends API connected")
            except Exception as e:
                print(f"⚠ Could not connect to Google Trends: {e}")
                self.use_google_trends = False
        
        # Cache for embeddings (performance optimization)
        self._embedding_cache = {}
        
    def get_trends(self, custom_trends: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Fetch current trending topics from various sources
        
        Args:
            custom_trends: Optional list of custom trends to use instead of fetching
            
        Returns:
            Dictionary of trend names with their popularity scores (0-100)
        """
        trends = {}
        
        # Use custom trends if provided
        if custom_trends:
            for trend in custom_trends:
                trends[trend.lower()] = 70  # Default popularity score
            return trends
        
        # Fetch from Google Trends
        if self.use_google_trends and self.pytrends:
            try:
                trending_searches = self.pytrends.trending_searches(pn=self.region)
                for i, keyword in enumerate(trending_searches[0].head(20)):
                    # Popularity decreases with rank
                    popularity = max(30, 100 - (i * 3))
                    trends[keyword.lower()] = popularity
            except Exception as e:
                print(f"⚠ Google Trends fetch error: {e}")
        
        # Add trending categories (always available)
        trending_categories = {
            # Technology
            'artificial intelligence': 95, 'machine learning': 92, 'deep learning': 90,
            'chatgpt': 98, 'ai tools': 94, 'generative ai': 96,
            'python programming': 88, 'data science': 85, 'cloud computing': 82,
            'cybersecurity': 84, 'blockchain': 78, 'cryptocurrency': 75,
            
            # Social Media & Viral
            'viral video': 95, 'trending now': 92, 'social media': 88,
            'tiktok trend': 94, 'instagram reels': 90, 'youtube shorts': 89,
            
            # Entertainment
            'movie review': 85, 'netflix series': 88, 'gaming': 86,
            'streaming': 84, 'music': 82, 'celebrity news': 80,
            
            # Education & How-to
            'tutorial': 87, 'how to': 90, 'beginner guide': 85,
            'tips and tricks': 88, 'step by step': 84,
            
            # News & Current Events
            'breaking news': 93, 'latest update': 90, 'explained': 86,
            'analysis': 85, 'what happened': 88,
            
            # Product & Reviews
            'unboxing': 82, 'product review': 85, 'best of': 83,
            'comparison': 84, 'top 10': 86
        }
        
        for trend, popularity in trending_categories.items():
            trends[trend] = max(trends.get(trend, 0), popularity)
        
        return trends
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using Sentence-BERT with caching
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector
        """
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        # Handle empty text
        if not text or len(text.strip()) < 3:
            text = "empty topic"
        
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        self._embedding_cache[text] = embedding
        return embedding
    
    def match_topic_with_trends(self, 
                                 topic_name: str, 
                                 topic_keywords: List[str],
                                 trends: Dict[str, float]) -> List[TrendMatch]:
        """
        Match a single topic with trending themes using semantic similarity
        
        Args:
            topic_name: Name/description of the topic
            topic_keywords: List of keywords associated with the topic
            trends: Dictionary of trend names with popularity scores
            
        Returns:
            List of TrendMatch objects sorted by combined score
        """
        # Create combined topic representation
        topic_text = f"{topic_name} {' '.join(topic_keywords[:5])}"
        topic_embedding = self.encode_text(topic_text)
        
        matches = []
        
        for trend_name, popularity in trends.items():
            # Encode trend
            trend_embedding = self.encode_text(trend_name)
            
            # Calculate cosine similarity
            similarity = float(cosine_similarity(
                topic_embedding.reshape(1, -1),
                trend_embedding.reshape(1, -1)
            )[0][0])
            
            # Apply threshold
            if similarity < self.similarity_threshold:
                continue
            
            # Calculate combined score: 70% similarity + 30% popularity
            combined_score = (similarity * 0.7) + (popularity / 100 * 0.3)
            combined_score = min(1.0, combined_score) * 100  # Convert to percentage
            
            matches.append(TrendMatch(
                trend_name=trend_name,
                similarity_score=round(similarity * 100, 1),
                trend_popularity=popularity,
                combined_score=round(combined_score, 1)
            ))
        
        # Sort by combined score (highest first)
        matches.sort(key=lambda x: x.combined_score, reverse=True)
        
        return matches[:5]  # Return top 5 matches
    
    def calculate_growth_potential(self, 
                                   matches: List[TrendMatch],
                                   topic_size: int,
                                   avg_engagement: float) -> Dict[str, Any]:
        """
        Calculate overall growth potential based on matches and topic metrics
        
        Args:
            matches: List of trend matches
            topic_size: Number of comments in this topic
            avg_engagement: Average engagement score (0-100)
            
        Returns:
            Dictionary with growth metrics
        """
        if not matches:
            return {
                'score': 25,
                'level': 'Low',
                'explanation': 'No strong trend matches found'
            }
        
        # Factor 1: Best match score (40% weight)
        best_match_score = matches[0].combined_score
        
        # Factor 2: Number of good matches (20% weight)
        good_matches = sum(1 for m in matches if m.combined_score > 50)
        match_diversity = min(25, good_matches * 5)
        
        # Factor 3: Topic volume (20% weight)
        volume_score = min(25, topic_size // 5) if topic_size else 0
        
        # Factor 4: Engagement potential (20% weight)
        engagement_score = avg_engagement * 0.25
        
        # Calculate final score
        final_score = (
            best_match_score * 0.4 +
            match_diversity * 0.2 +
            volume_score * 0.2 +
            engagement_score * 0.2
        )
        
        final_score = min(100, max(0, final_score))
        
        # Determine growth level
        if final_score >= 70:
            level = "High"
            explanation = "Strong trend alignment with high growth potential"
        elif final_score >= 50:
            level = "Medium"
            explanation = "Moderate trend alignment - consider optimizing content"
        else:
            level = "Low"
            explanation = "Weak trend alignment - pivot or add trending angles"
        
        return {
            'score': round(final_score, 1),
            'level': level,
            'explanation': explanation,
            'best_match_score': best_match_score,
            'match_diversity': match_diversity,
            'volume_score': volume_score,
            'engagement_score': round(engagement_score, 1)
        }
    
    def evaluate_topic_growth(self, 
                              topics: List[Dict[str, Any]],
                              custom_trends: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main pipeline: Evaluate growth potential for all topics
        
        Args:
            topics: List of topic dictionaries from clustering module
                   Each topic should have: 'name', 'keywords', 'size'
            custom_trends: Optional list of custom trends to use
            
        Returns:
            Structured JSON with evaluation results
        """
        if not topics:
            return {
                'success': False,
                'error': 'No topics provided',
                'topic_evaluations': [],
                'summary': {},
                'timestamp': datetime.now().isoformat()
            }
        
        # Get current trends
        trends = self.get_trends(custom_trends)
        
        # Evaluate each topic
        evaluations = []
        
        for topic in topics:
            topic_name = topic.get('name', 'Unknown Topic')
            topic_keywords = topic.get('keywords', [])
            topic_size = topic.get('size', 0)
            avg_engagement = topic.get('engagement_score', 50)
            
            # Find matching trends
            matches = self.match_topic_with_trends(
                topic_name, topic_keywords, trends
            )
            
            # Calculate growth potential
            growth = self.calculate_growth_potential(
                matches, topic_size, avg_engagement
            )
            
            # Prepare matches for output
            matches_output = [
                {
                    'trend': m.trend_name,
                    'similarity_score': m.similarity_score,
                    'trend_popularity': m.trend_popularity,
                    'combined_score': m.combined_score
                }
                for m in matches
            ]
            
            evaluations.append({
                'topic': {
                    'name': topic_name,
                    'keywords': topic_keywords[:5],
                    'size': topic_size
                },
                'top_trend_matches': matches_output,
                'growth_potential': {
                    'score': growth['score'],
                    'level': growth['level'],
                    'explanation': growth['explanation']
                },
                'detailed_scores': {
                    'best_match_score': growth.get('best_match_score', 0),
                    'volume_score': growth.get('volume_score', 0),
                    'engagement_score': growth.get('engagement_score', 0)
                }
            })
        
        # Sort by growth score (highest first)
        evaluations.sort(key=lambda x: x['growth_potential']['score'], reverse=True)
        
        # Generate summary
        summary = self._generate_summary(evaluations)
        
        return {
            'success': True,
            'total_topics': len(evaluations),
            'topic_evaluations': evaluations,
            'summary': summary,
            'trends_used': list(trends.keys())[:20],
            'similarity_threshold': self.similarity_threshold,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_summary(self, evaluations: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics from evaluations"""
        if not evaluations:
            return {}
        
        high = sum(1 for e in evaluations if e['growth_potential']['level'] == 'High')
        medium = sum(1 for e in evaluations if e['growth_potential']['level'] == 'Medium')
        low = sum(1 for e in evaluations if e['growth_potential']['level'] == 'Low')
        
        avg_score = sum(e['growth_potential']['score'] for e in evaluations) / len(evaluations)
        
        # Get top opportunity
        top_topic = evaluations[0]['topic']['name'] if evaluations else None
        
        return {
            'high_growth_topics': high,
            'medium_growth_topics': medium,
            'low_growth_topics': low,
            'average_growth_score': round(avg_score, 1),
            'top_opportunity': top_topic,
            'recommendation': self._get_recommendation(high, medium, low)
        }
    
    def _get_recommendation(self, high: int, medium: int, low: int) -> str:
        """Generate content recommendation based on summary"""
        if high >= 2:
            return "Strong opportunities detected! Create content around High growth topics immediately."
        elif high >= 1:
            return "Good opportunities found. Focus on the High growth topic for maximum engagement."
        elif medium >= 2:
            return "Medium potential topics available. Optimize with trending keywords to boost growth."
        else:
            return "Low trend alignment. Consider pivoting content strategy or using different angles."


# =============================================================================
# EXAMPLE USAGE (for testing/demo)
# =============================================================================

if __name__ == "__main__":
    # Sample topics from a clustering module
    sample_topics = [
        {
            'name': 'iPhone 15 Pro Max Camera Review',
            'keywords': ['iphone', 'camera', 'review', 'quality', 'photo'],
            'size': 45,
            'engagement_score': 82
        },
        {
            'name': 'Python Programming for Beginners',
            'keywords': ['python', 'coding', 'tutorial', 'learn', 'programming'],
            'size': 38,
            'engagement_score': 75
        },
        {
            'name': 'Make Money Online Fast',
            'keywords': ['money', 'earn', 'online', 'fast', 'income'],
            'size': 12,
            'engagement_score': 45
        },
        {
            'name': 'Cute Cat Compilation',
            'keywords': ['cat', 'cute', 'funny', 'animals', 'pets'],
            'size': 28,
            'engagement_score': 70
        }
    ]
    
    # Initialize matcher
    matcher = TrendTopicMatcher(
        similarity_threshold=0.25,
        use_google_trends=False  # Set True for real trends
    )
    
    # Evaluate growth potential
    result = matcher.evaluate_topic_growth(sample_topics)
    
    # Print results
    print("\n" + "="*60)
    print("TREND TOPIC MATCHING RESULTS")
    print("="*60)
    
    print(f"\n📊 SUMMARY:")
    print(f"   High Growth Topics: {result['summary']['high_growth_topics']}")
    print(f"   Medium Growth Topics: {result['summary']['medium_growth_topics']}")
    print(f"   Low Growth Topics: {result['summary']['low_growth_topics']}")
    print(f"   Average Growth Score: {result['summary']['average_growth_score']}%")
    print(f"\n💡 Recommendation: {result['summary']['recommendation']}")
    
    print("\n📋 DETAILED TOPIC EVALUATIONS:")
    for eval_data in result['topic_evaluations']:
        print(f"\n   🔹 Topic: {eval_data['topic']['name']}")
        print(f"      Growth Potential: {eval_data['growth_potential']['level']} ({eval_data['growth_potential']['score']}%)")
        print(f"      Explanation: {eval_data['growth_potential']['explanation']}")
        
        if eval_data['top_trend_matches']:
            print(f"      Top Trend Match: {eval_data['top_trend_matches'][0]['trend']} "
                  f"(similarity: {eval_data['top_trend_matches'][0]['similarity_score']}%)")
    
    # Output JSON
    print("\n" + "="*60)
    print("📄 JSON OUTPUT STRUCTURE:")
    print(json.dumps({
        'topic': result['topic_evaluations'][0]['topic']['name'],
        'growth_potential': result['topic_evaluations'][0]['growth_potential'],
        'top_matches': result['topic_evaluations'][0]['top_trend_matches'][:2]
    }, indent=2))