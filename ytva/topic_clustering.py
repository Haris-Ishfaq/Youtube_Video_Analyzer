"""
Topic Clustering Module using BERTopic
No training required - uses pre-trained SBERT embeddings
"""

import os
import pickle
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from collections import Counter

class TopicClustering:
    def __init__(self, model_path='models/topic_model.pkl', use_bertopic=True):
        """
        Initialize topic clustering with BERTopic (no training needed)
        
        Args:
            model_path: Path to save/load the model
            use_bertopic: Use BERTopic (True) or simple KMeans (False)
        """
        self.model_path = model_path
        self.use_bertopic = use_bertopic
        self.topic_model = None
        self.embedding_model = None
        
        # Initialize embedding model (SBERT - no training!)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ SBERT embedding model loaded")
        
        if use_bertopic and os.path.exists(model_path):
            try:
                self.topic_model = BERTopic.load(model_path)
                print("✓ BERTopic model loaded from disk")
            except:
                print("⚠ Could not load saved model, will retrain")
                self.topic_model = None
        
    def cluster_comments(self, comments, num_topics=None, min_topic_size=5):
        """
        Cluster comments into topics
        
        Args:
            comments: List of comment texts
            num_topics: Number of topics (auto-detected if None)
            min_topic_size: Minimum comments per topic
            
        Returns:
            Dictionary with topics, assignments, and topic info
        """
        if not comments or len(comments) == 0:
            return {
                'success': False,
                'error': 'No comments provided',
                'topics': [],
                'assignments': [],
                'topic_counts': {}
            }
        
        # Clean and filter empty comments
        cleaned_comments = []
        original_indices = []
        for i, comment in enumerate(comments):
            if comment and len(comment.strip()) > 5:  # Skip very short comments
                cleaned_comments.append(comment)
                original_indices.append(i)
        
        if len(cleaned_comments) < min_topic_size:
            return {
                'success': False,
                'error': f'Not enough comments (minimum {min_topic_size} needed)',
                'topics': [],
                'assignments': [],
                'topic_counts': {}
            }
        
        if self.use_bertopic:
            return self._cluster_with_bertopic(cleaned_comments, original_indices, min_topic_size)
        else:
            return self._cluster_with_kmeans(cleaned_comments, original_indices, num_topics)
    
    def _cluster_with_bertopic(self, comments, original_indices, min_topic_size):
        """Use BERTopic for advanced clustering"""
        try:
            # Create or load BERTopic model
            if self.topic_model is None:
                # Custom vectorizer to remove common words
                vectorizer = CountVectorizer(
                    stop_words="english",
                    min_df=2,
                    max_df=0.95,
                    ngram_range=(1, 2)
                )
                
                self.topic_model = BERTopic(
                    embedding_model=self.embedding_model,
                    vectorizer_model=vectorizer,
                    min_topic_size=min_topic_size,
                    calculate_probabilities=True,
                    verbose=False
                )
            
            # Fit and transform
            topics, probs = self.topic_model.fit_transform(comments)
            
            # Get topic info
            topic_info = self.topic_model.get_topic_info()
            
            # Map back to original indices
            assignments = [{'comment_idx': original_indices[i], 'topic': int(topics[i]), 'probability': float(probs[i]) if probs[i] else 0} 
                          for i in range(len(comments))]
            
            # Extract topic names and keywords
            topics_dict = {}
            for topic_id in set(topics):
                if topic_id != -1:  # -1 is outlier topic
                    topic_words = self.topic_model.get_topic(topic_id)
                    if topic_words:
                        top_words = [word for word, _ in topic_words[:5]]
                        topics_dict[str(topic_id)] = {
                            'id': int(topic_id),
                            'name': ', '.join(top_words[:3]),
                            'keywords': top_words,
                            'size': int(sum(1 for t in topics if t == topic_id))
                        }
            
            # Add outlier topic
            outlier_count = sum(1 for t in topics if t == -1)
            if outlier_count > 0:
                topics_dict['-1'] = {
                    'id': -1,
                    'name': 'Other / Mixed Topics',
                    'keywords': [],
                    'size': outlier_count
                }
            
            # Save model for future use
            if self.topic_model and not os.path.exists(self.model_path):
                self.topic_model.save(self.model_path)
                print(f"✓ Topic model saved to {self.model_path}")
            
            return {
                'success': True,
                'method': 'BERTopic',
                'topics': list(topics_dict.values()),
                'assignments': assignments,
                'topic_counts': {str(k): v['size'] for k, v in topics_dict.items()},
                'total_comments': len(comments),
                'num_topics': len([t for t in topics_dict.keys() if t != '-1'])
            }
            
        except Exception as e:
            print(f"❌ BERTopic error: {e}")
            return self._cluster_with_kmeans(comments, original_indices)
    
    def _cluster_with_kmeans(self, comments, original_indices, num_topics=5):
        """Fallback: Simple KMeans clustering using embeddings"""
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(comments, show_progress_bar=False)
            
            # Determine number of topics
            n_topics = min(num_topics, len(comments) // 3, 10)
            n_topics = max(2, n_topics)  # At least 2 topics
            
            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
            topics = kmeans.fit_predict(embeddings)
            
            # Get cluster centers and keywords
            topics_dict = {}
            topic_keywords = {}
            
            for topic_id in range(n_topics):
                # Get comments in this cluster
                cluster_indices = [i for i, t in enumerate(topics) if t == topic_id]
                cluster_comments = [comments[i] for i in cluster_indices]
                
                # Extract simple keywords
                all_words = ' '.join(cluster_comments).lower().split()
                word_counts = Counter(all_words)
                # Remove common stop words
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'to', 'of', 'for', 'in', 'on', 'at', 'with', 'by', 'is', 'was', 'are', 'this', 'that', 'it', 'you', 'i', 'me', 'my', 'your', 'video', 'comment'}
                
                keywords = [word for word, count in word_counts.most_common(10) 
                           if word not in stop_words and len(word) > 2][:5]
                
                topics_dict[str(topic_id)] = {
                    'id': int(topic_id),
                    'name': ', '.join(keywords[:3]) if keywords else f'Topic {topic_id + 1}',
                    'keywords': keywords,
                    'size': len(cluster_comments)
                }
            
            # Create assignments
            assignments = [{'comment_idx': original_indices[i], 'topic': int(topics[i]), 'probability': 0} 
                          for i in range(len(comments))]
            
            return {
                'success': True,
                'method': 'KMeans (Fallback)',
                'topics': list(topics_dict.values()),
                'assignments': assignments,
                'topic_counts': {str(k): v['size'] for k, v in topics_dict.items()},
                'total_comments': len(comments),
                'num_topics': n_topics
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'topics': [],
                'assignments': [],
                'topic_counts': {}
            }
    
    def get_topic_summary(self, comments, assignments):
        """Generate human-readable summary of topics"""
        if not assignments:
            return "No topics generated"
        
        summary = []
        topic_groups = {}
        
        for assignment in assignments:
            topic = str(assignment['topic'])
            if topic not in topic_groups:
                topic_groups[topic] = []
            topic_groups[topic].append(assignment['comment_idx'])
        
        for topic_id, indices in topic_groups.items():
            sample_comments = [comments[idx] for idx in indices[:3]]
            summary.append({
                'topic_id': topic_id,
                'count': len(indices),
                'sample_comments': sample_comments
            })
        
        return summary