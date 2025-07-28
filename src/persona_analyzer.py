# persona_analyzer.py - FIXED VERSION with proper Travel Planner scoring and subsection analysis
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from typing import Dict, List, Any
import logging
import re

class PersonaAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger("persona_analyzer")
        self.model = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9
        )
        
        # Download NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize sentence transformer
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Loaded sentence transformer model")
        except Exception as e:
            self.logger.warning(f"Could not load sentence transformer: {e}")
            self.model = None

    def analyze_persona_relevance(self, documents: List[Dict], persona: str, job_to_be_done: str) -> List[Dict]:
        """Analyze document sections for Travel Planner persona relevance"""
        
        # Create enhanced persona profile for travel planning
        persona_profile = self._create_travel_planner_profile(persona, job_to_be_done)
        
        # Extract all sections from documents
        all_sections = []
        for doc in documents:
            for section in doc.get('sections', []):
                section_data = {
                    'document': doc['file_name'],
                    'page_number': section['page_number'],
                    'section_title': section['title'],
                    'content': section['content'],
                    'word_count': section.get('word_count', 0),
                    'relevance_score': 0.0
                }
                all_sections.append(section_data)
        
        if not all_sections:
            self.logger.warning("No sections found in documents")
            return []
        
        # Score sections for travel planning relevance
        scored_sections = self._score_travel_sections(all_sections, persona_profile)
        
        # Rank sections by relevance
        ranked_sections = sorted(scored_sections, key=lambda x: x['relevance_score'], reverse=True)
        
        # Add importance ranks
        for i, section in enumerate(ranked_sections, 1):
            section['importance_rank'] = i
            section['relevance_score'] = round(section['relevance_score'], 4)
        
        return ranked_sections

    def _create_travel_planner_profile(self, persona: str, job_to_be_done: str) -> Dict[str, Any]:
        """Create enhanced profile specifically for travel planning"""
        
        # Travel planner keywords - much more comprehensive
        travel_keywords = [
            # Core travel planning
            'trip', 'travel', 'plan', 'planning', 'itinerary', 'vacation', 'holiday', 'journey',
            'visit', 'explore', 'tour', 'guide', 'destination', 'adventure', 'experience',
            
            # Activities and attractions  
            'activities', 'things to do', 'attractions', 'sightseeing', 'entertainment',
            'nightlife', 'bars', 'clubs', 'beach', 'coastal', 'museum', 'festival',
            'hiking', 'outdoor', 'water sports', 'adventure', 'cultural', 'historical',
            
            # Food and dining
            'restaurants', 'dining', 'food', 'cuisine', 'culinary', 'cooking', 'wine',
            'tasting', 'local dishes', 'seafood', 'market', 'cafe', 'bar', 'bistro',
            
            # Accommodation and logistics
            'hotels', 'accommodation', 'stay', 'booking', 'location', 'transport',
            'getting around', 'packing', 'tips', 'tricks', 'budget', 'cost',
            
            # Group travel specific
            'group', 'friends', 'college', 'young', 'budget', 'affordable', 'fun',
            'party', 'social', 'together', 'shared', 'split'
        ]
        
        # Extract key terms from job description
        job_keywords = self._extract_job_keywords(job_to_be_done)
        
        # Combine all keywords
        all_keywords = travel_keywords + job_keywords
        
        return {
            'persona_text': persona,
            'job_text': job_to_be_done,
            'travel_keywords': travel_keywords,
            'job_keywords': job_keywords,
            'all_keywords': list(set(all_keywords)),
            'combined_text': f"{persona} {job_to_be_done}",
            'group_size': self._extract_group_size(job_to_be_done),
            'trip_duration': self._extract_trip_duration(job_to_be_done)
        }

    def _extract_job_keywords(self, job_text: str) -> List[str]:
        """Extract important keywords from job description"""
        # Remove common stop words and extract meaningful terms
        words = re.findall(r'\b\w+\b', job_text.lower())
        keywords = []
        
        for word in words:
            if (len(word) > 3 and 
                word not in self.stop_words and 
                not word.isdigit()):
                keywords.append(word)
        
        return keywords

    def _extract_group_size(self, job_text: str) -> int:
        """Extract group size from job description"""
        import re
        numbers = re.findall(r'\b(\d+)\b', job_text)
        for num in numbers:
            if 2 <= int(num) <= 50:  # Reasonable group size range
                return int(num)
        return 1

    def _extract_trip_duration(self, job_text: str) -> int:
        """Extract trip duration from job description"""
        import re
        # Look for patterns like "4 days", "one week", etc.
        day_matches = re.findall(r'(\d+)\s*days?', job_text.lower())
        if day_matches:
            return int(day_matches[0])
        
        week_matches = re.findall(r'(\d+)\s*weeks?', job_text.lower())
        if week_matches:
            return int(week_matches[0]) * 7
            
        return 7  # Default to one week

    def _score_travel_sections(self, sections: List[Dict], persona_profile: Dict) -> List[Dict]:
        """Score sections specifically for travel planning relevance"""
        
        if not sections:
            return sections
        
        # Method 1: Travel-specific keyword scoring
        self._travel_keyword_scoring(sections, persona_profile)
        
        # Method 2: TF-IDF scoring with travel focus
        self._travel_tfidf_scoring(sections, persona_profile)
        
        # Method 3: Semantic similarity (if available)
        if self.model:
            self._semantic_scoring(sections, persona_profile)
        
        # Method 4: Section type and content quality scoring
        self._content_quality_scoring(sections, persona_profile)
        
        return sections

    def _travel_keyword_scoring(self, sections: List[Dict], persona_profile: Dict):
        """Score based on travel-specific keywords"""
        travel_keywords = persona_profile['travel_keywords']
        job_keywords = persona_profile['job_keywords']
        
        for section in sections:
            text = f"{section['section_title']} {section['content']}".lower()
            
            # Count travel keyword matches
            travel_matches = sum(1 for keyword in travel_keywords if keyword in text)
            job_matches = sum(1 for keyword in job_keywords if keyword in text)
            
            # Weight travel keywords more heavily
            keyword_score = (travel_matches * 0.7 + job_matches * 0.3) / max(len(travel_keywords), 1)
            
            # Bonus for high-value travel terms
            high_value_terms = ['activities', 'things to do', 'attractions', 'restaurants', 
                               'nightlife', 'coastal', 'adventure', 'guide', 'tips']
            bonus = sum(0.1 for term in high_value_terms if term in text)
            
            section['travel_keyword_score'] = min(keyword_score + bonus, 1.0)

    def _travel_tfidf_scoring(self, sections: List[Dict], persona_profile: Dict):
        """TF-IDF scoring focused on travel content"""
        try:
            section_texts = [f"{s['section_title']} {s['content']}" for s in sections]
            persona_text = persona_profile['combined_text']
            
            # Add travel-focused vocabulary
            travel_vocab = " ".join(persona_profile['travel_keywords'])
            all_texts = section_texts + [persona_text + " " + travel_vocab]
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            persona_vector = tfidf_matrix[-1]
            section_vectors = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(section_vectors, persona_vector).flatten()
            
            for i, section in enumerate(sections):
                section['tfidf_score'] = float(similarities[i])
                
        except Exception as e:
            self.logger.warning(f"TF-IDF scoring failed: {e}")
            for section in sections:
                section['tfidf_score'] = 0.0

    def _semantic_scoring(self, sections: List[Dict], persona_profile: Dict):
        """Semantic similarity scoring"""
        try:
            section_texts = [f"{s['section_title']} {s['content']}" for s in sections]
            persona_text = persona_profile['combined_text']
            
            section_embeddings = self.model.encode(section_texts, show_progress_bar=False)
            persona_embedding = self.model.encode([persona_text])
            
            similarities = cosine_similarity(section_embeddings, persona_embedding).flatten()
            
            for i, section in enumerate(sections):
                section['semantic_score'] = float(similarities[i])
                
        except Exception as e:
            self.logger.warning(f"Semantic scoring failed: {e}")
            for section in sections:
                section['semantic_score'] = 0.0

    def _content_quality_scoring(self, sections: List[Dict], persona_profile: Dict):
        """Score based on content quality and travel relevance"""
        
        for section in sections:
            title = section['section_title'].lower()
            content = section['content'].lower()
            word_count = section['word_count']
            
            # Base quality score
            quality_score = 0.0
            
            # Content length factor (prefer substantial content)
            if word_count > 200:
                quality_score += 0.3
            elif word_count > 100:
                quality_score += 0.2
            elif word_count > 50:
                quality_score += 0.1
            
            # Travel relevance in title (high weight)
            travel_title_terms = ['guide', 'activities', 'things to do', 'attractions', 
                                'restaurants', 'hotels', 'nightlife', 'coastal', 'adventures',
                                'culinary', 'cuisine', 'entertainment', 'tips', 'tricks']
            title_matches = sum(0.15 for term in travel_title_terms if term in title)
            quality_score += min(title_matches, 0.4)
            
            # Avoid generic sections
            generic_terms = ['introduction', 'conclusion', 'overview', 'summary']
            if any(term in title for term in generic_terms):
                quality_score *= 0.3  # Heavy penalty for generic sections
            
            # Group travel relevance
            group_terms = ['group', 'friends', 'college', 'budget', 'affordable', 'young']
            if any(term in content for term in group_terms):
                quality_score += 0.1
            
            section['quality_score'] = min(quality_score, 1.0)
            
            # Combine all scores
            travel_kw = section.get('travel_keyword_score', 0.0)
            tfidf = section.get('tfidf_score', 0.0)
            semantic = section.get('semantic_score', 0.0)
            quality = section.get('quality_score', 0.0)
            
            # Weighted combination
            section['relevance_score'] = (
                0.25 * travel_kw +
                0.25 * tfidf +
                0.25 * semantic +
                0.25 * quality
            )

    def generate_subsection_analysis(self, top_sections: List[Dict], max_sections: int = 5) -> List[Dict]:
        """Generate detailed subsection analysis from top sections"""
        subsection_analysis = []
        
        for section in top_sections[:max_sections]:
            # Split content into meaningful chunks
            content_chunks = self._split_into_meaningful_chunks(section['content'])
            
            for chunk in content_chunks[:2]:  # Max 2 chunks per section
                if len(chunk.split()) >= 40:  # Only substantial chunks
                    subsection_analysis.append({
                        'document': section['document'],
                        'refined_text': chunk.strip(),
                        'page_number': section['page_number']
                    })
        
        return subsection_analysis

    def _split_into_meaningful_chunks(self, content: str, min_words: int = 40, max_words: int = 150) -> List[str]:
        """Split content into meaningful, coherent chunks"""
        
        # First try splitting by paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        good_chunks = []
        
        for para in paragraphs:
            para = para.strip()
            word_count = len(para.split())
            
            if min_words <= word_count <= max_words:
                good_chunks.append(para)
            elif word_count > max_words:
                # Split large paragraphs by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    if len(test_chunk.split()) <= max_words:
                        current_chunk = test_chunk
                    else:
                        if current_chunk and len(current_chunk.split()) >= min_words:
                            good_chunks.append(current_chunk.strip())
                        current_chunk = sentence
                
                if current_chunk and len(current_chunk.split()) >= min_words:
                    good_chunks.append(current_chunk.strip())
        
        return good_chunks[:3]  # Return max 3 chunks