"""
Crypto Twitter Sentiment Analyzer

A self-contained Python module for real-time analysis of Twitter sentiment
related to cryptocurrency token price movements. Designed for use as signals
in prediction and reinforcement learning models.

Requirements:
pip install tweepy textblob vaderSentiment pymongo python-dotenv requests numpy pandas

Environment Variables Required (.env file):
TWITTER_BEARER_TOKEN=your_bearer_token_here
TWITTER_API_KEY=your_api_key_here
TWITTER_API_SECRET=your_api_secret_here
TWITTER_ACCESS_TOKEN=your_access_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret_here
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=crypto_sentiment
"""

import os
import re
import time
import json
import logging
import datetime as dt
from datetime import datetime, timezone
from typing import Dict, List, Optional

# External dependencies
import twikit
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pydantic import BaseModel, field_validator

from cryptotrading.data.models import TweetDataPoint, TweetSentiment

# Map TweetData to TweetDataPoint
TweetData = TweetDataPoint

# Map SentimentScore to TweetSentiment
SentimentScore = TweetSentiment

def to_date(date_str):
    return dt.datetime.strptime(date_str, '%a %b %d %H:%M:%S %z %Y')


class TwitterUser(BaseModel):
    can_dm: Optional[bool] = None
    can_media_tag: Optional[bool] = None
    created_at: Optional[datetime] = None
    default_profile: Optional[bool] = None
    default_profile_image: Optional[bool] = None
    description: Optional[str] = None
    entities: Optional[dict] = None
    fast_followers_count: Optional[int] = None
    favourites_count: Optional[int] = None
    followers_count: Optional[int] = None
    friends_count: Optional[int] = None
    has_custom_timelines: Optional[bool] = None
    is_translator: Optional[bool] = None
    listed_count: Optional[int] = None
    location: Optional[str] = None
    media_count: Optional[int] = None
    name: Optional[str] = None
    normal_followers_count: Optional[int] = None
    pinned_tweet_ids_str: Optional[List[str]] = None
    possibly_sensitive: Optional[bool] = None
    profile_banner_url: Optional[str] = None
    profile_image_url_https: Optional[str] = None
    profile_interstitial_type: Optional[str] = None
    screen_name: Optional[str] = None
    statuses_count: Optional[int] = None
    translator_type: Optional[str] = None
    url: Optional[str] = None
    verified: Optional[bool] = None
    verified_type: Optional[str] = None
    want_retweets: Optional[bool] = None
    withheld_in_countries: Optional[List[str]] = None

    @field_validator('created_at', mode='before')
    @classmethod
    def parse_created_at(cls, v):
        if isinstance(v, str):
            # Twitter format: 'Sun Oct 26 03:52:51 +0000 2025'
            return datetime.strptime(v, '%a %b %d %H:%M:%S %z %Y')
        return v

class Tweet(BaseModel):
    bookmark_count: Optional[int] = None
    bookmarked: Optional[bool] = None
    created_at: Optional[datetime] = None
    conversation_id_str: Optional[str] = None
    display_text_range: Optional[List[int]] = None
    entities: Optional[dict] = None
    favorite_count: Optional[int] = None
    favorited: Optional[bool] = None
    full_text: Optional[str] = None
    is_quote_status: Optional[bool] = None
    lang: Optional[str] = None
    quote_count: Optional[int] = None
    reply_count: Optional[int] = None
    retweet_count: Optional[int] = None
    retweeted: Optional[bool] = None
    scopes: Optional[dict] = None
    user_id_str: Optional[str] = None
    id_str: Optional[str] = None
    retweeted_status_result: Optional[dict] = None
    
    @field_validator('created_at', mode='before')
    @classmethod
    def parse_created_at(cls, v):
        if isinstance(v, str):
            # Twitter format: 'Sun Oct 26 03:52:51 +0000 2025'
            return datetime.strptime(v, '%a %b %d %H:%M:%S %z %Y')
        return v

class CryptoSentimentAnalyzer:
    """
    Main class for analyzing Twitter sentiment related to cryptocurrency tokens
    """
    
    def __init__(self, mongo_uri: str = None, db_name: str = None):
        """Initialize the sentiment analyzer"""
        self.logger = self._setup_logging()
        
        # Helper to run async methods from sync context
        def run_async(coro):
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                    except ImportError:
                        pass
                    return loop.run_until_complete(coro)
                else:
                    return loop.run_until_complete(coro)
            except RuntimeError:
                return asyncio.run(coro)
        self._run_async = run_async

        # Twitter API setup
        self.twitter_client = twikit.Client('en-US')
        
        # Database adapter setup
        from cryptotrading.data.factory import get_twitter_adapter
        self.db_adapter = get_twitter_adapter()
        self._run_async(self.db_adapter.initialize())
        
        # Sentiment analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Price direction keywords
        self.bullish_keywords = {
            'moon', 'bullish', 'pump', 'rocket', 'up', 'rise', 'surge', 'rally',
            'breakout', 'bull', 'green', 'buy', 'hodl', 'diamond', 'hands',
            'to the moon', 'ath', 'all time high', 'going up', 'massive gain'
        }
        
        self.bearish_keywords = {
            'dump', 'crash', 'bear', 'down', 'fall', 'drop', 'sell', 'red',
            'bearish', 'correction', 'dip', 'plummet', 'decline', 'tank',
            'bleeding', 'rekt', 'paper hands', 'going down', 'massive loss'
        }
        
        # Crypto-specific terms that amplify sentiment
        self.crypto_amplifiers = {
            'bitcoin', 'btc', 'ethereum', 'eth', 'altcoin', 'defi', 'nft',
            'blockchain', 'crypto', 'coin', 'token', 'whale', 'satoshi'
        }
        
        self.logger.info("CryptoSentimentAnalyzer initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('crypto_sentiment')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _setup_twitter_account(self) -> Account:
        cookies = json.loads(os.getenv("TWITTER_COOKIES"))
        account = Account(cookies=cookies)        
        
        return account

    def _setup_twitter_scraper(self) -> Scraper:
        cookies = json.loads(os.getenv("TWITTER_COOKIES"))
        scraper = Scraper(cookies)
        return scraper
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess tweet text"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions and hashtags for sentiment analysis (keep for context)
        clean_text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        clean_text = ' '.join(clean_text.split())
        
        return clean_text.lower()
    
    def extract_crypto_tokens(self, text: str) -> List[str]:
        """Extract cryptocurrency token symbols from tweet text"""
        # Common patterns for crypto tokens
        patterns = [
            r'\$([A-Z]{2,10})',  # $BTC, $ETH format
            r'#([A-Z]{2,10})',   # #BTC, #ETH format
            r'\b([A-Z]{2,10})\b' # Standalone symbols
        ]
        
        tokens = set()
        text_upper = text.upper()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_upper)
            tokens.update(matches)
        
        # Filter out common false positives
        false_positives = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'HAD', 'HAS', 'HIS', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        return list(tokens - false_positives)
    
    def calculate_sentiment(self, text: str, token_symbol: str = None) -> TweetSentiment:
        """Calculate comprehensive sentiment score"""
        clean_text = self.preprocess_text(text)
        
        # VADER sentiment (good for social media)
        vader_scores = self.vader_analyzer.polarity_scores(clean_text)
        
        # TextBlob sentiment
        blob = TextBlob(clean_text)
        
        # Price direction signals
        price_signals = self._calculate_price_direction_signals(clean_text, token_symbol)
        
        # Confidence based on text length and crypto relevance
        confidence = self._calculate_confidence(text, token_symbol)
        
        return TweetSentiment(
            compound=vader_scores['compound'],
            positive=vader_scores['pos'],
            negative=vader_scores['neg'],
            neutral=vader_scores['neu'],
            polarity=blob.sentiment.polarity,
            subjectivity=blob.sentiment.subjectivity,
            confidence=confidence
        )
    
    def _calculate_price_direction_signals(self, text: str, token_symbol: str = None) -> Dict[str, float]:
        """Calculate signals for price direction prediction"""
        signals = {
            'bullish_signal': 0.0,
            'bearish_signal': 0.0,
            'uncertainty_signal': 0.0,
            'volume_signal': 0.0
        }
        
        words = text.split()
        text_set = set(words)
        
        # Bullish signals
        bullish_matches = len(text_set.intersection(self.bullish_keywords))
        signals['bullish_signal'] = min(bullish_matches / 3.0, 1.0)  # Normalize to 0-1
        
        # Bearish signals
        bearish_matches = len(text_set.intersection(self.bearish_keywords))
        signals['bearish_signal'] = min(bearish_matches / 3.0, 1.0)
        
        # Uncertainty (conflicting signals)
        if bullish_matches > 0 and bearish_matches > 0:
            signals['uncertainty_signal'] = 0.8
        
        # Volume indicators (caps, exclamation marks, repetition)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        exclamation_count = text.count('!')
        
        signals['volume_signal'] = min((caps_ratio * 2) + (exclamation_count / 10), 1.0)
        
        # Token-specific amplification
        if token_symbol and token_symbol.lower() in text:
            for key in signals:
                signals[key] *= 1.2  # Amplify signals when token is mentioned
        
        return signals
    
    def _calculate_confidence(self, text: str, token_symbol: str = None) -> float:
        """Calculate confidence score for the sentiment analysis"""
        confidence = 0.5  # Base confidence
        
        # Text length factor
        text_length = len(text.split())
        if text_length >= 10:
            confidence += 0.2
        elif text_length >= 20:
            confidence += 0.3
        
        # Crypto relevance
        crypto_terms = len(set(text.lower().split()).intersection(self.crypto_amplifiers))
        confidence += min(crypto_terms * 0.1, 0.3)
        
        # Token mention
        if token_symbol and token_symbol.lower() in text.lower():
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def get_following_users(self, user_id: str = None, max_results: int = 1000) -> List[Dict]:
        """Get list of users followed by the authenticated user or specified user"""
        try:
            if user_id is None:
                # Get authenticated user's following
                user_id = self.twitter_client.get_me()
            
            following = self.twitter_client.get_following(user_id)
            
            self.logger.info(f"Retrieved {len(following)} following users")
            return following
            
        except Exception as e:
            self.logger.error(f"Error getting following users: {e}")
            return []
    
    def analyze_user_tweets(self, user_id: str, token_symbol: str, 
                          max_tweets: int = 50) -> List[TweetData]:
        """Analyze recent tweets from a specific user for token sentiment"""
        try:
            tweets = self.twitter_client.get_users_tweets(
                id=user_id,
                max_results=min(max_tweets, 100),
                tweet_fields=['created_at', 'public_metrics', 'context_annotations'],
                exclude=['retweets', 'replies']
            )
            
            if not tweets.data:
                return []
            
            # Get user info
            user_info = self.twitter_client.get_user(
                id=user_id,
                user_fields=['public_metrics', 'verified']
            )
            
            analyzed_tweets = []
            
            for tweet in tweets.data:
                # Check if tweet mentions the token
                if not self._tweet_mentions_token(tweet.text, token_symbol):
                    continue
                
                sentiment = self.calculate_sentiment(tweet.text, token_symbol)
                price_signals = self._calculate_price_direction_signals(
                    self.preprocess_text(tweet.text), token_symbol
                )
                
                tweet_data = TweetData(
                    tweet_id=tweet.id,
                    user_id=user_id,
                    username=user_info.data.username,
                    text=tweet.text,
                    timestamp=tweet.created_at.replace(tzinfo=timezone.utc),
                    token_symbol=token_symbol.upper(),
                    sentiment=sentiment,
                    price_direction_signals=price_signals,
                    follower_count=user_info.data.public_metrics['followers_count'] if user_info.data.public_metrics else 0,
                    verified=user_info.data.verified or False,
                    retweet_count=tweet.public_metrics['retweet_count'] if tweet.public_metrics else 0,
                    like_count=tweet.public_metrics['like_count'] if tweet.public_metrics else 0,
                    reply_count=tweet.public_metrics['reply_count'] if tweet.public_metrics else 0,
                    raw_data=tweet.data
                )
                
                analyzed_tweets.append(tweet_data)
            
            return analyzed_tweets
            
        except Exception as e:
            self.logger.error(f"Error analyzing user tweets: {e}")
            return []
    
    def _tweet_mentions_token(self, text: str, token_symbol: str) -> bool:
        """Check if tweet mentions the specified token"""
        text_lower = text.lower()
        token_lower = token_symbol.lower()
        
        # Check various formats
        formats = [
            f'${token_lower}',
            f' {token_lower} ',
            f'{token_lower}/',
            f'/{token_lower}'
        ]
        
        return any(fmt in text_lower for fmt in formats)
    
    def save_to_database(self, tweet_data: TweetData) -> bool:
        """Save analyzed tweet data to active database backend"""
        try:
            success = self._run_async(self.db_adapter.save_tweet_sentiment(tweet_data))
            if success:
                self.logger.debug(f"Saved tweet {tweet_data.tweet_id} to database")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error saving to database: {e}")
            return False
    
    def run_analysis(self, token_symbol: str, duration_minutes: int = 60,
                    max_following: int = 100) -> None:
        """Run continuous sentiment analysis for a specified duration"""
        self.logger.info(f"Starting analysis for {token_symbol} for {duration_minutes} minutes")
        
        # Get following users
        following_users = self.get_following_users(max_results=max_following)
        
        if not following_users:
            self.logger.error("No following users found")
            return
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        processed_tweets = set()
        
        while time.time() < end_time:
            try:
                for user in following_users:
                    if time.time() >= end_time:
                        break
                    
                    tweets = self.analyze_user_tweets(
                        user['id'], token_symbol, max_tweets=10
                    )
                    
                    for tweet_data in tweets:
                        if tweet_data.tweet_id not in processed_tweets:
                            success = self.save_to_database(tweet_data)
                            if success:
                                processed_tweets.add(tweet_data.tweet_id)
                                self.logger.info(
                                    f"Processed tweet from @{tweet_data.username}: "
                                    f"Sentiment={tweet_data.sentiment.compound:.3f}, "
                                    f"Bullish={tweet_data.price_direction_signals['bullish_signal']:.3f}"
                                )
                    
                    # Rate limiting
                    time.sleep(1)
                
                # Wait before next cycle
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                time.sleep(10)
        
        self.logger.info(f"Analysis completed. Processed {len(processed_tweets)} tweets")
    
    def get_aggregated_sentiment(self, token_symbol: str, 
                               hours_back: int = 1) -> Dict:
        """Get aggregated sentiment data for ML models"""
        try:
            return self._run_async(self.db_adapter.get_aggregated_sentiment(token_symbol, hours_back))
        except Exception as e:
            self.logger.error(f"Error getting aggregated sentiment: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self._run_async(self.db_adapter.shutdown())
        except Exception as e:
            pass
        self.logger.info("Cleanup completed")

# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = CryptoSentimentAnalyzer()
    
    try:
        # Example: Analyze sentiment for Bitcoin over 30 minutes
        analyzer.run_analysis(token_symbol="BTC", duration_minutes=30, max_following=50)
        
        # Get aggregated results for ML model
        sentiment_data = analyzer.get_aggregated_sentiment("BTC", hours_back=1)
        print("Aggregated Sentiment Data:")
        print(json.dumps(sentiment_data, indent=2, default=str))
        
    except KeyboardInterrupt:
        print("Analysis interrupted by user")
    finally:
        analyzer.cleanup()