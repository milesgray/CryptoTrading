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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# External dependencies
from twitter.account import Account
from twitter.scraper import Scraper
import pymongo
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
from dotenv import load_dotenv



# Load environment variables
load_dotenv()

@dataclass
class SentimentScore:
    """Sentiment analysis results"""
    compound: float
    positive: float
    negative: float
    neutral: float
    polarity: float
    subjectivity: float
    confidence: float

def to_date(date_str):
    return dt.datetime.strptime(date_str, '%a %b %d %H:%M:%S %z %Y')

class TweetData:
    tweet_id: str
    user_id: int
    username: str
    text: str
    timestamp: dt.datetime
    token_symbol: str
    sentiment: str
    price_direction_signals: list
    follower_count: int
    verified: bool
    retweet_count: int
    like_count: int
    reply_count: int
    raw_data: dict

class TwitterClient:
    def __init__(self):
        self.cookies = {
            "ct0": os.getenv("TWITTER_COOKIE_CT0"),
            "auth_token": os.getenv("TWITTER_COOKIE_AUTH_TOKEN"),
            "twid": os.getenv("TWITTER_COOKIE_TWID"),
        }
        self.user_id = int(self.cookies["twid"])
        self.account = Account(cookies=self.cookies)
        self.scraper = Scraper(cookies=self.cookies)

    def get_me(self):
        return self.user_id

    def get_timeline_tweets(self, limit: int = 100, cursor: Optional[str] = None):

        def get_entries(timeline_data: dict) -> list:
            return timeline_data[0]['data']['home']['home_timeline_urt']['instructions'][0]['entries'] 
        def get_tweet_metadata(entry: dict) -> dict:
            return entry['content']['itemContent']['tweet_results']['result']['legacy']
        def get_tweet(entry: dict) -> dict:
            try:
                tweet_info = get_tweet_metadata(entry)
            
                return {
                    "id": tweet_info['id_str'],
                    "created_at": to_date(tweet_info['created_at']),
                    "text": tweet_info['full_text'],
                    "user_id": tweet_info['user_id_str'],
                }
            except KeyError:
                return None
        
        timeline = self.account.home_latest_timeline(limit, cursor)
        tweets = [get_tweet(entry) for entry in get_entries(timeline)]
        return [t for t in tweets if t]

    def get_following(self, user_id):
        raw = self.scraper.following([user_id])

        def parse_result(entry):
            return entry['content']['itemContent']['user_results']['result']
        def parse_user(entry):
            try:
                raw_user = parse_result(entry)

                return {
                    "id": raw_user["rest_id"],
                    "username": raw_user["legacy"]["screen_name"],
                    "name": raw_user["legacy"]["name"],
                    "verified": raw_user["legacy"]["verified"] or False,
                    "followers_count": raw_user["legacy"]["followers_count"]
                }
            except KeyError:
                return None
        instructions_list = [r['data']['user']['result']['timeline']['timeline']['instructions'] for r in raw]
        instructions = [[i for i in instructions if i['type'] == 'TimelineAddEntries'] for instructions in instructions_list]
        entries_list = [i[0]['entries'] for i in instructions]
        entries = sum(entries_list, [])
        users = [parse_user(e) for e in entries]

        return [u for u in users if u]

    def get_users_tweets(self, user_id):
        return self.scraper.tweets([user_id])

    def get_users(self, user_ids):
        return self.scraper.users_by_ids(user_ids)

    

class CryptoSentimentAnalyzer:
    """
    Main class for analyzing Twitter sentiment related to cryptocurrency tokens
    """
    
    def __init__(self, mongo_uri: str = None, db_name: str = None):
        """Initialize the sentiment analyzer"""
        self.logger = self._setup_logging()
        
        # Twitter API setup
        self.twitter_client = TwitterClient()
        
        # MongoDB setup
        self.mongo_client = pymongo.MongoClient(
            mongo_uri or os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        )
        self.db = self.mongo_client[db_name or os.getenv('MONGODB_DATABASE', 'crypto_sentiment')]
        self.collection = self.db.tweet_sentiment
        
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
    
    def calculate_sentiment(self, text: str, token_symbol: str = None) -> SentimentScore:
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
        
        return SentimentScore(
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
    
    def save_to_mongodb(self, tweet_data: TweetData) -> bool:
        """Save analyzed tweet data to MongoDB"""
        try:
            document = asdict(tweet_data)
            document['sentiment'] = asdict(tweet_data.sentiment)
            document['created_at'] = datetime.now(timezone.utc)
            
            # Create index for efficient querying
            self.collection.create_index([
                ('token_symbol', 1),
                ('timestamp', -1),
                ('user_id', 1)
            ])
            
            result = self.collection.insert_one(document)
            
            if result.inserted_id:
                self.logger.debug(f"Saved tweet {tweet_data.tweet_id} to MongoDB")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error saving to MongoDB: {e}")
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
                            success = self.save_to_mongodb(tweet_data)
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
            cutoff_time = datetime.now(timezone.utc) - pd.Timedelta(hours=hours_back)
            
            pipeline = [
                {
                    '$match': {
                        'token_symbol': token_symbol.upper(),
                        'timestamp': {'$gte': cutoff_time}
                    }
                },
                {
                    '$group': {
                        '_id': None,
                        'avg_compound': {'$avg': '$sentiment.compound'},
                        'avg_bullish_signal': {'$avg': '$price_direction_signals.bullish_signal'},
                        'avg_bearish_signal': {'$avg': '$price_direction_signals.bearish_signal'},
                        'avg_uncertainty': {'$avg': '$price_direction_signals.uncertainty_signal'},
                        'avg_volume_signal': {'$avg': '$price_direction_signals.volume_signal'},
                        'weighted_sentiment': {
                            '$avg': {
                                '$multiply': [
                                    '$sentiment.compound',
                                    '$sentiment.confidence',
                                    {'$log10': {'$add': ['$follower_count', 1]}}
                                ]
                            }
                        },
                        'total_tweets': {'$sum': 1},
                        'verified_tweets': {
                            '$sum': {'$cond': ['$verified', 1, 0]}
                        },
                        'high_engagement_tweets': {
                            '$sum': {
                                '$cond': [
                                    {'$gt': [{'$add': ['$like_count', '$retweet_count']}, 10]},
                                    1, 0
                                ]
                            }
                        }
                    }
                }
            ]
            
            result = list(self.collection.aggregate(pipeline))
            
            if result:
                data = result[0]
                data['token_symbol'] = token_symbol.upper()
                data['timestamp'] = datetime.now(timezone.utc)
                data['hours_analyzed'] = hours_back
                return data
            else:
                return {
                    'token_symbol': token_symbol.upper(),
                    'timestamp': datetime.now(timezone.utc),
                    'hours_analyzed': hours_back,
                    'total_tweets': 0
                }
                
        except Exception as e:
            self.logger.error(f"Error getting aggregated sentiment: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()
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