import os
import json
import datetime as dt
from datetime import datetime, timezone
from typing import Any, Union
from cryptotrading.data.models import TweetDataPoint

from pymongo import ASCENDING
from cryptotrading.data.postgres import get_connection, init_pool, _pool

from twitter.account import Account
from twitter.scraper import Scraper

from cryptotrading.data.mongo import get_db
from cryptotrading.config import (
    TWEET_COLLECTION_NAME
)

def to_date(date_str):
    return dt.datetime.strptime(date_str, '%a %b %d %H:%M:%S %z %Y')

class TwitterMongoAdapter:
    def __init__(self):
        self.db = get_db()

    async def initialize(self):
        self.collections = await self.db.list_collection_names()
        await self.init_tweet_collection()

    async def init_tweet_collection(self):
        self.tweet_collection = self.db[TWEET_COLLECTION_NAME]
        
        # Create time series collection if it doesn't exist
        if TWEET_COLLECTION_NAME not in self.collections:
            try:
                await self.db.create_collection(TWEET_COLLECTION_NAME)
            except:
                await self.db.create_collection(TWEET_COLLECTION_NAME)
        # Create indexes for faster queries
        await self.tweet_collection.create_index([("timestamp", ASCENDING)])

    async def save_tweet_sentiment(self, tweet_data: Union[TweetDataPoint, Any]) -> bool:
        try:
            import dataclasses
            document = dataclasses.asdict(tweet_data)
            document['sentiment'] = dataclasses.asdict(tweet_data.sentiment)
            document['created_at'] = dt.datetime.now(dt.timezone.utc)
            
            # Create index for efficient querying
            await self.tweet_collection.create_index([
                ('token_symbol', 1),
                ('timestamp', -1),
                ('user_id', 1)
            ])
            
            result = await self.tweet_collection.insert_one(document)
            return bool(result.inserted_id)
        except Exception as e:
            return False

    async def get_aggregated_sentiment(self, token_symbol: str, hours_back: int = 1) -> dict:
        import pandas as pd
        cutoff_time = dt.datetime.now(dt.timezone.utc) - pd.Timedelta(hours=hours_back)
        
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
        
        cursor = self.tweet_collection.aggregate(pipeline)
        result = await cursor.to_list(length=1)
        if result:
            data = result[0]
            data['token_symbol'] = token_symbol.upper()
            data['timestamp'] = dt.datetime.now(dt.timezone.utc)
            data['hours_analyzed'] = hours_back
            return data
        else:
            return {
                'token_symbol': token_symbol.upper(),
                'timestamp': dt.datetime.now(dt.timezone.utc),
                'hours_analyzed': hours_back,
                'total_tweets': 0
            }

class TwitterReader:
    def __init__(self):
        cookies = json.loads(os.getenv("X_COOKIE"))
        self.account = Account(cookies=cookies)
        self.scraper = Scraper(cookies)

    def get_user_id(self, username):
        users = self.scraper.users([username])
        if users:
            return users[0].id
        else:
            return None 

    def get_timeline(self) -> list[dict]:
        """
        Get timeline using the Account-based approach with improved error handling.
        Returns list of tuples containing (tweet_text, tweet_id).        
        """
                
        def get_post_data(tweet_info: dict) -> dict:
            return {
                "content": tweet_info['full_text'],
                "user_id": tweet_info['user_id_str'],
                "tweet_id": tweet_info['id_str'],
                "comment_count": tweet_info['reply_count'],
                "like_count": tweet_info['favorite_count'],
                "created_at": to_date(tweet_info['created_at'])
            }
        def get_entries(timeline_data: dict) -> list:
            return timeline_data[0]['data']['home']['home_timeline_urt']['instructions'][0]['entries'] 
        def get_tweet_info(entry: dict) -> dict:
            return entry['content']['itemContent']['tweet_results']['result']['legacy']
        def parse_entry(entry: dict) -> dict:
            try:
                tweet_info = get_tweet_info(entry)
                return get_post_data(tweet_info)
            except KeyError:
                return None
        # method body
        try:
            # Get timeline with error handling
            timeline = self.account.home_latest_timeline(200)

            results = [parse_entry(entry) for entry in get_entries(timeline)]

            return [r for r in results if r]

        except Exception as e:            
            self.logger.error(f"Error getting timeline: {e}", context="get_timeline")
            return []
    
    def get_notifications(self):
        try:
            data = self.account.notifications()

            results = []
            
            if 'globalObjects' not in data or 'tweets' not in data['globalObjects']:
                return results
            tweets = data['globalObjects']['tweets']
            processed_roots = set()            

            sorted_tweets = sorted(
                tweets.items(),
                key=lambda x: to_date(x[1]['created_at']),
                reverse=True
            )

            for tweet_id, _ in sorted_tweets:
                root_id = self._get_root_tweet_id(tweets, tweet_id)

                if root_id not in processed_roots:                    
                    processed_roots.add(root_id)
                    conversation = self._build_conversation(data, tweet_id)
                    if conversation:
                        self.logger.info(f"Conversation found: {root_id}", context="get_notifications")
                        # Format the conversation for LLM
                        output = ["New reply to my original conversation thread or a Mention from somebody:"]

                        for i, tweet in enumerate(conversation, 1):
                            reply_context = (f"[Replying to {next((t['username'] for t in conversation if t['id'] == tweet['reply_to']), 'unknown')}]"
                                            if tweet['reply_to'] else "[Original tweet]")

                            output.append(f"{i}. {tweet['username']} {reply_context}:")
                            output.append(f"   \"{tweet['text']}\"")
                            output.append("")

                        content = "\n".join(output)
                        results.append(
                            ReadEvent(
                                content=content, 
                                channel_id=tweet_id,
                                channel_name="tweet"
                            )
                        )
            return results
        except Exception as e:
            self.logger.error(f'Error getting notifications: {str(e)}', context="get_notifications")
            return []
    
    def _get_root_tweet_id(self, tweets, start_id):
        """Find the root tweet ID of a conversation."""
        current_id = start_id
        while True:
            tweet = tweets.get(str(current_id))
            if not tweet:
                return current_id
            parent_id = tweet.get('in_reply_to_status_id_str')
            if not parent_id or parent_id not in tweets:
                return current_id
            current_id = parent_id

    def _build_conversation(self, data, tweet_id):
        """Convert a conversation tree into LLM-friendly format."""
        tweets = data['globalObjects']['tweets']
        users = data['globalObjects']['users']

        def get_post_data(current_tweet, user_info):
            return {
                "content": current_tweet['full_text'],
                "username": user_info['screen_name'],
                "tweet_id": current_tweet['id_str'],
                "comment_count": current_tweet['reply_count'],
                "like_count": current_tweet['favorite_count'],
                "image_path": user_info.get('profile_image_url_https'),
                "created_at": to_date(current_tweet['created_at'])
            }

        def get_conversation_chain(current_id, processed_ids=None):
            if processed_ids is None:
                processed_ids = set()

            if not current_id or current_id in processed_ids:
                return []

            processed_ids.add(current_id)
            current_tweet = tweets.get(str(current_id))
            if not current_tweet:
                return []

            user_info = users.get(str(current_tweet['user_id']))
            user = self.data.get_user(user_info['screen_name'])
            if not user:
                user = self.data.save_user({"username": user_info["screen_name"], "email": None})
            post = self.data.get_post(current_tweet['id_str'])
            if not post:
                post = self.data.save_post(
                    user=user,
                    post_data=get_post_data(current_tweet, user_info)
                )
                
            username = f"@{user.username}" if user else "Unknown User"

            chain = [{
                'id': current_id,
                'username': username,
                'text': current_tweet['full_text'],
                'reply_to': current_tweet.get('in_reply_to_status_id_str')
            }]

            for potential_reply_id, potential_reply in tweets.items():
                if potential_reply.get('in_reply_to_status_id_str') == current_id:
                    chain.extend(get_conversation_chain(potential_reply_id, processed_ids))

            return chain

        root_id = self._get_root_tweet_id(tweets, tweet_id)
        conversation = get_conversation_chain(root_id)

        return conversation


class TwitterPostgresAdapter:
    def __init__(self):
        pass

    async def initialize(self):
        if _pool is None:
            await init_pool()

    async def shutdown(self):
        pass

    async def save_tweet_sentiment(self, tweet_data: Union[TweetDataPoint, Any]) -> bool:
        try:
            tweet_id = tweet_data.tweet_id
            timestamp = tweet_data.timestamp
            user_id = str(tweet_data.user_id)
            text = tweet_data.text
            sentiment = tweet_data.sentiment
            score = sentiment.compound
            magnitude = sentiment.confidence
            
            import dataclasses
            # Prepare metadata dict
            meta = {
                "username": tweet_data.username,
                "token_symbol": tweet_data.token_symbol,
                "sentiment": dataclasses.asdict(sentiment),
                "price_direction_signals": tweet_data.price_direction_signals,
                "follower_count": tweet_data.follower_count,
                "verified": tweet_data.verified,
                "retweet_count": tweet_data.retweet_count,
                "like_count": tweet_data.like_count,
                "reply_count": tweet_data.reply_count,
                "raw_data": tweet_data.raw_data
            }
            
            async with get_connection() as conn:
                await conn.execute('''
                    INSERT INTO tweet_data (id, created_at, author_id, text, sentiment_score, sentiment_magnitude, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (id) DO UPDATE
                    SET created_at = EXCLUDED.created_at, metadata = EXCLUDED.metadata,
                        sentiment_score = EXCLUDED.sentiment_score, sentiment_magnitude = EXCLUDED.sentiment_magnitude;
                ''', tweet_id, timestamp, user_id, text, score, magnitude, meta)
            return True
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error saving tweet to Postgres: {e}")
            return False

    async def get_aggregated_sentiment(self, token_symbol: str, hours_back: int = 1) -> dict:
        try:
            from datetime import timedelta
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            
            query = '''
                SELECT 
                    AVG(sentiment_score) AS avg_compound,
                    AVG((metadata->'price_direction_signals'->>'bullish_signal')::double precision) AS avg_bullish_signal,
                    AVG((metadata->'price_direction_signals'->>'bearish_signal')::double precision) AS avg_bearish_signal,
                    AVG((metadata->'price_direction_signals'->>'uncertainty_signal')::double precision) AS avg_uncertainty,
                    AVG((metadata->'price_direction_signals'->>'volume_signal')::double precision) AS avg_volume_signal,
                    AVG(sentiment_score * sentiment_magnitude * LOG(10, (metadata->>'follower_count')::double precision + 1)) AS weighted_sentiment,
                    COUNT(*)::int AS total_tweets,
                    SUM(CASE WHEN (metadata->>'verified')::boolean = TRUE THEN 1 ELSE 0 END)::int AS verified_tweets,
                    SUM(CASE WHEN (metadata->>'like_count')::int + (metadata->>'retweet_count')::int > 10 THEN 1 ELSE 0 END)::int AS high_engagement_tweets
                FROM tweet_data
                WHERE metadata->>'token_symbol' = $1 AND created_at >= $2;
            '''
            
            async with get_connection() as conn:
                row = await conn.fetchrow(query, token_symbol.upper(), cutoff_time)
                
            if row and row['total_tweets'] > 0:
                res = dict(row)
                res['token_symbol'] = token_symbol.upper()
                res['timestamp'] = datetime.now(timezone.utc)
                res['hours_analyzed'] = hours_back
                return res
            else:
                return {
                    'token_symbol': token_symbol.upper(),
                    'timestamp': datetime.now(timezone.utc),
                    'hours_analyzed': hours_back,
                    'total_tweets': 0
                }
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error getting aggregated sentiment from Postgres: {e}")
            return {}

    