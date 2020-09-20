from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
import random
import requests
import time

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from instagram_private_api import Client, ClientCompatPatch


class InstagramApi(Client):
    """
    Insert Doc String Here
    """
    def __init__(self, cached_cookie=None, **kwargs):
        if cached_cookie:
            with open(cached_cookie, 'rb') as handle:
                cookie = pickle.load(handle)
            super().__init__(cookie=cookie, **kwargs)
        else:
            super().__init__(**kwargs)

        self.now = datetime.now()
        self.check_cookie(**kwargs)


    @staticmethod
    def unix_to_datetime(timestamp):
        ts = datetime.fromtimestamp(int(timestamp))
        return ts

    @staticmethod
    def datetime_to_unix(timestamp):
        return int(time.mktime(timestamp.timetuple()))

    def check_cookie(self, **kwargs):
        expiration = self.unix_to_datetime(self.cookie_jar.auth_expires)
        if expiration <= self.now:
            print(f'Warning: Cookie is expired.')
        else:
            print(f'Warning: Cookie expires {expiration}')

    def dump_cookie(self):
        with open('insta_cookie.pickle', 'wb') as handle:
            pickle.dump(self.cookie_jar.dump(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_user_metadata(self, user_name):
        r = requests.get(f'https://www.instagram.com/{user_name}/?__a=1')
        user_id = int(r.json()['logging_page_id'].strip('profilePage_'))

        data = {
            'id': user_id,
            'json': r.json(),
            'info': self.user_info(user_id)['user']
        }

        return data

    def init_user(self, user_name):
        self.user_name = user_name
        self.user_data = self.get_user_metadata(user_name)
        self.user_id = self.user_data['id']
        self.follower_count = self.user_data['info']['follower_count']

    def make_df_post_analytics(self, post_list):
        cols = ['id', 'taken_at', 'like_count', 'comment_count', 'caption']

        column_dict = {col: [post[col] for post in post_list] for col in cols}
        df = pd.DataFrame(column_dict)

        df['taken_at'] = df['taken_at'].apply(self.unix_to_datetime)
        df['caption'] = df['caption'].apply(lambda x: x['text'] if x else x)
        df['caption_length'] = df['caption'].apply(lambda x: len(x.replace('\n\n', ' ')) if x else x)
        df['caption_word_count'] = df['caption'].apply(lambda x: len(x.split()) if x else x)
        df['engagements'] = df['like_count'] + df['comment_count']
        df['engagement_rate'] = np.round(100 * (df['engagements'] / self.follower_count), 2)

        return df

    def collect_user_posts(self, start_date):
        if not getattr(self, 'user_name', None):
            m = 'Please init_user before using this method'
            raise NotImplementedError(m)

        unix_ts = self.datetime_to_unix(start_date)
        init_feed = self.user_feed(self.user_id, min_timestamp=unix_ts)
        if init_feed['num_results'] == 0:
            print(f'Warning: {self.user_name} has no posts since {str(start_date.date())}')
            print('Using default user feed (last 18 posts)')
            init_feed = self.user_feed(self.user_id, min_timestamp=unix_ts)
        posts = init_feed['items']

        last_post = posts[-1]
        last_ts = self.unix_to_datetime(last_post['taken_at'])
        last_id = last_post['id']

        while last_ts > start_date:
            next_feed = self.user_feed(self.user_id, max_id=last_id, min_timestamp=unix_ts)
            if next_feed['num_results'] > 0:
                last_post = next_feed['items'][-1]
                last_ts = self.unix_to_datetime(last_post['taken_at'])
                last_id = last_post['id']
                posts = posts + next_feed['items']
            else:
                break

        self.post_dict = {post['id']: post for post in posts}
        df_post_analytics = self.make_df_post_analytics(posts)
        ranked_post_ids = list(df_post_analytics.sort_values('engagements', ascending=False)['id'])

        return df_post_analytics, ranked_post_ids

    def download_post_photo(self, post_id):
        carousel_media = self.post_dict[post_id].get('carousel_media')
        if carousel_media:
            urls = [(pic['image_versions2']['candidates'][0]['url'], n) for n, pic in enumerate(carousel_media)]
        else:
            urls = [(self.post_dict[post_id]['image_versions2']['candidates'][0]['url'], 0)]
        for url, n in urls:
            r = requests.get(url)
            filename = f'{self.user_name}/{post_id}_{n}.png'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as file:
                file.write(r.content)

    def download_top_ten_posts(self, df):
        top_ten_posts = df.sort_values('like_count', ascending=False).head(10)['id']
        for post_id in top_ten_posts:
            self.download_post_photo(post_id)

    def strip_hashtags(self, string):
        hashtags = []
        for  word in string.split():
            if '#' == word[0]:
                hashtags.append(word)

        return hashtags

    def get_hashtags(self, post_id):
        caption = self.media_info(post_id)['items'][0]['caption']['text']
        hashtags = self.strip_hashtags(caption)

        comment_raw_data = self.media_n_comments(post_id, n=100)
        comments = [comment['text'] for comment in comment_raw_data]
        for comment in comments:
            hashtags = hashtags + self.strip_hashtags(comment)
        return hashtags

    def get_engagers(self, post_id):
        # Need to mark users as private or public here.
        # Later on, get list of public engagers
        # Then either run analysis on all of them or sample them
        # depending on the size
        likes_raw_data = self.media_likers(post_id)
        likers = [like['pk'] for like in likes_raw_data['users']]
        comments_raw_data = self.media_n_comments(post_id)
        commenters = [comment['user']['pk'] for comment in comments_raw_data]
        engagers = list(set(likers + commenters))

        return engagers

    def get_comment_hashtags(self, post_id):
        comment_raw_data = self.media_n_comments(post_id, n=100)
        comments = [comment['text'] for comment in comment_raw_data]
        hashtags = []
        for comment in comments:
            hashtags = hashtags + self.strip_hashtags(comment)
        return hashtags

    def engagement_hashtag_analysis_multiple_posts(self, post_list, engager_limit=50):
        if isinstance(post_list, str):
            post_list = [post_list]
        engagers = set()
        for post in post_list:
            post_engagers = set(self.get_engagers(post))
            engagers = engagers | post_engagers

        if len(engagers) > engager_limit:
            engagers = random.sample(engagers, engager_limit)
            print(f'Engager limit reached, using random sample of {engager_limit} engagers')
        print(f'Total Engagers Used {len(engagers)}')
        private_user_count = 0
        failed_count = 0

        hashtags = []
        for user in engagers:

            try:
                feed = self.user_feed(user)
                user_post_ids = [feed_post['id'] for feed_post in feed['items']]
                captions = [feed_post['caption']['text'] for feed_post in feed['items']]

                caption_hashtags = []
                for caption in captions:
                    caption_hashtags = caption_hashtags + self.strip_hashtags(caption)

                user_hashtags = []
                for post in user_post_ids:
                    post_hashtags = []
                    try:
                        post_hashtags = self.get_comment_hashtags(post)
                        user_hashtags = user_hashtags + post_hashtags
                    except:
                        failed_count += 1

                hashtags = hashtags + caption_hashtags + user_hashtags
            except:
                private_user_count += 1
                continue

        df_hashtag_ranks = pd.DataFrame(pd.Series(hashtags).value_counts()).reset_index()
        df_hashtag_ranks.columns = ['hashtag', 'count']

        print(f'Private Users: {private_user_count}')
        print(f'Failed Count: {failed_count}')

        return df_hashtag_ranks, hashtags

    def make_string_for_word_cloud(self, df, n):
        string = ''
        if n > len(df):
            print(f'Warning: {n} is larger than len(df) = {len(df)}')
            print('setting n = len(df)')
            n = len(df)
        for x in range(n):
            hashtag = df.iloc[x]['hashtag']
            count = df.iloc[x]['count']
            string = string + (hashtag + ' ') * count

        return string

    def make_hashtag_word_cloud(self, df, n, stopwords=None, font_path=None):
        if not font_path:
            font_path = 'fonts/coolvetica/coolvetica rg.ttf'

        wc_string = self.make_string_for_word_cloud(df, n)

        wordcloud = WordCloud(
            background_color = 'white',
            collocations=False,
            width=2000, height=1000,
            random_state=40,
            stopwords=['photography'],
            font_path=font_path
        )

        wc_plot = wordcloud.generate(wc_string)

        plt.figure(figsize=(40,20))
        plt.imshow(wc_plot, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(f'{self.user_name}_ht_word_cloud.png', format='png')
