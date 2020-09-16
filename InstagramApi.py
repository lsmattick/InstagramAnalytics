from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import requests
import time
import os

from instagram_private_api import Client, ClientCompatPatch


class InstagramApi(Client):
    """
    Insert Doc String Here. Sup Dude
    """
    def __init__(self, **kwargs):
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

    def check_cookie(self, cookie, **kwargs):
        if cookie:
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

    def make_df(self, feed):
        cols = ['id', 'taken_at', 'like_count', 'comment_count', 'caption']

        posts = feed['items']

        posts_dict = {col: [post[col] for post in posts] for col in cols}
        df = pd.DataFrame(posts_dict)

        df['taken_at'] = df['taken_at'].apply(self.unix_to_datetime)
        df['caption'] = df['caption'].apply(lambda x: x['text'] if x else x)
        df['caption_length'] = df['caption'].apply(lambda x: len(x.replace('\n\n', ' ')) if x else x)
        df['caption_word_count'] = df['caption'].apply(lambda x: len(x.split()) if x else x)

        return df

    def calc_engagement(self, df):
        df['engagements'] = df['like_count'] + df['comment_count']
        df['engagement_rate'] = np.round(100 * (df['engagements'] / self.follower_count), 2)

    def collect_user_posts(self, user_name, start_date):
        self.user_name = user_name
        self.user_data = self.get_user_metadata(user_name)
        self.user_id = self.user_data['id']
        self.follower_count = self.user_data['info']['follower_count']

        unix_ts = self.datetime_to_unix(start_date)
        init_feed = self.user_feed(self.user_id, min_timestamp=unix_ts)
        posts = init_feed['items']

        df = self.make_df(init_feed)
        last_ts = df['taken_at'].min()
        last_id = df.set_index('taken_at').loc[last_ts]['id']

        while last_ts > start_date:
            next_feed = self.user_feed(self.user_id, max_id=last_id, min_timestamp=unix_ts)
            next_df = self.make_df(next_feed)
            df = df.append(next_df)
            if len(next_df) > 0:
                last_ts = next_df['taken_at'].min()
                last_id = next_df.set_index('taken_at').loc[last_ts]['id']
                posts = posts + next_feed['items']
            else:
                break

        self.post_dict = {post['id']: post for post in posts}
        df = df.reset_index(drop=True)
        self.calc_engagement(df)
        self.df = df

        return df, self.post_dict

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

    def download_top_ten_posts(self):
        top_ten_posts = self.df.sort_values('like_count', ascending=False).head(10)['id']
        for post_id in top_ten_posts:
            self.download_post_photo(post_id)
