from bs4 import BeautifulSoup
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt")
# import malaya
import json
import csv
import pandas as pd
import numpy as np
import gspread
import datetime
from dateutil.parser import parse

# import cloudscraper
# import statistics
# import collections
from hyper import HTTPConnection
from hyper.contrib import HTTP20Adapter
from facebook_scraper import get_posts
# import httpx
import cloudscraper
import time as tm
import os

os.environ['TZ'] = 'Asia/Kuala_Lumpur'

now = datetime.datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

gc = gspread.service_account(filename="creds.json")
sh_entry = gc.open("tigerears_alpha").sheet1
sh_word_distribution = gc.open("tigerears_alpha").get_worksheet(2)

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd
import numpy
import tensorflow as tf

from facebook_scraper import get_posts

model_name = "xlnet-tiny-bahasa-cased"
loaded_tokenizer = AutoTokenizer.from_pretrained(
    f"{model_name}-sentiment-v7", do_lower_case=True
)
loaded_model = TFAutoModelForSequenceClassification.from_pretrained(
    f"{model_name}-sentiment-v7"
)


def predict_sentiment(sentence):
    predict_input = loaded_tokenizer.encode(
        sentence,
        truncation=True,
        max_length=256,
        padding=True,
        add_special_tokens=True,  # add [CLS], [SEP]
        return_tensors="tf",
    )

    tf_output = loaded_model.predict(predict_input)[0]
    tf_prediction = tf.nn.softmax(tf_output, axis=1).numpy()[0]

    sentiment = "Negative" if tf_prediction[0] > tf_prediction[1] else "Positive"
    print(tf_prediction[0], tf_prediction[1])
    print(sentence)
    print(sentiment)
    return sentiment


def lowyat_search(keywords=["kw1", "kw2"]):
    s = cloudscraper.create_scraper()

    querystring = {"act": "Search", "CODE": "01"}

    results_all = []
    timestamps = []
    keyword_used = []
    post_urls = []
    usernames = []
    profile_urls = []
    lowyat_entries = []
    for keyword in keywords:

        payload = (
            "keywords="
            + keyword
            + "&namesearch=&forums%5B%5D=80&forums%5B%5D=1&forums%5B%5D=2&forums%5B%5D=196&forums%5B%5D=328&forums%5B%5D=81&forums%5B%5D=73&forums%5B%5D=25&forums%5B%5D=3&forums%5B%5D=281&forums%5B%5D=294&forums%5B%5D=4&forums%5B%5D=101&forums%5B%5D=306&forums%5B%5D=248&forums%5B%5D=193&forums%5B%5D=164&forums%5B%5D=102&forums%5B%5D=139&forums%5B%5D=140&forums%5B%5D=100&forums%5B%5D=187&forums%5B%5D=5&forums%5B%5D=130&forums%5B%5D=129&forums%5B%5D=9&forums%5B%5D=37&forums%5B%5D=163&forums%5B%5D=7&forums%5B%5D=88&forums%5B%5D=35&forums%5B%5D=63&forums%5B%5D=92&forums%5B%5D=143&forums%5B%5D=83&forums%5B%5D=115&forums%5B%5D=174&forums%5B%5D=255&forums%5B%5D=213&forums%5B%5D=230&forums%5B%5D=256&forums%5B%5D=314&forums%5B%5D=61&forums%5B%5D=325&forums%5B%5D=222&forums%5B%5D=254&forums%5B%5D=308&forums%5B%5D=209&forums%5B%5D=205&forums%5B%5D=295&forums%5B%5D=298&forums%5B%5D=299&forums%5B%5D=300&forums%5B%5D=301&forums%5B%5D=302&forums%5B%5D=62&forums%5B%5D=106&forums%5B%5D=13&forums%5B%5D=12&forums%5B%5D=36&forums%5B%5D=27&forums%5B%5D=137&forums%5B%5D=189&forums%5B%5D=330&forums%5B%5D=198&forums%5B%5D=99&forums%5B%5D=107&forums%5B%5D=170&forums%5B%5D=154&forums%5B%5D=231&forums%5B%5D=304&forums%5B%5D=153&forums%5B%5D=172&forums%5B%5D=235&forums%5B%5D=321&forums%5B%5D=327&forums%5B%5D=74&forums%5B%5D=68&forums%5B%5D=214&forums%5B%5D=67&forums%5B%5D=282&forums%5B%5D=233&forums%5B%5D=283&forums%5B%5D=95&forums%5B%5D=161&forums%5B%5D=197&forums%5B%5D=320&forums%5B%5D=313&forums%5B%5D=252&forums%5B%5D=31&forums%5B%5D=311&forums%5B%5D=42&forums%5B%5D=6&forums%5B%5D=145&forums%5B%5D=111&forums%5B%5D=183&forums%5B%5D=18&forums%5B%5D=241&forums%5B%5D=260&forums%5B%5D=246&forums%5B%5D=228&forums%5B%5D=96&forums%5B%5D=309&forums%5B%5D=243&forums%5B%5D=97&forums%5B%5D=108&forums%5B%5D=109&forums%5B%5D=110&forums%5B%5D=182&forums%5B%5D=253&forums%5B%5D=310&forums%5B%5D=82&forums%5B%5D=23&forums%5B%5D=175&forums%5B%5D=28&forums%5B%5D=239&forums%5B%5D=323&forums%5B%5D=20&forums%5B%5D=244&forums%5B%5D=247&forums%5B%5D=103&forums%5B%5D=312&forums%5B%5D=44&forums%5B%5D=117&forums%5B%5D=34&forums%5B%5D=53&forums%5B%5D=118&forums%5B%5D=194&forums%5B%5D=133&forums%5B%5D=285&forums%5B%5D=315&forums%5B%5D=236&forums%5B%5D=89&forums%5B%5D=195&forums%5B%5D=105&forums%5B%5D=190&forums%5B%5D=322&forums%5B%5D=85&forums%5B%5D=158&forums%5B%5D=284&forums%5B%5D=156&forums%5B%5D=157&forums%5B%5D=319&forums%5B%5D=318&forums%5B%5D=305&forums%5B%5D=119&forums%5B%5D=162&forums%5B%5D=15&forums%5B%5D=84&forums%5B%5D=11&forums%5B%5D=261&forums%5B%5D=276&forums%5B%5D=275&forums%5B%5D=278&forums%5B%5D=57&forums%5B%5D=55&forums%5B%5D=134&forums%5B%5D=266&forums%5B%5D=279&forums%5B%5D=155&forums%5B%5D=98&forums%5B%5D=267&forums%5B%5D=297&forums%5B%5D=265&forums%5B%5D=277&forums%5B%5D=54&forums%5B%5D=262&forums%5B%5D=132&forums%5B%5D=180&forums%5B%5D=238&forums%5B%5D=264&forums%5B%5D=135&forums%5B%5D=216&forums%5B%5D=217&forums%5B%5D=219&forums%5B%5D=268&forums%5B%5D=307&forums%5B%5D=72&forums%5B%5D=272&forums%5B%5D=273&forums%5B%5D=271&forums%5B%5D=274&forums%5B%5D=263&forums%5B%5D=113&forums%5B%5D=218&forums%5B%5D=269&forums%5B%5D=220&forums%5B%5D=224&forums%5B%5D=225&forums%5B%5D=226&forums%5B%5D=227&forums%5B%5D=237&forums%5B%5D=326&forums%5B%5D=270&forums%5B%5D=148&forums%5B%5D=128&forums%5B%5D=152&forums%5B%5D=296&searchsubs=1&prune=0&prune_type=newer&sort_key=rank&sort_order=desc&search_in=posts&result_type=posts"
        )
        headers = {
            "cookie": "REDACTED",
            "authority": "forum.lowyat.net",
            "cache-control": "max-age=0",
            "sec-ch-ua": '"Chromium";v="88", "Google Chrome";v="88", ";Not A Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "upgrade-insecure-requests": "1",
            "origin": "https://forum.lowyat.net",
            "content-type": "application/x-www-form-urlencoded",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "sec-fetch-site": "same-origin",
            "sec-fetch-mode": "navigate",
            "sec-fetch-user": "?1",
            "sec-fetch-dest": "document",
            "referer": "https://forum.lowyat.net/index.php?act=Search&f=",
            "accept-language": "en-US,en;q=0.9",
        }

        response = s.post(
            "https://forum.lowyat.net/index.php",
            data=payload,
            headers=headers,
            params=querystring,
        )

        # soup = BeautifulSoup(response.text, 'html.parser')
        # print(soup)
        # results = soup.find_all("div", class_="postcolor")
        # dates = soup.find_all("td", {"width":"100%!"}, class_="row2" )
        # posts_url = soup.find_all("div", class_="maintitle")
        # usernames_soup = soup.find_all("b")

        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.find_all("div", class_="postcolor")
        for x in results:
            if "QUOTE" in x.text:
                quote = x.find(class_="quotemain")
                quote.decompose()
                quote = x.find(class_="quotetop")
                quote.decompose()
                print(x.text)
                cleanPost = x.text.replace("(MISSING)", "")
                if "QUOTE(" in cleanPost:
                    cleanPost = cleanPost.split(")")[1]
                results_all.append(cleanPost.strip())
                keyword_used.append(keyword)
            else:
                cleanPost = x.text.replace("(MISSING)", "")
                results_all.append(cleanPost.strip())
                keyword_used.append(keyword)

        dates = soup.find_all(
            "td", {"width": "100%!"}, class_="row2", style="height:200px;overflow:auto"
        )
        print(dates)
        posts_url = soup.find_all("div", class_="maintitle")
        usernames_soup = soup.find_all("b")
        nextURL = soup.find("a", title="Jump to page...").get("href")
        print(nextURL)

        # for x in results:
        #     results_all.append(x.text)
        #     keyword_used.append(keyword)

        for x in posts_url:
            post_urls.append(x.contents[1].get("href"))

        for times in dates:
            today = datetime.date.today()
            yesterday = today - datetime.timedelta(days=1)
            date = times.text.replace("Today", str(today))
            date = date.replace("Yesterday", str(yesterday))
            dt = parse(date.split(":")[1] + ":" + date.split(":")[2])
            timestamps.append(dt.strftime("%b %d, %Y, %H:%M"))

        for x in usernames_soup:
            if "showuser" in str(x):
                usernames.append(str(x.text))
                profile_urls.append(str(x.contents[0].get("href")))

        count = 25

        while count < 250:
            tm.sleep(3)
            response_2 = s.get(response.url + "&st=" + str(count), headers=headers)
            soup = BeautifulSoup(response_2.text, "html.parser")
            souptext = BeautifulSoup(response_2.text, "html.parser")
            results = souptext.find_all("td", {"width": "100%"}, class_="post1")
            dates = soup.find_all("td", {"width": "100%"}, class_="row2")
            usernames_soup = soup.find_all("b")
            for x in results:
                if "QUOTE" in x.text:
                    quote = x.find(class_="quotemain")
                    quote.decompose()
                    quote = x.find(class_="quotetop")
                    quote.decompose()
                    cleanPost = x.text.replace("(MISSING)", "")
                    if "QUOTE(" in cleanPost:
                        cleanPost = cleanPost.split(")")[1]
                    results_all.append(cleanPost.strip())
                    keyword_used.append(keyword)
                else:
                    cleanPost = x.text.replace("(MISSING)", "")
                    results_all.append(cleanPost.strip())
                    keyword_used.append(keyword)
            # for x in results:
            #     results_all.append(x.text)
            #     keyword_used.append(keyword)

            for x in posts_url:
                post_urls.append(x.contents[1].get("href"))

            for time in dates:
                today = datetime.date.today()
                yesterday = today - datetime.timedelta(days=1)
                date = time.text.replace("Today", str(today))
                date = date.replace("Yesterday", str(yesterday))
                dt = parse(date.split(":")[1] + ":" + date.split(":")[2])
                timestamps.append(dt.strftime("%b %d, %Y, %H:%M"))

            for u in usernames_soup:
                if "showuser" in str(u):
                    usernames.append(str(u.text))
                    profile_urls.append(str(u.contents[0].get("href")))
            count += 25

    #     for result in results:
    #         if  any(x in result.text.lower() for x in positive):
    #             print('Positive Sentiment detected')
    #             positiveSentiments += 1
    #         elif  any(x in result.text.lower() for x in negative):
    #             print('Negative Sentiment detected')
    #             negativeSentiments += 1
    #         else:
    #             print('Neutral Sentiment detected')
    #             neutralSentiments += 1

    # print('--------------------------Final Results------------------------------')

    # print('Number of Positive Sentiments: '+ str(positiveSentiments))
    # print('Number of Negative Sentiments: '+ str(negativeSentiments))
    # print('Number of Neutral Sentiments: '+ str(neutralSentiments))

    sentiment_results = []

    for x in results_all:
        result_1 = predict_sentiment(x)
        sentiment_results.append(str(result_1))

    for time, post, keyword, url, username, profile_url, result in zip(
        timestamps,
        results_all,
        keyword_used,
        post_urls,
        usernames,
        profile_urls,
        sentiment_results,
    ):
        lowyat_entries.append(
            {
                "source": "lowyat",
                "timestamp": str(time),
                "keyword": str(keyword),
                "post": str(post),
                "name": "-",
                "username": username,
                "post_url": url,
                "user_url": profile_url,
                "sentiment": result,
            }
        )

    lowyat_entries_all = []
    post_list_db = sh_entry.col_values(4)
    duplicate_count = 0
    for x in lowyat_entries:
        lowyat_entry = []
        lowyat_entry.append(str(x["source"]))
        lowyat_entry.append(str(x["timestamp"]))
        lowyat_entry.append(str(x["keyword"]))
        lowyat_entry.append(str(x["post"]))
        lowyat_entry.append(str(x["name"]))
        lowyat_entry.append(str(x["username"]))
        lowyat_entry.append(str(x["post_url"]))
        lowyat_entry.append(str(x["user_url"]))
        lowyat_entry.append(str(x["sentiment"]))
        if x["post"] in post_list_db:
            print("Duplicate Found. Removing...")
            duplicate_count += 1
        else:
            lowyat_entries_all.append(lowyat_entry)

    print(duplicate_count, "duplicates found")

    sh_entry.append_rows(lowyat_entries_all)

    return


def twitterScraper():
    url = "https://api.twitter.com/2/tweets/search/recent"

    bearerToken = "REDACTED"

    querystring = {
        "query": '"kw1" - is:retweet',
        "tweet.fields": "created_at",
        "expansions": "author_id",
        "max_results": "100",
    }

    payload = ""
    headers = {"Authorization": "Bearer " + bearerToken}

    twitter_response = requests.request(
        "GET", url, data=payload, headers=headers, params=querystring
    )

    response_json = twitter_response.json()

    twitter_results = []
    usernames = []
    names = []
    timestamps = []
    tweet_urls = []
    profile_urls = []

    for x in response_json["data"]:
        twitter_results.append(x["text"])
        twitter_id = x["id"]
        author_id = x["author_id"]

        dt = parse(x["created_at"])
        dt = dt + datetime.timedelta(hours=8)
        timestamps.append(dt.strftime("%b %d, %Y, %H:%M"))

        for y in response_json["includes"]["users"]:
            if y["id"] == author_id:
                tweet_username = y["username"]
                usernames.append(y["username"])
                names.append(y["name"])
                url = "https://twitter.com/" + tweet_username + "/status/" + twitter_id
                tweet_urls.append(url)
                profile_urls.append("https://twitter.com/" + tweet_username)

    twitter_entries = []

    sentiment_results = []

    for x in twitter_results:
        result_1 = predict_sentiment(x)
        sentiment_results.append(str(result_1))

    data_rows = zip(
        timestamps,
        twitter_results,
        names,
        usernames,
        tweet_urls,
        profile_urls,
        sentiment_results,
    )

    for time, tweet, name, username, tweet_url, profile_url, result in data_rows:
        twitter_entries.append(
            {
                "source": "twitter",
                "timestamp": str(time),
                "keyword": "kw",
                "tweet": str(tweet),
                "name": name,
                "username": username,
                "post_url": tweet_url,
                "user_url": profile_url,
                "sentiment": result,
            }
        )

    twitter_entries_all = []
    tweet_list_db = sh_entry.col_values(4)
    duplicate_count = 0
    for x in twitter_entries:
        twitter_entry = []
        twitter_entry.append(str(x["source"]))
        twitter_entry.append(str(x["timestamp"]))
        twitter_entry.append(str(x["keyword"]))
        twitter_entry.append(str(x["tweet"]))
        twitter_entry.append(str(x["name"]))
        twitter_entry.append(str(x["username"]))
        twitter_entry.append(str(x["post_url"]))
        twitter_entry.append(str(x["user_url"]))
        twitter_entry.append(str(x["sentiment"]))
        if x["tweet"] in tweet_list_db:
            print("Duplicate Found. Removing...")
            duplicate_count += 1
        else:
            twitter_entries_all.append(twitter_entry)

    print(duplicate_count, "duplicates found")

    sh_entry.append_rows(twitter_entries_all)


def facebookScraper():

    post_type = []
    fb_results = []
    usernames = []
    names = []
    timestamps = []
    post_urls = []
    profile_urls = []

    for post in get_posts(
        "kw1",
        pages=10,
        cookies="fbcookies.txt",
        options={"comments": True, "reactors": True},
    ):
        post_type.append("FB-Post")
        fb_results.append(post["text"])
        usernames.append(post["username"])
        names.append(post["username"])
        post_urls.append(post["post_url"])
        profile_urls.append(post["user_url"])
        timestamps.append(post["time"].strftime("%b %d, %Y, %H:%M"))

        for comments in post["comments_full"]:
            post_type.append("FB-Comment")
            fb_results.append(comments["comment_text"])
            usernames.append(comments["commenter_name"])
            names.append(comments["commenter_name"])
            post_urls.append(comments["comment_url"])
            profile_urls.append(comments["commenter_url"])
            timestamps.append(comments["comment_time"].strftime("%b %d, %Y, %H:%M"))
            for replies in comments["replies"]:
                post_type.append("FB-Comment-Reply")
                fb_results.append(replies["comment_text"])
                usernames.append(replies["commenter_name"])
                names.append(replies["commenter_name"])
                post_urls.append(replies["comment_url"])
                profile_urls.append(replies["commenter_url"])
                timestamps.append(replies["comment_time"].strftime("%b %d, %Y, %H:%M"))

    twitter_entries = []

    sentiment_results = []

    for x in fb_results:
        result_1 = predict_sentiment(x)
        sentiment_results.append(str(result_1))

    data_rows = zip(
        post_type,
        timestamps,
        fb_results,
        names,
        usernames,
        post_urls,
        profile_urls,
        sentiment_results,
    )

    for type, time, tweet, name, username, tweet_url, profile_url, result in data_rows:
        twitter_entries.append(
            {
                "source": type,
                "timestamp": str(time),
                "keyword": "kw1",
                "tweet": str(tweet),
                "name": name,
                "username": username,
                "post_url": tweet_url,
                "user_url": profile_url,
                "sentiment": result,
            }
        )

    twitter_entries_all = []
    tweet_list_db = sh_entry.col_values(4)
    duplicate_count = 0
    for x in twitter_entries:
        twitter_entry = []
        twitter_entry.append(str(x["source"]))
        twitter_entry.append(str(x["timestamp"]))
        twitter_entry.append(str(x["keyword"]))
        twitter_entry.append(str(x["tweet"]))
        twitter_entry.append(str(x["name"]))
        twitter_entry.append(str(x["username"]))
        twitter_entry.append(str(x["post_url"]))
        twitter_entry.append(str(x["user_url"]))
        twitter_entry.append(str(x["sentiment"]))
        if x["tweet"] in tweet_list_db:
            print("Duplicate Found. Removing...")
            duplicate_count += 1
        else:
            twitter_entries_all.append(twitter_entry)

    print(duplicate_count, "duplicates found")

    sh_entry.append_rows(twitter_entries_all)


def main():
    a=1
    while a < 10:
        lowyat_search(keywords=["kw1", "kw2"])
        twitterScraper()
        facebookScraper()
        tm.sleep(2400)
    return


if __name__ == "__main__":
    main()
