#!/usr/bin/env python
# coding: utf-8

# ### PROJECT - SOCIAL MEDIA PERFORMANCE 

# #### Loading python libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("modules are sucessfully installed")


# #### Loading the data

# In[2]:


smp = pd.read_csv("social_media_performance.csv")


# In[3]:


smp.head()


# In[4]:


smp.tail()


# In[5]:


smp.info()


# In[6]:


smp.shape


# In[7]:


smp.describe()


# In[9]:


smp.describe(include="O")


# In[10]:


smp["post_datetime"] = pd.to_datetime(
    smp["post_datetime"],
    errors="coerce"
)


# In[11]:


smp.describe()


# In[12]:




smp.isnull().sum()


# In[13]:


missing_percentage = (smp.isnull().sum() / len(smp)) * 100

missing_percentage


# In[14]:


smp.duplicated().sum()


# In[15]:


smp["post_id"].duplicated().sum()


# In[16]:


smp.dtypes


# In[17]:


smp["post_datetime"] = pd.to_datetime(
    smp["post_datetime"],
    errors="coerce"
)


# In[18]:


smp.dtypes


# In[19]:


smp["post_datetime"].isnull().sum()


# In[20]:


numeric_columns = [
    "sentiment_score",
    "views",
    "likes",
    "comments",
    "shares",
    "engagement_rate",
    "is_viral"
]

smp[numeric_columns].dtypes


# In[21]:


performance_columns = [
    "views",
    "likes",
    "comments",
    "shares",
    "engagement_rate"
]

(smp[performance_columns] < 0).sum()


# In[22]:


smp["sentiment_score"].min(), smp["sentiment_score"].max()


# In[23]:


invalid_sentiment = smp[
    (smp["sentiment_score"] < -1) |
    (smp["sentiment_score"] > 1)
]

invalid_sentiment.shape


# In[24]:


smp["is_viral"].unique()


# In[25]:


smp["is_viral"].value_counts()


# In[26]:


smp["post_year"] = smp["post_datetime"].dt.year


# In[27]:


smp["post_month"] = smp["post_datetime"].dt.month


# In[28]:


smp["post_month_name"] = smp["post_datetime"].dt.month_name()


# In[29]:


smp["post_day"] = smp["post_datetime"].dt.day


# In[30]:


smp["day_of_week"] = smp["post_datetime"].dt.day_name()


# In[31]:


smp["post_hour"] = smp["post_datetime"].dt.hour


# In[32]:


smp.dtypes


# In[33]:


smp["day_type"] = np.where(
    smp["post_datetime"].dt.dayofweek >= 5,
    "Weekend",
    "Weekday"
)


# In[34]:


def categorize_sentiment(score):
    if score < -0.33:
        return "Negative"
    elif score <= 0.33:
        return "Neutral"
    else:
        return "Positive"

smp["sentiment_category"] = smp["sentiment_score"].apply(
    categorize_sentiment
)


# In[35]:


smp["hashtags"].head()


# In[36]:


smp["hashtag_count"] = (
    smp["hashtags"]
    .fillna("")
    .str.split()
    .str.len()
)


# In[37]:


smp["hashtag_count"].describe()


# In[38]:


platform_counts = smp["platform"].value_counts()

platform_counts


# In[39]:


plt.figure(figsize=(8, 5))

sns.countplot(
    data=smp,
    x="platform"
)

plt.title("Distribution of Posts by Platform")
plt.xlabel("Platform")
plt.ylabel("Number of Posts")
plt.xticks(rotation=45)
plt.show()


# In[40]:


smp["content_type"].value_counts()


# In[41]:


plt.figure(figsize=(9, 5))

sns.countplot(
    data=smp,
    x="content_type"
)

plt.title("Distribution of Content Types")
plt.xlabel("Content Type")
plt.ylabel("Number of Posts")
plt.xticks(rotation=45)
plt.show()


# In[42]:


plt.figure(figsize=(9, 5))

sns.histplot(
    data=smp,
    x="views",
    bins=30,
    kde=True
)

plt.title("Distribution of Post Views")
plt.xlabel("Views")
plt.ylabel("Frequency")
plt.show()


# In[43]:


plt.figure(figsize=(9, 5))

sns.histplot(
    data=smp,
    x="engagement_rate",
    bins=30,
    kde=True
)

plt.title("Distribution of Engagement Rate")
plt.xlabel("Engagement Rate")
plt.ylabel("Frequency")
plt.show()


# In[44]:


plt.figure(figsize=(8, 5))

sns.boxplot(
    data=smp,
    y="engagement_rate"
)

plt.title("Distribution and Outliers of Engagement Rate")
plt.ylabel("Engagement Rate")
plt.show()


# In[45]:


platform_engagement = (
    smp.groupby("platform")["engagement_rate"]
    .mean()
    .sort_values(ascending=False)
)

platform_engagement


# In[46]:


plt.figure(figsize=(9, 5))

sns.barplot(
    x=platform_engagement.index,
    y=platform_engagement.values
)

plt.title("Average Engagement Rate by Platform")
plt.xlabel("Platform")
plt.ylabel("Average Engagement Rate")
plt.xticks(rotation=45)
plt.show()


# In[47]:


content_engagement = (
    smp.groupby("content_type")["engagement_rate"]
    .mean()
    .sort_values(ascending=False)
)

content_engagement


# In[48]:


plt.figure(figsize=(9, 5))

sns.barplot(
    x=content_engagement.index,
    y=content_engagement.values
)

plt.title("Average Engagement Rate by Content Type")
plt.xlabel("Content Type")
plt.ylabel("Average Engagement Rate")
plt.xticks(rotation=45)
plt.show()


# In[49]:


plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=smp,
    x="views",
    y="likes",
    alpha=0.5
)

plt.title("Relationship Between Views and Likes")
plt.xlabel("Views")
plt.ylabel("Likes")
plt.show()


# In[50]:


plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=smp,
    x="views",
    y="shares",
    alpha=0.5
)

plt.title("Relationship Between Views and Shares")
plt.xlabel("Views")
plt.ylabel("Shares")
plt.show()


# In[51]:


sentiment_engagement = (
    smp.groupby("sentiment_category")["engagement_rate"]
    .mean()
)

sentiment_engagement


# In[52]:


plt.figure(figsize=(8, 5))

sns.barplot(
    x=sentiment_engagement.index,
    y=sentiment_engagement.values
)

plt.title("Average Engagement Rate by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Average Engagement Rate")
plt.show()


# In[53]:


monthly_posts = (
    smp.groupby("post_month")
    .size()
)

monthly_posts


# In[54]:


plt.figure(figsize=(10, 5))

sns.lineplot(
    x=monthly_posts.index,
    y=monthly_posts.values,
    marker="o"
)

plt.title("Number of Posts by Month")
plt.xlabel("Month")
plt.ylabel("Number of Posts")
plt.xticks(range(1, 13))
plt.show()


# In[56]:


day_engagement = (
    smp.groupby("day_of_week")["engagement_rate"]
    .mean()
)


# In[57]:


day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


# In[58]:


plt.figure(figsize=(10, 5))

sns.barplot(
    data=smp,
    x="day_of_week",
    y="engagement_rate",
    order=day_order
)

plt.title("Average Engagement Rate by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Engagement Rate")
plt.xticks(rotation=45)
plt.show()


# In[59]:


hour_engagement = (
    smp.groupby("post_hour")["engagement_rate"]
    .mean()
)


# In[60]:


plt.figure(figsize=(10, 5))

sns.lineplot(
    x=hour_engagement.index,
    y=hour_engagement.values,
    marker="o"
)

plt.title("Average Engagement Rate by Posting Hour")
plt.xlabel("Posting Hour")
plt.ylabel("Average Engagement Rate")
plt.xticks(range(0, 24))
plt.show()


# In[61]:


smp.groupby("day_type")["engagement_rate"].mean()


# In[62]:


plt.figure(figsize=(7, 5))

sns.barplot(
    data=smp,
    x="day_type",
    y="engagement_rate"
)

plt.title("Average Engagement: Weekday vs Weekend")
plt.xlabel("Day Type")
plt.ylabel("Average Engagement Rate")
plt.show()


# In[63]:


platform_content = pd.pivot_table(
    smp,
    values="engagement_rate",
    index="platform",
    columns="content_type",
    aggfunc="mean"
)

platform_content


# In[64]:


plt.figure(figsize=(12, 7))

sns.heatmap(
    platform_content,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Average Engagement Rate by Platform and Content Type")
plt.xlabel("Content Type")
plt.ylabel("Platform")
plt.show()


# In[65]:


viral_platform = pd.crosstab(
    smp["platform"],
    smp["is_viral"]
)

viral_platform


# In[66]:


viral_platform.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Viral vs Non-Viral Posts by Platform")
plt.xlabel("Platform")
plt.ylabel("Number of Posts")
plt.xticks(rotation=45)
plt.legend(
    title="Viral Status",
    labels=["Non-Viral", "Viral"]
)
plt.show()


# In[67]:


correlation_columns = [
    "views",
    "likes",
    "comments",
    "shares",
    "engagement_rate",
    "sentiment_score"
]

correlation = smp[correlation_columns].corr()

correlation


# In[68]:


plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix of Social Media Performance Metrics")
plt.show()


# In[69]:


plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=smp,
    x="hashtag_count",
    y="engagement_rate",
    alpha=0.5
)

plt.title("Hashtag Count vs Engagement Rate")
plt.xlabel("Number of Hashtags")
plt.ylabel("Engagement Rate")
plt.show()


# In[ ]:




