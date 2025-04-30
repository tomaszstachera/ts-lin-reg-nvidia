# NVIDIA stock prediction with linear regression and NumPy

While climbing the ladder of Data Science techniques there must be a place for good old linear regression. So let's try to predict stock prices of a quite demanding security - NVDA (NVIDIA stocks). Demanding because of latest [turmoils](https://www.reuters.com/technology/chinas-deepseek-sets-off-ai-market-rout-2025-01-27/) connected to the release of DeepSeekAI. During one of Data Science meetups that I've attended presenter shared financial stock prediciton results of being better than market in 56% times as a very good achievement. Taking this into account my final orange prediciton plot may not look so silly ;)

![final_plot](img/final_plot.png "final_plot")

## Data preparation

I came up with and idea that popularity of topics related to NVIDIA may be a decend indicator of the demand of NVIDIA shares. Since the start of the AI revolution NVIDIA is widely associated with this hype, because of GPU's role in the process of ML models creation and usage. Neverthless, still wide range of people not excited with AI may associate NVIDIA as leading graphics cards manufacturer. Taking all into account I've decided to use Google Trends data as a represantation of mentioned topics popularity as shown in the chart below these topics are more lesss correlated with NVIDIA popularity.

![gtrends](img/google-trends-comparison.png "google_trends")

If we compare this data with actual NVDA stock prices we can see also some correlation:

![feat_targ](img/feat_and_targ.png "feat_targ")

I am an MLOps to want to automate as much as possible. I wanted to do it also for source data preparation, found two libraries that potentially could pull Google Trends data programmatically - [`pytrends`](https://github.com/GeneralMills/pytrends) and [`trendspy`](https://github.com/sdil87/trendspy). Unfortunately Google don't want their API to be overwhelmend (probably that's why there is no official SDK) and my automation attempts failed.

See `trendspy` attempt:

![trendspy](img/trendspy-429.png "trendspy")

Check out `pytrends` attempt:

![pytrends](img/pytrends-429.png "pytrends")

So I needed to pull data manually, do some data preprocessing as below and data was ready for testing.


![data_prep](img/data_prep.png "data_prep")

## Feature selection

...

## First tuning

...

## Final tuning

...

## Results and conclusions

...
