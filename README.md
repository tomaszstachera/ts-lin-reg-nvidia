# NVIDIA stock prediction with linear regression and NumPy

While climbing the ladder of Data Science techniques there must be a place for good old linear regression. So let's try to predict stock prices of a quite demanding security - NVDA (NVIDIA stocks). Demanding because of latest [turmoils](https://www.reuters.com/technology/chinas-deepseek-sets-off-ai-market-rout-2025-01-27/) connected to the release of DeepSeekAI. During one of Data Science meetups that I've attended presenter shared financial stock prediciton results of being better than market in 56% times as a very good achievement. Taking this into account my final orange prediciton plot may not look so silly ;)

![final_plot](img/final_plot.png "final_plot")

## Data preparation / feature selection

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

Quick look at the correlation between features shows that some of them (obviously) are highly correlated, but let's keep them and not overengineer this simple example. Also we see that three features most correlates with NVDA price (`Close`) are `ai`, `nvidia` and `chatgpt` - all over 0.85.

![corr](img/corr.png "corr")

## First tuning

Based on few observations I've generated first set of hyperparameters, which is split of train/test data and set of features taken into model:

![hyper1](img/hyper1.png "hyper1")

R-squarred error plotted below gives us some clues where to search for best results - low split and small amount of features.

![r2-1](img/r2-1.png "r2-1")

## Final tuning

Let's generate more dense set of hyperparameters:

![hyper2](img/hyper2.png "hyper2")

And plot final set of extended metrics:

- for features of : `chatgpt`
![metrics_1f](img/metrics_1f.png "metrics_1f")

- for features of : `chatgpt`, `ai`
![metrics_2f](img/metrics_2f.png "metrics_2f")

- for features of : `chatgpt`, `ai`, `nvidia`
![metrics_3f](img/metrics_3f.png "metrics_3f")

- for features of : `chatgpt`, `ai`, `nvidia`, `gpu`
![metrics_4f](img/metrics_4f.png "metrics_4f")

- for features of : `chatgpt`, `ai`, `nvidia`, `gpu`, `graphics_card`
![metrics_5f](img/metrics_5f.png "metrics_5f")

## Results and conclusions

Taking into account only R-squarred error we can pick 5 top values of this metrics and one best set of features (which is the simplest model with only one feature `chatgpt` and train/test split of 0.653).

![final_res](img/final_res.png "final_res")

At the end we compare price predicted by best model (`orange`) with actual price (`blue`) and Google Trends popularity of phrase "chatgpt" (`green`).

![final_plot](img/final_plot.png "final_plot")
