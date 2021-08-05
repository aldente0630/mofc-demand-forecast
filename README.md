# MOFC Sales Forecasting with Time Series Analysis 
### Goals
* Compare the accuracy of various time series forecasting algorithms such as *Prophet*, *DeepAR*, *VAR*, *DeepVAR*, and *LightGBM*

### Requirements
* The dataset can be downloaded from [this Kaggle competition](https://www.kaggle.com/c/m5-forecasting-accuracy).
* In addition to the [Anaconda](https://www.anaconda.com) libraries, you need to install `altair`, `vega_datasets`, `category_encoders`, `mxnet`, `gluonts`, `kats` and `pandarallel`.
* `kats` requires Python 3.7 or higher.

## Competition, Datasets and Evaluation

## Algorithms
### Kats: Prophet
![Forecasting-1](./img/prophet-1.svg)
![Forecasting-2](./img/prophet-2.svg)
![Forecasting-3](./img/prophet-3.svg)

### Kats: VAR
![Forecasting-1](./img/var-1.svg)
![Forecasting-2](./img/var-2.svg)
![Forecasting-3](./img/var-3.svg)

### GluonTS: DeepAR
![Forecasting-1](./img/deepar-1.svg)
![Forecasting-2](./img/deepar-2.svg)
![Forecasting-3](./img/deepar-3.svg)

### GluonTS: DeepVAR
![Forecasting-1](./img/deepvar-1.svg)
![Forecasting-2](./img/deepvar-2.svg)
![Forecasting-3](./img/deepvar-3.svg)

## Algorithms Performance Summary
|Estimator|WRMSSE|MAPE|sMAPE|MAE|MASE|RMSE|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|DeepAR|0.7200|0.7749|1.5335|26.8035|1.3088|2.0850|
|VAR|0.7529||||||
|Prophet|0.8690|∞|1.4120|1.1167|1.6707|1.417|
|DeepVAR|2.1639|0.2100|0.2461|0.0134|2.2618|0.4092|

### References
* [Taylor SJ, Letham B. 2017. Forecasting at scale. *PeerJ Preprints* 5:e3190v2](https://peerj.com/preprints/3190.pdf)
* [Stock, James, H., Mark W. Watson. 2001. Vector Autoregressions. *Journal of Economic Perspectives*, 15 (4): 101-115.](https://www.princeton.edu/~mwatson/papers/Stock_Watson_JEP_2001.pdf)
* [David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski. 2020. DeepAR: Probabilistic forecasting with autoregressive recurrent networks, *International Journal of Forecasting*, 36 (3): 1181-1191.](https://arxiv.org/pdf/1704.04110.pdf)
* [David Salinas, Michael Bohlke-Schneider, Laurent Callot, Roberto Medico,
Jan Gasthaus. 2019. High-dimensional multivariate forecasting with low- rank Gaussian Copula Processes. *In Advances in Neural Information Processing Systems*. 6827–6837.](https://arxiv.org/pdf/1910.03002.pdf)
* [Kats - One Stop Shop for Time Series Analysis in Python](https://facebookresearch.github.io/Kats/)
* [GluonTS - Probabilistic Time Series Modeling](https://ts.gluon.ai/index.html)