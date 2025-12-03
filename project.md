<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## My Project

I applied machine learning techniques to investigate PM2.5 (particulate matter of 2.5μm or less) TW (tire wear) emissions that vehicles release. 

***

## Introduction 

Electric vehicles (EVs) are often described as “zero-emission vehicles,” a label that plays a central role in climate policy and public perception. When our team first began this project through UCLA’s International Urban Sustainability Student Corps (IUSSC) in partnership with the California Air Resources Board (CARB), we shared that same assumption. If EVs eliminate tailpipe pollution, then transitioning to them should automatically reduce particulate matter exposure in cities like Los Angeles. As we dug deeper, we learned that this may not be the case.

While EVs eliminate tailpipe emissions, they still produce pollution from non-exhaust sources, particularly tire and brake wear. These forms of PM2.5 (particulate matter of 2.5μm or less) enter the air regardless of whether a car runs on gasoline or electricity. This becomes especially important when considering that EVs are, on average, about 20 percent heavier than comparable gasoline vehicles due to their batteries. That added weight increases friction and stress on tires, which in turn leads to higher levels of tire-wear particles. As Los Angeles accelerates its transition to electric vehicles, understanding the implications of these overlooked emissions is becoming increasingly urgent.

With the current context of research in non-exhaust emissions of tire wear, there are many complications that arise when exploring this field. Throughout the 850 peer-reviewed scientific publications that were analyzed about tire wear emissions, the current knowledge base on tire wear emissions is scattered [1]. There are varying measurement methodologies that one approaches, but there are also many explanatory variables that affect the amount of tire wear emissions that cannot be fully controlled in a highly advanced laboratory setting.

To explore this issue with acknowledgement of the current research field, we turned to CARB’s EMFAC emissions model, focusing specifically on tire-wear PM2.5 in Los Angeles County. We performed machine learning to understand how to predict PM2.5 TW given other explanatory variables.  The objectives of this research are threefold: (1) develop machine learning models for predicting PM2.5 emissions from tire wear using vehicle and driving behavior patterns, (2) compare multiple algorithm classes using k-fold cross-validation and REC curves, and (3) identify the most accurate and stable modeling framework for non-exhaust PM2.5 TW prediction. By leveraging ensemble learning methods, this study seeks to advance predictive modeling in non-exhaust emissions and contribute to the broader adoption of data-driven approaches in the clean energy transportation sector.

## Data

The EMFAC (Emission Factors) model is an on-road mobile emissions model developed by the California Air Resources Board (CARB). It is used by CARB to support public policy and regulatory decision-making and is also publicly available for research and planning purposes. In this study, we investigate differences between gasoline and electric vehicles to understand how PM₂.₅ tire-wear (TW) emissions can be predicted across these categories.

Initially, the dataset contained n = 13 gasoline vehicles and n = 41 electric vehicles, indicating a substantial class imbalance. To address this sampling gap, natural gas vehicles (n = 29) were reclassified under the gasoline vehicle category. This pre-processing step was implemented to achieve a more approximately balanced sample between the gasoline and electric vehicle groups, thereby improving the statistical reliability of subsequent modeling.

We next developed a visualization to examine the distribution of PM2.5 TW values within the EMFAC dataset. From initial observations, the distribution is highly right-skewed, with the majority of PM2.5 TW values concentrated at the lower end of the metric range. This concentration indicates that most vehicles exhibit relatively low PM2.5 TW emissions, with only a small number producing substantially higher values. This distribution is illustrated in Figure 1.

![](assets/IMG/pre_processing.png)

*Figure 1: Pre-Processed PM2.5 TW Data of EMFAC [1].*

To improve the normality of the response variable, a logarithmic transformation was applied to the PM2.5 tire-wear (TW) values. This transformation reduces the influence of extreme values and mitigates the strong right-skew observed in the original distribution, resulting in data that are better suited for regression-based and machine learning models. The resulting transformed distribution is shown in Figure 2.

![](assets/IMG/post_processing.png)

*Figure 2: Post-Processed PM2.5 TW Data of EMFAC through Logarithmic Transformation  [2].*

With the data now normalized through the logarithmic transformation, PM2.5 tw can be analyzed more effectively within the modeling framework. Before proceeding further, it is important to examine how the behavior of PM₂.₅ TW is represented within the emissions modeling data. In this field of research, PM₂.₅ TW is often evaluated per vehicle miles traveled (VMT) to quantify the rate at which tire-wear particles are emitted over time and distance.

By incorporating machine learning to predict PM2.5 TW, we aim to evaluate whether electric vehicles exhibit a higher PM2.5 TW emission rate per VMT compared to gasoline vehicles, a hypothesis motivated by differences in vehicle mass and torque. Figure 3 presents a bar chart comparison between electric and gasoline vehicles, highlighting a key methodological change between the EMFAC 2021 and EMFAC 2025. The EMFAC 2025 model incorporates vehicle weight dependence, whereas the EMFAC 2021 model treated tire-wear emissions as weight-independent; hence, the PM2.5 TW emission rate per VMT between gasoline and electric vehicles was the same.

![](assets/IMG/pm_vmt.png)

*Figure 3: PM2.5 per VMT Bar Chart Comparison Between Electric and Gasoline Vehicles EMFAC 2025 [3].*

## Modelling

For this project, the main purpose of the investigation is to test some modeling algorithms that would best fit the dataset. Hence, I will be testing linear regression, ridge regression, neural network, random forest, decision tree regression, and simple vector regression (SVR). These methods are all commonly used in machine learning that help with quantitative predictions of the response variables, given extensive information about the explanatory variables. Each of these tactics has its own strengths and weaknesses so I will test the accuracy of each algorithm.

### Linear Regression

When selectng the machine learning model that I can work with, I fist thought that linear regression would be a 

### Ridge Regression

### Neural Network

### Random Forest Regression

### Decision Tree

### SVR

Here are some more details about the machine learning approach and why this was deemed appropriate for the dataset. 

<p>
When \(a \ne 0\), there are two solutions to \(ax^2 + bx + c = 0\) and they are
  \[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]
</p>

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

![](assets/IMG/dt_unconstrained.png)

![](assets/IMG/dt_constrained.png)

![](assets/IMG/dt_actual_true.png)

![](assets/IMG/rf_actual_true.png)

![](assets/IMG/rf_feature_importance.png)

![](assets/IMG/rec_combo.png)


Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] https://tireindustryproject.org/news/scientists-call-for-deeper-investigation-and-standardization-of-methodologies-for-measuring-and-assessing-tire-wear-emissions-in-first-of-its-kind-state-of-knowledge-papers/?utm_campaign=&utm_content=8e23ffc2-09a1-42b7-807b-1e64c46b25d7&utm_medium=linkedin&utm_term=continental&utm_source

[back](./)

