<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## My Project

I applied machine learning techniques to investigate PM2.5 (particulate matter of 2.5μm or less) TW (tire wear) emissions that vehicles release. 

***

## Introduction 

Electric vehicles (EVs) are often described as “zero-emission vehicles,” a label that plays a central role in climate policy and public perception. When our team first began this project through UCLA’s International Urban Sustainability Student Corps (IUSSC) in partnership with the California Air Resources Board (CARB), we shared that same assumption. If EVs eliminate tailpipe pollution, then transitioning to them should automatically reduce particulate matter exposure in cities like Los Angeles. As we dug deeper, we learned that this may not be the case.

While EVs eliminate tailpipe emissions, they still produce pollution from non-exhaust sources, particularly tire and brake wear. These forms of PM2.5 (particulate matter of 2.5μm or less) enter the air regardless of whether a car runs on gasoline or electricity. This becomes especially important when considering that EVs are, on average, about 20 percent heavier than comparable gasoline vehicles due to their batteries. That added weight increases friction and stress on tires, which in turn leads to higher levels of tire-wear particles. As Los Angeles accelerates its transition to electric vehicles, understanding the implications of these overlooked emissions is becoming increasingly urgent.

With the current context of research in non-exhaust emissions of tire wear, there are many complications that arise when exploring this field. Throughout the 850 peer-reviewed scientific publications that were analyzed about tire wear emissions, the current knowledge base on tire wear emissions is scattered. There are varying measurement methodologies that one approaches, but there are also many explanatory variables that affect the amount of tire wear emissions that cannot be fully controlled in a highly advanced laboratory setting.

To explore this issue with acknowledgement of the current research field, we turned to CARB’s EMFAC emissions model, focusing specifically on tire-wear PM2.5 in Los Angeles County. We performed machine learning to understand how to predict PM2.5 TW given other explanatory variables.  The objectives of this research are threefold: (1) develop machine learning models for predicting PM2.5 emissions from tire wear using vehicle and driving behavior patterns, (2) compare multiple algorithm classes using k-fold cross-validation and REC curves, and (3) identify the most accurate and stable modeling framework for non-exhaust PM2.5 TW prediction. By leveraging ensemble learning methods, this study seeks to advance predictive modeling in non-exhaust emissions and contribute to the broader adoption of data-driven approaches in the clean energy transportation sector.

## Data

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!

![](assets/IMG/datapenguin.png){: width="500" }

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

## Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

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

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] DALL-E 3

[back](./)

