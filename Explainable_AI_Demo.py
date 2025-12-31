#install and import shap
!pip install shap
import shap
shap.initjs
#create shap explainer
explainer=shap.TreeExplainer(model)
#calculate shap values
shap_values=explainer.shap_values(x_train)
#global explaination
shap.summary_plot(shap_values,x_train)
