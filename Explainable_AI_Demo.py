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
#waterfall plot
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=x_test.iloc[0],
        feature_names=x_test.columns
    )
)
