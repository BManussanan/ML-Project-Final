import joblib
import pandas as pd
import gradio as gr

attrition_model = joblib.load("rf_selected_features_model.joblib")

model_features = [
    'StockOptionLevel','Age','EnvironmentSatisfaction','JobLevel',
    'JobRole_Laboratory Technician','MonthlyIncome','YearsAtCompany',
    'JobSatisfaction','YearsSinceLastPromotion','JobRole_Healthcare Representative',
    'BusinessTravel_Travel_Frequently','PercentSalaryHike','YearsInCurrentRole',
    'OverTime_Yes','DailyRate'
]

def predict_attrition(age, overtime, monthly_income, years_at_company, years_in_current_role,
                      job_satisfaction, env_satisfaction, job_level, business_travel):

    sample = pd.DataFrame(0, index=[0], columns=model_features)

    sample['Age'] = age
    sample['OverTime_Yes'] = 1 if overtime == "Yes" else 0
    sample['MonthlyIncome'] = monthly_income
    sample['YearsAtCompany'] = years_at_company
    sample['YearsInCurrentRole'] = years_in_current_role
    sample['JobSatisfaction'] = job_satisfaction
    sample['EnvironmentSatisfaction'] = env_satisfaction
    sample['JobLevel'] = job_level
    sample['BusinessTravel_Travel_Frequently'] = 1 if business_travel == "Travel_Frequently" else 0

    sample['StockOptionLevel'] = 0
    sample['YearsSinceLastPromotion'] = 0
    sample['PercentSalaryHike'] = 10
    sample['DailyRate'] = 500
    sample['JobRole_Laboratory Technician'] = 0
    sample['JobRole_Healthcare Representative'] = 0

    pred = attrition_model.predict(sample)[0]
    return "Yes (ลาออก)" if pred == 1 else "No (ไม่ลาออก)"

overtime = ['Yes', 'No']
business_travel = ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        age = gr.Slider(label="Age", minimum=18, maximum=60, step=1)
        monthly_income = gr.Number(label="Monthly Income", minimum=0)

    with gr.Row():
        years_at_company = gr.Slider(label="Years At Company", minimum=0, maximum=40, step=1)
        years_in_current_role = gr.Slider(label="Years In Current Role", minimum=0, maximum=20, step=1)

    with gr.Row():
        job_satisfaction = gr.Slider(label="Job Satisfaction (1-4)", minimum=1, maximum=4, step=1)
        env_satisfaction = gr.Slider(label="Environment Satisfaction (1-4)", minimum=1, maximum=4, step=1)

    with gr.Row():
        job_level = gr.Slider(label="Job Level", minimum=1, maximum=5, step=1)
        overtime_input = gr.Dropdown(overtime, label="OverTime")
        business_travel_input = gr.Dropdown(business_travel, label="Business Travel")

    predict_btn = gr.Button("Predict Attrition", variant="primary")
    result = gr.Textbox(label="Prediction Result")

    inputs = [age, overtime_input, monthly_income, years_at_company, years_in_current_role,
              job_satisfaction, env_satisfaction, job_level, business_travel_input]

    predict_btn.click(predict_attrition, inputs=inputs, outputs=[result])

if __name__ == "__main__":
    demo.launch(share=True)
