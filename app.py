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

satisfaction_map = {
    "1 - Bad (แย่สุด)": 1,
    "2 - Moderate (ปานกลาง)": 2,
    "3 - Good (ดี)": 3,
    "4 - Very Good (ดีมาก)": 4
}

job_level_map = {
    "Entry-level (Staff / New Grad)": 1,
    "Officer / Staff": 2,
    "Senior / Specialist": 3,
    "Manager": 4,
    "Director / Executive": 5
}

business_travel_map = {
    "Never travel (0 ครั้ง/เดือน)": "Non-Travel",
    "Travel occasionally (1-2 ครั้ง/เดือน)": "Travel_Rarely",
    "Travel frequently (>3 ครั้ง/เดือน)": "Travel_Frequently"
}

def scale_income(income_baht):
    min_baht, max_baht = 10000, 100000
    min_dataset, max_dataset = 1009, 19999
    scaled = (income_baht - min_baht) * (max_dataset - min_dataset) / (max_baht - min_baht) + min_dataset
    return max(min_dataset, min(scaled, max_dataset))

def predict_attrition(age, overtime, monthly_income, years_at_company, years_in_current_role,
                      job_satisfaction, env_satisfaction, job_level, business_travel):

    sample = pd.DataFrame(0, index=[0], columns=model_features)

    sample['Age'] = age
    sample['OverTime_Yes'] = 1 if overtime == "Yes" else 0

    scaled_income = scale_income(monthly_income)
    sample['MonthlyIncome'] = scaled_income

    sample['YearsAtCompany'] = years_at_company
    sample['YearsInCurrentRole'] = years_in_current_role
    sample['JobSatisfaction'] = satisfaction_map[job_satisfaction]
    sample['EnvironmentSatisfaction'] = satisfaction_map[env_satisfaction]
    sample['JobLevel'] = job_level_map[job_level]
    
    travel_value = business_travel_map[business_travel]
    sample['BusinessTravel_Travel_Frequently'] = 1 if travel_value == "Travel_Frequently" else 0

    sample['StockOptionLevel'] = 0
    sample['YearsSinceLastPromotion'] = 0
    sample['PercentSalaryHike'] = 10
    sample['DailyRate'] = 500
    sample['JobRole_Laboratory Technician'] = 0
    sample['JobRole_Healthcare Representative'] = 0

    pred = attrition_model.predict(sample)[0]
    return "Yes (ลาออก)" if pred == 1 else "No (ไม่ลาออก)"


overtime = ['Yes', 'No']
satisfaction_choices = list(satisfaction_map.keys())
job_level_choices = list(job_level_map.keys())
business_travel_choices = list(business_travel_map.keys())

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        age = gr.Slider(label="Age", minimum=18, maximum=60, step=1)
        monthly_income = gr.Number(
            label="Monthly Income (บาท)",
            minimum=10000,
            maximum=100000,
        )

    with gr.Row():
        years_at_company = gr.Slider(label="Years At Company", minimum=0, maximum=40, step=1)
        years_in_current_role = gr.Slider(label="Years In Current Role", minimum=0, maximum=20, step=1)

    with gr.Row():
        job_satisfaction = gr.Dropdown(satisfaction_choices, label="Job Satisfaction")
        env_satisfaction = gr.Dropdown(satisfaction_choices, label="Environment Satisfaction")

    with gr.Row():
        job_level = gr.Dropdown(job_level_choices, label="Job Level")
        overtime_input = gr.Dropdown(overtime, label="Do you often work overtime?")
        business_travel_input = gr.Dropdown(business_travel_choices, label="Business Travel (ครั้ง/เดือน)")

    predict_btn = gr.Button("Predict Attrition", variant="primary")
    result = gr.Textbox(label="Prediction Result")

    inputs = [age, overtime_input, monthly_income, years_at_company, years_in_current_role,
              job_satisfaction, env_satisfaction, job_level, business_travel_input]

    predict_btn.click(predict_attrition, inputs=inputs, outputs=[result])

if __name__ == "__main__":
    demo.launch(share=True)
