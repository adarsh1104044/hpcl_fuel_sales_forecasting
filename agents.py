from crewai import Agent, Task, Crew, LLM
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

llm = LLM(
    model='gemini/gemini-2.0-pro',
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2
)

data_engineer = Agent(
    role="Excel Data Specialist",
    goal="Transform wide-format HPCL fuel sales Excel data into clean, time-series format for analysis.",
    backstory=(
        "Expert in handling petroleum industry Excel reports. "
        "You ensure all data is accurate, consistent, and analysis-ready."
    ),
    llm=llm,
    verbose=True
)

forecast_analyst = Agent(
    role="Fuel Sales Forecaster",
    goal="Generate accurate forecasts using Prophet and interpret the results for business use.",
    backstory=(
        "Time-series forecasting expert specializing in petroleum product demand. "
        "You use statistical models to predict future sales and explain uncertainty."
    ),
    llm=llm,
    verbose=True
)

business_strategist = Agent(
    role="Fuel Markets Strategist",
    goal="Translate forecasts into actionable business recommendations for HPCL managers.",
    backstory=(
        "A former HPCL regional manager turned AI-powered strategist, you bridge data science and operations, "
        "providing inventory, marketing, and risk management insights."
    ),
    llm=llm,
    verbose=True
)

def create_tasks(outlet, fuel_type, periods):
    from crewai import Task

    return [
        Task(
            description=f"Clean and structure raw Excel data for outlet {outlet} and fuel type {fuel_type}.",
            expected_output="Structured DataFrame with 'Month', 'Sales', and 'Fuel_Type' columns.",
            agent=data_engineer,
            output_file='structured_data.csv'
        ),
        Task(
            description=f"Forecast {fuel_type} sales for outlet {outlet} for {periods} months using Prophet. "
                        "Include confidence intervals and explain the results.",
            expected_output="Forecast DataFrame with predictions, confidence intervals, and summary.",
            agent=forecast_analyst,
            output_file='forecast_results.csv'
        ),
        Task(
            description=f"Based on the forecasted sales for outlet '{outlet}' and fuel type '{fuel_type}', "
                        "generate a business report with:\n"
                        "- Key forecast numbers\n- Inventory recommendations\n"
                        "- Marketing strategy ideas\n- Risk analysis for HPCL managers.",
            expected_output="A business report with actionable recommendations, supporting charts, and an executive summary.",
            agent=business_strategist,
            output_file='business_report.txt'
        )
    ]

