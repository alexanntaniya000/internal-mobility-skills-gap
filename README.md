# internal-mobility-skills-gap
A Python project that uses HR data to find which employees are ready for promotion, which skills the organization is short of, and who could succeed in senior roles.

## Problem
Companies often hire externally for roles that current employees could fill. This project shows how HR can use data to spot internal promotion candidates and plan training.

## Data
- Employee/HR dataset from Kaggle (ravindrasinghrana), 3,000 employee records
- Skill levels are simulated, because the dataset has no skills data. Each employee's skills are based on their job's requirements, performance rating, tenure and some randomness (fixed random seed, so results are repeatable)

## Method
1. Cleaned the data: removed extra spaces, converted dates, created tenure, age and attrition columns
2. Explored attrition rate by department and the performance rating distribution
3. Built a career ladder of 31 job titles across 7 job families, with the skill levels (1-5) each role needs across 22 skills
4. Scored each employee's fit (0-100%) against the next role in their career path
5. Classified promotion readiness, found skill gaps, matched each gap to a training recommendation, and shortlisted the top 3 internal candidates for senior roles

## Key results
- 2,458 active employees analysed
- 404 ready for promotion now, and 669 ready in 6-12 months
- Most common skill gaps: Communication, Quality & Safety and Equipment Handling
- Top 3 succession candidates identified for 16 senior roles

## Charts
![Attrition by department](attrition_by_department.png)
![Promotion readiness](promotion_readiness.png)
![Top skill gaps](top_skill_gaps.png)

## Output files
- promotion_readiness.csv: readiness group and fit score for every active employee
- skill_gaps_and_training.csv: each skill gap with a training recommendation
- org_skill_gaps.csv: how often each skill is a gap across the company
- succession_shortlists.csv: top 3 internal candidates per senior role
- synthetic_skills_matrix.csv: the simulated skills data

## How to run
1. Download employee_data.csv from Kaggle and put it in the same folder as the script
2. Open internal_mobility_skills_gap_analysis.py in Spyder
3. Press F5 to run the whole file. Results are saved in an "outputs" folder

## Tools
Python, pandas, NumPy, matplotlib (Spyder)

## Limitations
- Skill levels are simulated, so the results show the method, not real employee abilities
- The career ladder and readiness thresholds (85% and 65% fit) are my own assumptions
- With real skills data, the same method could be applied to a real company
