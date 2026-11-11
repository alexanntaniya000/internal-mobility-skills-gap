#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:05:40 2026

@author: anntaniyaalex
"""

# %% step 1: import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)

# %% step 2: load the data
csv_path = "/Users/AnnTaniyaAlex/Downloads/employee_data.csv"

# results will be saved in an "outputs" folder next to the csv
output_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "outputs")
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
df.info()
print("Rows, columns:", df.shape)
print(df.head())

# %% step 3: first look at the data
print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Missing values per column ---")
print(df.isna().sum()[df.isna().sum() > 0])
# ExitDate is blank for people who never left, so that is normal

print("\n--- Duplicate employee IDs ---")
print(df["EmpID"].duplicated().sum())

print("\n--- Employee status counts ---")
print(df["EmployeeStatus"].value_counts())

# %% step 4: clean the data
# remove extra spaces in text columns
df["Title"] = df["Title"].str.strip()
df["DepartmentType"] = df["DepartmentType"].str.strip()
df["EmployeeStatus"] = df["EmployeeStatus"].str.strip()
df["Performance Score"] = df["Performance Score"].str.strip()

# change text into real dates
df["StartDate"] = pd.to_datetime(df["StartDate"], format="%d-%b-%y")
df["ExitDate"] = pd.to_datetime(df["ExitDate"], format="%d-%b-%y")
df["DOB"] = pd.to_datetime(df["DOB"], format="%d-%m-%Y")

# "today" for this data = the latest start date in the file
AS_OF = df["StartDate"].max()

# new columns to make the analysis easier
df["TenureYears"] = (df["ExitDate"].fillna(AS_OF) - df["StartDate"]).dt.days / 365
df["Age"] = (AS_OF - df["DOB"]).dt.days // 365
df["FullName"] = df["FirstName"] + " " + df["LastName"]

# attrition flag: 1 = left the company, 0 = did not leave
df["Left"] = 0
df.loc[df["EmployeeStatus"].str.contains("Terminated"), "Left"] = 1

# rename columns
df = df.rename(columns={"Performance Score": "PerformanceScore",
                        "Current Employee Rating": "Rating",
                        "DepartmentType": "Department"})

# drop columns we do not need
df = df.drop(columns=["ADEmail", "TerminationDescription", "DOB",
                      "FirstName", "LastName", "LocationCode"])

print(df[["EmpID", "FullName", "Title", "Department",
          "TenureYears", "Age", "Rating", "Left"]].head())

# %% step 5: exploratory analysis
# leave out future employees, they cannot have left yet
base = df[df["EmployeeStatus"] != "Future Start"]

# attrition rate by department
attr_by_dept = (base.groupby("Department")["Left"].mean() * 100).sort_values()
print("\nAttrition rate (%) by department:")
print(attr_by_dept.round(1))

# performance distribution
print("\nPerformance score counts:")
print(df["PerformanceScore"].value_counts())

# chart
plt.figure(figsize=(8, 4))
attr_by_dept.plot(kind="barh", color="#4C72B0")
plt.xlabel("Attrition rate (%)")
plt.title("Attrition rate by department")
plt.tight_layout()
plt.show()

# %% step 6: career map (job family and level for every title)
# level 1 = entry, level 5 = director / executive
TITLE_MAP = {
    # title                         : (family,           level)
    "Production Technician I":        ("Production",      1),
    "Production Technician II":       ("Production",      2),
    "Production Manager":             ("Production",      4),
    "Director of Operations":         ("Production",      5),

    "Area Sales Manager":             ("Sales",           3),
    "Sales Manager":                  ("Sales",           4),
    "Director of Sales":              ("Sales",           5),

    "IT Support":                     ("IT Infrastructure", 1),
    "Network Engineer":               ("IT Infrastructure", 2),
    "Sr. Network Engineer":           ("IT Infrastructure", 3),
    "IT Manager - Infra":             ("IT Infrastructure", 4),
    "IT Manager - Support":           ("IT Infrastructure", 4),
    "IT Director":                    ("IT Infrastructure", 5),

    "Data Analyst":                   ("Data & BI",       1),
    "BI Developer":                   ("Data & BI",       2),
    "Database Administrator":         ("Data & BI",       2),
    "Senior BI Developer":            ("Data & BI",       3),
    "Sr. DBA":                        ("Data & BI",       3),
    "Data Architect":                 ("Data & BI",       4),
    "IT Manager - DB":                ("Data & BI",       4),
    "Principal Data Architect":       ("Data & BI",       5),
    "BI Director":                    ("Data & BI",       5),
    "Enterprise Architect":           ("Data & BI",       5),

    "Software Engineer":              ("Software",        2),
    "Software Engineering Manager":   ("Software",        4),

    "Administrative Assistant":       ("Finance & Admin", 1),
    "Accountant I":                   ("Finance & Admin", 1),
    "Sr. Accountant":                 ("Finance & Admin", 2),
    "Shared Services Manager":        ("Finance & Admin", 4),

    "CIO":                            ("Executive",       5),
    "President & CEO":                ("Executive",       5),
}

# check that every title in the data is in the map
missing_titles = set(df["Title"]) - set(TITLE_MAP)
print("Titles missing from TITLE_MAP:", missing_titles or "none")

# add job family and job level to every employee
df["JobFamily"] = ""
df["JobLevel"] = 0
for title in TITLE_MAP:
    df.loc[df["Title"] == title, "JobFamily"] = TITLE_MAP[title][0]
    df.loc[df["Title"] == title, "JobLevel"] = TITLE_MAP[title][1]

print(df[["Title", "JobFamily", "JobLevel"]].head())

# %% step 7: skills for each job family
CORE_SKILLS = {
    "Production":        ["Process Operations", "Quality & Safety", "Equipment Handling"],
    "Sales":             ["Sales & Negotiation", "Client Relations", "Market Knowledge"],
    "IT Infrastructure": ["Networking", "Systems Support", "Cybersecurity", "Cloud Infrastructure"],
    "Data & BI":         ["Data Analysis", "SQL & Databases", "Data Visualization", "Programming"],
    "Software":          ["Programming", "Software Design", "SQL & Databases"],
    "Finance & Admin":   ["Accounting", "Compliance", "Office Systems"],
    "Executive":         ["Strategic Thinking", "Leadership", "Compliance"],
}

# one list with every skill (no repeats)
ALL_SKILLS = []
for family in CORE_SKILLS:
    for skill in CORE_SKILLS[family]:
        if skill not in ALL_SKILLS:
            ALL_SKILLS.append(skill)

# skills every family needs as people get more senior
for skill in ["Communication", "Leadership", "Project Management", "Strategic Thinking"]:
    if skill not in ALL_SKILLS:
        ALL_SKILLS.append(skill)

ALL_SKILLS.sort()
print("Total skills in the model:", len(ALL_SKILLS))

# %% step 8: skill level (1-5) that each job needs
# start with a table of zeros (0 = skill not needed)
roles = pd.DataFrame(0, index=list(TITLE_MAP), columns=ALL_SKILLS)
roles.index.name = "Title"

# go through every job title and fill in what it needs
# (everything below must stay indented inside this loop)
for title in TITLE_MAP:
    family = TITLE_MAP[title][0]
    level = TITLE_MAP[title][1]

    for skill in CORE_SKILLS[family]:
        roles.loc[title, skill] = min(level, 4)

    roles.loc[title, "Communication"] = max(2, level)

    if level >= 3:
        roles.loc[title, "Leadership"] = level - 1
        roles.loc[title, "Project Management"] = level - 1

    if level >= 4:
        roles.loc[title, "Strategic Thinking"] = level - 1
print(roles.head())
print("Check - total of the table (should not be 0):", roles.sum().sum())

# %% step 9: skills matrix for employees (simulated)
# there is no skills data in the file, so we create it.
# every skill starts at what the person's current job needs, then we adjust it:
#   better rating -> higher, longer tenure -> slightly higher, plus some randomness
rng = np.random.default_rng(42)      # fixed seed = same result every run

# only active employees can be promoted or moved
active = df[df["EmployeeStatus"] == "Active"].copy().reset_index(drop=True)
print("\nActive employees in the talent pool:", len(active))

skill_rows = []
for i in range(len(active)):
    title = active.loc[i, "Title"]
    rating = active.loc[i, "Rating"]
    tenure = active.loc[i, "TenureYears"]

    perf_effect = (rating - 3) * 0.5                     # rating 5 = +1, rating 1 = -1
    tenure_effect = min(max(tenure - 2, 0), 3) * 0.15    # small bonus after 2 years

    row = [active.loc[i, "EmpID"]]
    for skill in ALL_SKILLS:
        needed = roles.loc[title, skill]
        noise = rng.normal(0, 0.6)                       # people are all different

        if needed > 0:
            level = needed + perf_effect + tenure_effect + noise
        else:
            level = rng.uniform(1, 2) + 0.3 * perf_effect + 0.5 * noise

        level = int(round(level))
        level = min(max(level, 1), 5)                    # keep between 1 and 5
        row.append(level)

    skill_rows.append(row)

skills = pd.DataFrame(skill_rows, columns=["EmpID"] + ALL_SKILLS)
print("\nSynthetic skills matrix (first 5 employees):")
print(skills.head())

# %% step 10: fit score - how well does an employee match a role?
# for each skill the role needs, the employee earns credit equal to their level,
# but never more than the level needed.
# fit score = credit earned / credit possible x 100

skills_idx = skills.set_index("EmpID")      # lets us look up skills by employee id

# test it on one employee and one role first
test_id = active.loc[0, "EmpID"]
test_role = "Production Manager"

have = skills_idx.loc[test_id]       # the employee's skills
need = roles.loc[test_role]          # skills the role needs

total_need = 0                       # credit possible
total_have = 0                       # credit earned

for skill in ALL_SKILLS:
    if need[skill] > 0:                               # only skills the role needs
        credit = min(have[skill], need[skill])        # never more than needed
        total_need = total_need + need[skill]
        total_have = total_have + credit
        print(skill, "| needs", need[skill], "| has", have[skill], "| credit", credit)

if total_need > 0:
    fit = total_have / total_need * 100
    print("\nFit score for employee", test_id, "->", test_role, ":", round(fit, 1), "%")
else:
    print("total_need is 0, so the roles table is empty. Re-run steps 6, 7 and 8.")

# %% step 11: find each employee's next promotion role
# next role = a job in the same family, one level higher

# 11a. for every job title, list its possible next roles
next_roles_of = {}
for title in TITLE_MAP:
    family = TITLE_MAP[title][0]
    level = TITLE_MAP[title][1]

    # levels above this job in the same family
    higher_levels = []
    for other in TITLE_MAP:
        if TITLE_MAP[other][0] == family and TITLE_MAP[other][1] > level:
            higher_levels.append(TITLE_MAP[other][1])

    next_roles_of[title] = []           # empty list = already at the top
    if len(higher_levels) > 0:
        next_level = min(higher_levels)  # the closest level above
        for other in TITLE_MAP:
            if TITLE_MAP[other][0] == family and TITLE_MAP[other][1] == next_level:
                next_roles_of[title].append(other)

print(next_roles_of["Production Technician II"])

# 11b. for every active employee, find the next role they fit best
results = []
for i in range(len(active)):
    emp_id = active.loc[i, "EmpID"]
    have = skills_idx.loc[emp_id]
    candidates = next_roles_of[active.loc[i, "Title"]]

    best_role = None
    best_score = None
    for role in candidates:
        need = roles.loc[role]
        total_need = 0
        total_have = 0
        for skill in ALL_SKILLS:
            if need[skill] > 0:
                total_need = total_need + need[skill]
                total_have = total_have + min(have[skill], need[skill])
        score = total_have / total_need * 100

        if best_score is None or score > best_score:     # keep the best one
            best_score = score
            best_role = role

    if best_score is not None:
        best_score = round(best_score, 1)
    results.append([emp_id, best_role, best_score])

promotion = pd.DataFrame(results, columns=["EmpID", "TargetRole", "FitScore"])

# join with employee details into one table
readiness = active[["EmpID", "FullName", "Title", "Department", "JobFamily",
                    "JobLevel", "TenureYears", "PerformanceScore", "Rating"]]
readiness = readiness.merge(promotion, on="EmpID")
print(readiness.head())

# %% step 12: promotion readiness
# everyone starts in the lowest group, then better groups overwrite it.
# the last rule that applies wins.
readiness["Readiness"] = "Needs development"
readiness.loc[readiness["FitScore"] >= 65, "Readiness"] = "Ready in 6-12 months"
readiness.loc[readiness["FitScore"] >= 85, "Readiness"] = "Ready now"

# low performers are not eligible, whatever their fit score is
low_performers = readiness["PerformanceScore"].isin(["PIP", "Needs Improvement"])
readiness.loc[low_performers, "Readiness"] = "Not eligible (performance)"

# no target role = already at the top of the ladder
readiness.loc[readiness["TargetRole"].isna(), "Readiness"] = "Top of ladder"

print(readiness["Readiness"].value_counts())

readiness["Readiness"].value_counts().plot(kind="bar", color="#55A868", figsize=(8, 4))
plt.title("Promotion readiness of active employees")
plt.ylabel("Employees")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# %% step 13: skill gaps
# compare each employee's skills with their target role.
# if the role needs more than they have, save it as a gap.
gap_list = []
for i in range(len(readiness)):
    emp_id = readiness.loc[i, "EmpID"]
    target = readiness.loc[i, "TargetRole"]

    if pd.isna(target):                  # no target role, skip this person
        continue

    need = roles.loc[target]
    have = skills_idx.loc[emp_id]

    for skill in ALL_SKILLS:
        if need[skill] > have[skill]:
            gap_list.append([emp_id, target, skill, need[skill], have[skill],
                             need[skill] - have[skill]])

gaps = pd.DataFrame(gap_list, columns=["EmpID", "TargetRole", "Skill",
                                       "Required", "Current", "Gap"])
print("Total skill gaps found:", len(gaps))

# which skills are missing most often across the company?
org_gaps = gaps["Skill"].value_counts()
print(org_gaps.head(10))

org_gaps.head(10).sort_values().plot(kind="barh", color="#C44E52", figsize=(8, 5))
plt.xlabel("Employees below the level needed for their next role")
plt.title("Top 10 skill gaps")
plt.tight_layout()
plt.show()

# %% step 14: training recommendations
# simple lookup: skill -> suggested training
TRAINING = {
    "Process Operations": "Lean / process improvement workshop",
    "Quality & Safety": "Quality management and workplace safety certification",
    "Equipment Handling": "Advanced equipment operation training",
    "Sales & Negotiation": "Consultative selling and negotiation course",
    "Client Relations": "Key account management programme",
    "Market Knowledge": "Market analysis and competitor research module",
    "Networking": "Network engineering certification (e.g. CCNA)",
    "Systems Support": "ITIL service management foundation",
    "Cybersecurity": "Security fundamentals (e.g. CompTIA Security+)",
    "Cloud Infrastructure": "Cloud practitioner certification (AWS / Azure)",
    "Data Analysis": "Applied data analysis with Python or Excel",
    "SQL & Databases": "Advanced SQL and database design course",
    "Data Visualization": "Power BI / Tableau dashboard training",
    "Programming": "Software development bootcamp module",
    "Software Design": "Software architecture and design patterns",
    "Accounting": "Advanced accounting / IFRS course",
    "Compliance": "Regulatory compliance and audit training",
    "Office Systems": "Advanced office and ERP tools workshop",
    "Communication": "Business communication and presentation skills",
    "Leadership": "First-time manager / leadership development programme",
    "Project Management": "Project management fundamentals (PMP / PRINCE2)",
    "Strategic Thinking": "Strategy and business acumen executive programme",
}

gaps["RecommendedTraining"] = gaps["Skill"].map(TRAINING)
print(gaps.head(10))

# %% step 15: succession planning
# for every senior role (level 4 and 5), score the employees below that level
# who are rated 3 or higher, then keep the best 3.
shortlist_list = []
for role in TITLE_MAP:
    role_level = TITLE_MAP[role][1]
    if role_level < 4:                   # only senior roles
        continue

    need = roles.loc[role]
    rows = []
    for i in range(len(readiness)):
        if readiness.loc[i, "JobLevel"] < role_level and readiness.loc[i, "Rating"] >= 3:
            emp_id = readiness.loc[i, "EmpID"]
            have = skills_idx.loc[emp_id]

            total_need = 0
            total_have = 0
            for skill in ALL_SKILLS:
                if need[skill] > 0:
                    total_need = total_need + need[skill]
                    total_have = total_have + min(have[skill], need[skill])
            score = total_have / total_need * 100

            rows.append([emp_id, readiness.loc[i, "FullName"],
                         readiness.loc[i, "Title"], readiness.loc[i, "Department"],
                         round(score, 1)])

    result = pd.DataFrame(rows, columns=["EmpID", "FullName", "Title",
                                         "Department", "FitScore"])
    result = result.sort_values("FitScore", ascending=False).head(3)
    result.insert(0, "Role", role)       # add a column with the role name
    shortlist_list.append(result)

succession_all = pd.concat(shortlist_list)

# look at the shortlist for one role (change the name to see another)
print(succession_all[succession_all["Role"] == "IT Director"])
print(succession_all[succession_all["Role"] == "Production Manager"])

# %% step 16: employee report (change the id to look at anyone)
report_id = readiness.loc[0, "EmpID"]

row = readiness[readiness["EmpID"] == report_id].iloc[0]
print("Name        :", row["FullName"])
print("Current job :", row["Title"], "|", row["Department"])
print("Rating      :", row["Rating"])
print("Target role :", row["TargetRole"])
print("Fit score   :", row["FitScore"])
print("Readiness   :", row["Readiness"])
print(gaps[gaps["EmpID"] == report_id][["Skill", "Required", "Current",
                                         "RecommendedTraining"]])

# %% step 17: save results
readiness.to_csv(os.path.join(output_dir, "promotion_readiness.csv"), index=False)
gaps.to_csv(os.path.join(output_dir, "skill_gaps_and_training.csv"), index=False)
org_gaps.to_csv(os.path.join(output_dir, "org_skill_gaps.csv"))
skills.to_csv(os.path.join(output_dir, "synthetic_skills_matrix.csv"), index=False)
succession_all.to_csv(os.path.join(output_dir, "succession_shortlists.csv"), index=False)
print("Files saved in:", output_dir)

# %% step 18: summary numbers for the CV / report
print("Active employees analysed :", len(readiness))
print("Job titles modelled       :", len(roles))
print("Skills in the model       :", len(ALL_SKILLS))
print("Ready now                 :", (readiness["Readiness"] == "Ready now").sum())
print("Ready in 6-12 months      :", (readiness["Readiness"] == "Ready in 6-12 months").sum())
print("Most common skill gap     :", org_gaps.index[0])
