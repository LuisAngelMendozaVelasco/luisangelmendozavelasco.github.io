# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: tensorflow
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Teaching Ratings

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Data_Science_Fundamentals_with_Python_and_SQL_Specialization/portfolio-1.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Analyze the evaluation of professors with different characteristics and determine whether there are external influences on their teaching evaluation scores.

# %% [markdown]
# ## Import libraries

# %%
from scipy.stats import norm, levene, ttest_ind, f_oneway, chi2_contingency, pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# %% [markdown]
# ## Load the dataset

# %%
file_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/teachingratings.csv'
df = pd.read_csv(file_url).loc[:, :"prof"]
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# | Variable    | Description                                                                                                                                          |
# | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
# | minority    | Does the instructor belong to a minority (non-Caucasian) group?                                                                                      |
# | age         | The instructor's age                                                                                                                                  |
# | gender      | Indicating whether the instructor was male or female.                                                                                                |
# | credits     | Is the course a single-credit elective?                                                                                                              |
# | beauty      | Rating of the instructor's physical appearance by a panel of six students averaged across the six panelists and standardized to have a mean of zero. |
# | eval        | Course overall teaching evaluation score, on a scale of 1 (very unsatisfactory) to 5 (excellent).                                                    |
# | division    | Is the course an upper or lower division course?                                                                                                     |
# | native      | Is the instructor a native English speaker?                                                                                                          |
# | tenure      | Is the instructor on a tenure track?                                                                                                                 |
# | students    | Number of students that participated in the evaluation.                                                                                              |
# | allstudents | Number of students enrolled in the course.                                                                                                           |
# | prof        | Indicating instructor identifier.                                                                                                                    |

# %%
df.info()

# %% [markdown]
# ## Visualize the dataset

# %%
number_features = df.columns.size - 1
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), df.columns[:number_features]):
    if df[feature].dtype == 'O':
        labels, sizes = np.unique(df[feature], return_counts=True)
        sns.barplot(x=labels, y=sizes, hue=labels, ax=ax, legend=False)
        ax.set_xlabel("")
        ax.set_title(feature)
    else:
        sns.histplot(data=df, x=feature, ax=ax)
        ax.set_xlabel("")
        ax.set_title(feature)

for ax in axs.flatten()[number_features:]:
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Analysis

# %% [markdown]
# ### Does the average beauty score vary by gender?

# %%
df_subset = df.drop_duplicates(subset=['prof'])
gender_beauty_mean = df_subset.groupby('gender')["beauty"].mean().round(2)
print(gender_beauty_mean)

plt.figure()
sns.histplot(data=df_subset, x="beauty", hue="gender")
plt.axvline(gender_beauty_mean["female"], color="blue", linestyle='dashed', linewidth=2)
plt.axvline(gender_beauty_mean["male"], color="orange", linestyle='dashed', linewidth=2)
plt.show()

# %% [markdown]
# ### What is the percentage of males and females that are tenured professors?

# %%
gender_tenure_count = df[df.tenure == 'yes'].drop_duplicates(subset=['prof']).groupby('gender')["tenure"].count()
labels = gender_tenure_count.keys()
sizes = gender_tenure_count.values

fig, ax = plt.subplots()
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend(labels)
ax.set_title("Tenured professors")
plt.show()

# %% [markdown]
# ### What is the percentage of professors from minority and non-minority groups that are tenured?

# %%
minority_tenure_count = df[df.tenure == 'yes'].drop_duplicates(subset=['prof']).groupby('minority')["tenure"].count()
labels = ["non-minority" if i == "no" else "minority" for i in minority_tenure_count.keys()]
sizes = minority_tenure_count.values

fig, ax = plt.subplots()
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend(labels)
plt.title("Tenured professors")
plt.show()

# %% [markdown]
# ### Does the average age vary by tenure?

# %%
df_subset = df.drop_duplicates(subset=['prof'])
tenure_age_mean = df_subset.groupby('tenure')["age"].mean().round(2)
print(tenure_age_mean)

plt.figure()
sns.histplot(data=df_subset, x="age", hue="tenure")
plt.axvline(tenure_age_mean["yes"], color="blue", linestyle="dashed", linewidth=2)
plt.axvline(tenure_age_mean["no"], color="orange", linestyle="dashed", linewidth=2)
plt.show()

# %% [markdown]
# ### What is the average evaluation score for tenured professors?

# %%
tenure_eval_mean = df.groupby('tenure')["eval"].mean().round(2)
print(tenure_eval_mean)

plt.figure()
sns.histplot(data=df, x="eval", hue="tenure")
plt.axvline(tenure_eval_mean["yes"], color="blue", linestyle="dashed", linewidth=2)
plt.axvline(tenure_eval_mean["no"], color="orange", linestyle="dashed", linewidth=2)
plt.show()

# %% [markdown]
# ### Do instructors teaching lower-division courses receive higher average teaching evaluations?

# %%
division_eval_mean = df.groupby('division')["eval"].mean().round(2)
print(division_eval_mean)

plt.figure()
sns.histplot(data=df, x="eval", hue="division")
plt.axvline(division_eval_mean["lower"], color="orange", linestyle="dashed", linewidth=2)
plt.axvline(division_eval_mean["upper"], color="blue", linestyle="dashed", linewidth=2)
plt.show()

# %% [markdown]
# ### Do beauty scores vary by course credits?

# %%
print(df.groupby('credits')["beauty"].describe().round(2).iloc[:, 1:])

plt.figure()
sns.boxplot(x="credits", y='beauty', hue="credits", data=df)
plt.show()

# %% [markdown]
# ### What is the number of courses taught by gender?

# %%
courses_gender_count = df["gender"].value_counts()
print(courses_gender_count)

plt.figure()
sns.countplot(data=df, x="gender", hue="gender")
plt.show()

# %% [markdown]
# ### What is the number of professors by gender and tenure?

# %%
df_subset = df.drop_duplicates(subset=['prof'])
gender_tenure_count = df_subset.groupby("gender")["tenure"].value_counts()
print(gender_tenure_count)

plt.figure()
sns.countplot(data=df_subset, x="gender", hue="tenure")
plt.show()

# %% [markdown]
# ### Does the average evaluation score vary by gender?

# %%
gender_eval_mean = df.groupby('gender')["eval"].mean().round(2)
print(gender_eval_mean)

plt.figure()
sns.histplot(data=df, x="eval", hue="gender")
plt.axvline(gender_eval_mean["female"], color="blue", linestyle='dashed', linewidth=2)
plt.axvline(gender_eval_mean["male"], color="orange", linestyle='dashed', linewidth=2)
plt.show()

# %% [markdown]
# ### What is the average age by gender?

# %%
df_subset = df.drop_duplicates(subset=['prof'])
gender_eval_mean = df_subset.groupby('gender')["age"].mean().round(2)
print(gender_eval_mean)

plt.figure()
sns.histplot(data=df_subset, x="age", hue="gender")
plt.axvline(gender_eval_mean["female"], color="blue", linestyle='dashed', linewidth=2)
plt.axvline(gender_eval_mean["male"], color="orange", linestyle='dashed', linewidth=2)
plt.show()

# %% [markdown]
# ### What is the average age by gender and tenure?

# %%
df_subset = df.drop_duplicates(subset=['prof'])
print(df_subset.groupby(['tenure', "gender"])["age"].describe().round(2).iloc[:, 1:])

plt.figure()
sns.boxplot(x="tenure", y="age", hue="gender", data=df_subset)
plt.show()

# %% [markdown]
# ### Do native professors have higher beauty scores?

# %%
df_subset = df.drop_duplicates(subset=['prof'])
native_beauty_mean = df_subset.groupby('native')["beauty"].mean().round(2)
print(native_beauty_mean)

plt.figure()
sns.histplot(data=df_subset, x="beauty", hue="native")
plt.axvline(native_beauty_mean["yes"], color="blue", linestyle="dashed", linewidth=2)
plt.axvline(native_beauty_mean["no"], color="orange", linestyle="dashed", linewidth=2)
plt.show()

# %% [markdown]
# ### What is the average age of professors from minority and non-minority groups?

# %%
df_subset = df.drop_duplicates(subset=['prof'])
print(df_subset.groupby(["minority"])["age"].describe().round(2).iloc[:, 1:])

plt.figure()
sns.boxplot(x="minority", y="age", hue="minority", data=df_subset)
plt.show()

# %% [markdown]
# ### What is the number of professors by gender from minority and non-minority groups?

# %%
df_subset = df.drop_duplicates(subset=['prof'])
gender_minority_count = df_subset.groupby("gender")["minority"].value_counts()
print(gender_minority_count)

plt.figure()
sns.countplot(data=df_subset, x="gender", hue="minority")
plt.show()

# %% [markdown]
# ### What is the probability of receiving an evaluation score greater than 4.5?

# %%
eval_statistics = df["eval"].describe().round(2).iloc[1:]
probability = norm.cdf((4.5 - eval_statistics["mean"]) / eval_statistics["std"])
print(eval_statistics)
print("\n\033[1mProbability\033[0m = {:.2f}%".format(100*(1 - probability)))

plt.figure()
sns.boxplot(x='eval', data=df)
plt.show()

# %% [markdown]
# ### What is the probability of receiving an evaluation score greater than 3.5 and less than 4.2?

# %%
probability_1 = norm.cdf((3.5 - eval_statistics["mean"]) / eval_statistics["std"])
probability_2 = norm.cdf((4.2 - eval_statistics["mean"]) / eval_statistics["std"])
print("\033[1mProbability\033[0m = {:.2f}%".format(100*(probability_2 - probability_1)))

# %% [markdown]
# ### Using t-test, does gender affect teaching evaluation scores?

# %% [markdown]
# For the **t-test** for independent samples, the following assumptions must be met:
#
# * One independent, categorical variable with two levels or groups.
# * One dependent continuous variable.
# * There is no relationship between the observations in each group.
# * The dependent variable must follow a normal distribution.
# * Assumption of homogeneity of variance.
#
# State the hypothesis:
#
# * $H_0: µ_1 = µ_2$ (There is no difference in evaluation scores between male and females)
# * $H_1: µ_1 ≠ µ_2$ (There is a difference in evaluation scores between male and females)

# %%
df.groupby("gender")["eval"].describe()

# %% [markdown]
# Use the Levene's test to test the null hypothesis that all input samples are from populations with equal variances.

# %%
levene(df[df['gender'] == 'female']['eval'], 
       df[df['gender'] == 'male']['eval'], 
       center='mean')

# %% [markdown]
# Since the p-value is greater than 0.05, we can assume equality of variance.

# %%
ttest_ind(df[df['gender'] == 'female']['eval'], 
          df[df['gender'] == 'male']['eval'], 
          equal_var=True)

# %% [markdown]
# **Answer**: Since the p-value is less than 0.05, we can reject the null hypothesis as there is enough proof that there is a statistical difference in teaching evaluations based on gender.

# %% [markdown]
# ### Using ANOVA test, do beauty scores vary by age?

# %% [markdown]
# The data must be grouped into categories as the one-way ANOVA can't work with continuous variable, then the categories will be professors that are:
#
# * 40 years and younger.
# * Between 40 and 60 years.
# * 60 years and older.
#
# State the hypothesis:
#
# * $H_0: µ_1 = µ_2 = µ_3$ (The three population means are equal)
# * $H_1:$ At least one of the means differ.

# %%
df.loc[(df['age'] <= 40), 'age_group'] = '40 years and younger'
df.loc[(df['age'] > 40) & (df['age'] < 60), 'age_group'] = 'between 40 and 60 years'
df.loc[(df['age'] >= 60), 'age_group'] = '60 years and older'

# %%
df.groupby("age_group")["beauty"].describe()

# %% [markdown]
# Test for equality of variance.

# %%
levene(df[df['age_group'] == '40 years and younger']['beauty'],
       df[df['age_group'] == 'between 40 and 60 years']['beauty'],
       df[df['age_group'] == '60 years and older']['beauty'], 
       center='mean')

# %% [markdown]
# Since the p-value is less than 0.05, the variances are not equal.

# %%
f_oneway(df[df['age_group'] == '40 years and younger']['beauty'], 
         df[df['age_group'] == 'between 40 and 60 years']['beauty'], 
         df[df['age_group'] == '60 years and older']['beauty'])

# %% [markdown]
# **Answer**: Since the p-value is less than 0.05, we reject the null hypothesis as there is significant evidence that at least one of the means differs.

# %% [markdown]
# ### Using ANOVA test, do teaching evaluation scores vary by age?

# %% [markdown]
# State the hypothesis:
#
# * $H_0: µ_1 = µ_2 = µ_3$ (The three population means are equal)
# * $H_1:$ At least one of the means differ.

# %%
df.groupby("age_group")["eval"].describe()

# %%
levene(df[df['age_group'] == '40 years and younger']['eval'],
       df[df['age_group'] == 'between 40 and 60 years']['eval'],
       df[df['age_group'] == '60 years and older']['eval'], 
       center='mean')

# %%
f_oneway(df[df['age_group'] == '40 years and younger']['eval'], 
         df[df['age_group'] == 'between 40 and 60 years']['eval'], 
         df[df['age_group'] == '60 years and older']['eval'])

# %% [markdown]
# **Answer**: Since the p-value is greater than 0.05, we cannot reject the null hypothesis as there is no significant evidence that at least one of the means differs.

# %% [markdown]
# ### Using chi-square, is there an association between tenure and gender?

# %% [markdown]
# State the hypothesis:
#
# * $H_0:$ The proportion of professors who are tenured is independent of gender.
# * $H_1:$ The proportion of professors who are tenured is associated with gender.

# %%
cross_tenure_gender = pd.crosstab(df['tenure'], df['gender'])
cross_tenure_gender

# %%
chi2_contingency(cross_tenure_gender, correction=False)

# %% [markdown]
# **Answer**: Since the p-value is greater than 0.05, we cannot reject the null hypothesis as there is no sufficient evidence that professors are tenured as a result of gender.

# %% [markdown]
# ### Using Pearson correlation, is teaching evaluation score correlated with beauty score?

# %% [markdown]
# State the hypothesis:
#
# * $H_0:$ Teaching evaluation score is not correlated with beauty score.
# * $H_1:$ Teaching evaluation score is correlated with beauty score.

# %%
sns.lmplot(data=df, x="beauty", y="eval", line_kws={"color": "red"})
plt.show()

# %%
pearsonr(df['beauty'], df['eval'])

# %% [markdown]
# **Answer**: Since the p-value is less than 0.05, we reject the null hypothesis as there exists a relationship between beauty and teaching evaluation score.

# %% [markdown]
# ### Using t-test, does tenure affect teaching evaluation scores?

# %% [markdown]
# State the hypothesis
#
# * $H_0: µ_1 = µ_2$ (There is no difference in evaluation scores between tenure and non-tenure)
# * $H_1: µ_1 ≠ µ_2$ (There is a difference in evaluation scores between tenure and non-tenure)

# %%
df.groupby("tenure")["eval"].describe()

# %%
ttest_ind(df[df['tenure'] == 'yes']['eval'],
          df[df['tenure'] == 'no']['eval'],
          equal_var=True)

# %% [markdown]
# **Answer**: Since the p-value is less than 0.05, we reject the null hypothesis as there evidence that being tenured affects teaching evaluation scores.

# %% [markdown]
# ### Using chi-square, is there an association between age and tenure?

# %% [markdown]
# State the hypothesis:
#
# * $H_0:$ There is no association between age and tenure.
# * $H_1:$ There is an association between age and tenure.

# %%
cross_tenure_age = pd.crosstab(df['tenure'], df['age_group'])
cross_tenure_age

# %%
chi2_contingency(cross_tenure_age, correction=True)

# %% [markdown]
# **Answer**: Since the p-value is less than 0.05, we reject the null hypothesis as there is evidence of an association between age and tenure.

# %% [markdown]
# ### Using chi-square, is there an association between visible minorities and tenure?

# %% [markdown]
# State the hypothesis:
#
# * $H_0:$ There is no association between tenure and visible minorities.
# * $H_1:$ There is an association between tenure and visible minorities.

# %%
cross_minority_tenure = pd.crosstab(df['minority'], df['tenure'])
cross_minority_tenure

# %%
chi2_contingency(cross_minority_tenure, correction=True)

# %% [markdown]
# **Answer**: Since the p-value is greater than 0.05, we cannot reject the null hypothesis as there is no evidence of an association between visible minorities and tenure.

# %% [markdown]
# ### Using regression with t-test, does gender affect teaching evaluation scores?

# %% [markdown]
# State the hypothesis:
#
# * $H_0: β_1$ = 0 (Gender has no effect on teaching evaluation scores)
# * $H_1: β_1$ is not equal to 0 (Gender has an effect on teaching evaluation scores)

# %%
# X is the input variables (or independent variables)
X = df['gender'].map({"female": 1, "male": 0})

# y is the target/dependent variable
y = df['eval']

# Add an intercept (beta_0) to our model
X = sm.add_constant(X) 

# Ordinary Least Squares
model = sm.OLS(y, X).fit() 

# Print out the statistics
model.summary()

# %% [markdown]
# **Answer**: Since the p-value is less than 0.05, we reject the null hypothesis as there is evidence that there is a difference in mean evaluation scores based on gender. The coefficient -0.1680 means that females get 0.168 scores less than men.

# %% [markdown]
# ### Using regression with ANOVA, do beauty scores vary by age?

# %% [markdown]
# State the hypothesis:
#
# * $H_0: µ_1 = µ_2 = µ_3$ (The three population means are equal)
# * $H_1:$ At least one of the means differ.

# %%
model = ols('beauty ~ age_group', data=df).fit()
anova_table = sm.stats.anova_lm(model)
anova_table

# %% [markdown]
# **Answer**: Since the p-value is less than 0.05, we reject the null hypothesis as there is significant evidence that at least one of the means differs.

# %% [markdown]
# ### Using regression with t-test, does tenure affect beauty scores?

# %% [markdown]
# State the hypothesis:
#
# * $H_0:$ The average beauty scores for tenured and non-tenured instructors are equal.
# * $H_1:$ There is a difference in the average beauty scores for tenured and non-tenured instructors.

# %%
# X is the input variables (or independent variables)
X = df['tenure'].map({"yes": 1, "no": 0})

# y is the target/dependent variable
y = df['beauty']

# Add an intercept (beta_0) to our model
X = sm.add_constant(X) 

# Ordinary Least Squares
model = sm.OLS(y, X).fit() 

# Print out the statistics
model.summary()

# %% [markdown]
# **Answer**: Since the p-value is greater than 0.05, we cannot reject the null hypothesis as there is no evidence that the mean difference of tenured and untenured instructors are different.

# %% [markdown]
# ### Using regression with t-test, are more students assigned to native professors?

# %% [markdown]
# State the hypothesis:
#
# * $H_0:$ The average number of students assigned to native english speakers vs non-native english speakers are equal.
# * $H_1:$ There is a difference in the average number of students assigned to native english speakers vs non-native English speakers.

# %%
# X is the input variables (or independent variables)
X = df["native"].map({"yes": 1, "no": 0})

# y is the target/dependent variable
y = df['allstudents']

# Add an intercept (beta_0) to our model
X = sm.add_constant(X) 

# Ordinary Least Squares
model = sm.OLS(y, X).fit() 

# Print out the statistics
model.summary()

# %% [markdown]
# **Answer**: Since the p-value is greater than 0.05, we cannot reject the null hypothesis as there is no evidence that being a native english speaker or a non-native english speaker affects the number of students assigned.
