# TestLine_Student_Recommendation
Developed a Python-based solution to analyze quiz performance and provide students with personalized recommendations to improve their preparation.

Personalized Student Recommendations

Project Overview
This project analyzes student quiz performance and provides personalized recommendations to improve preparation. The system:
- Processes quiz performance data.
- Identifies weak areas and learning trends.
- Generates insights and improvement suggestions.
- Predicts a student's NEET rank using machine learning.
- Provides performance visualizations.

Setup Instructions

Prerequisites
Ensure you have the following installed:
- Python 3.12.1
- Required libraries: `numpy`, `json`, `collections`, `sklearn`, `matplotlib`, `seaborn`, `scipy`

Installation
1. Clone the repository or download the script.
2. Install dependencies:
   ```
   pip install numpy scikit-learn matplotlib seaborn scipy
   ```
3. Place the dataset files (`Quiz_Endpoint Data.json`, `QuizSubmissionData.json`, `API Endpoint Data.json`) in the project directory.
4. Run the script:
   ```
   python student_recommendations.py
   ```

Approach Description

1. Data Loading
   - Reads three JSON datasets:
     - `Quiz_Endpoint Data.json`: Contains quiz metadata, including topics and difficulty levels.
     - `QuizSubmissionData.json`: Holds details of a user's latest quiz submission, including scores and responses.
     - `API Endpoint Data.json`: Stores historical performance data of students over multiple quizzes.

2. Performance Analysis
   - Extracts quiz scores, accuracy percentages, topics, and mistake counts for each user.
   - Identifies weak topics by analyzing performance trends over multiple quizzes.
   - Uses linear regression to determine trends in accuracy and performance for each topic (Improving vs. Declining).

3. Recommendation Generation
   - If accuracy is below 75%, recommends:
     - Reviewing weak topics.
     - Practicing more medium-to-hard difficulty questions.
     - Analyzing incorrect answers.
   - If accuracy is above 75%, recommends maintaining consistency through timed mock tests.

4. Student Persona Definition
   - Based on overall accuracy and performance trends, categorizes students as:
     - Top Performer: Strong across all topics.
     - Steady Improver: Gradual and consistent progress.
     - Subject Specialist: Excels in some topics, needs improvement in others.
     - Struggler: Needs to focus on fundamentals and weak areas.
     - High Risk: Requires structured revision and immediate intervention.

5. Machine Learning-Based Rank Prediction
   - Uses a `RandomForestRegressor` to predict a student’s potential NEET rank.
   - Model is trained using a hypothetical ranking scale based on quiz scores.
   - Predicts a rank based on the student’s most recent quiz performance.

6. Visualization
   - Generates line plots using Matplotlib and Seaborn to show:
     - Score trends across multiple quizzes.
    ![image](https://github.com/user-attachments/assets/79ea6388-2ad2-45ff-9054-26923b04eba1)
     - User's Accuracy progression
     ![image](https://github.com/user-attachments/assets/c57a3029-9f17-46de-889f-ef7c62a55c0e)
     - Student Persona & Recommendations
     ![image](https://github.com/user-attachments/assets/da289b2c-d14e-4b6b-a090-3ee9756877e6)
     - Probablistic NEET RANK
     ![image](https://github.com/user-attachments/assets/57fa973e-1496-4370-94be-79ad65e5383e)
 
