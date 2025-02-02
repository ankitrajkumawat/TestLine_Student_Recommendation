#importing libraries
import json
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore")

# Load Data
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
# Loading Dataset
quiz_endpoint_data = load_json(r"Quiz_Endpoint Data.json")
quiz_submission_data = load_json(r"QuizSubmissionData.json")
historical_data = load_json(r"API Endpoint Data.json")

# Extract relevant fields
def extract_user_performance(historical_data, user_id):
    # User-wise Record Entry
    user_records = [entry for entry in historical_data if entry['user_id'] == user_id]
    performance = {
        "scores": [record['score'] for record in user_records],
        "accuracy": [float(record['accuracy'].replace('%', '')) for record in user_records],
        "topics": [record['quiz']['topic'] for record in user_records],
        "mistakes": [record['initial_mistake_count'] for record in user_records],
    }
    return performance

# Identify weak topics
def identify_weak_topics(performance):
    # Identify weaker topics
    topic_counts = Counter(performance['topics'])
    weak_topics = [topic for topic, count in topic_counts.items() if count > 1]
    return weak_topics

# Explore schema and identify performance patterns
def analyze_student_performance(historical_data, user_id):
    user_records = [entry for entry in historical_data if entry['user_id'] == user_id]
    topic_performance = {}

    for record in user_records:
        topic = record['quiz']['topic']
        accuracy = float(record['accuracy'].replace('%', ''))
        difficulty = record['quiz'].get('difficulty_level', 'Unknown')

        if topic not in topic_performance:
            topic_performance[topic] = {'accuracy': [], 'difficulty': difficulty, 'trend': []}
        topic_performance[topic]['accuracy'].append(accuracy)

    # Calculate average accuracy and trend per topic
    for topic in topic_performance:
        topic_performance[topic]['avg_accuracy'] = np.mean(topic_performance[topic]['accuracy'])
        trend_slope, _, _, _, _ = linregress(range(len(topic_performance[topic]['accuracy'])), topic_performance[topic]['accuracy'])
        topic_performance[topic]['trend'] = "Improving" if trend_slope > 0 else "Declining"

    return topic_performance

# Define student persona
def define_student_persona(performance, topic_analysis):
    avg_accuracy = np.mean(performance['accuracy'])
    consistent_performer = all(data['trend'] == "Improving" for data in topic_analysis.values())

    if avg_accuracy > 90:
        persona = "Top Performer - Strong across all topics"
    elif avg_accuracy > 75 and consistent_performer:
        persona = "Steady Improver - Gradual and consistent progress"
    elif avg_accuracy > 75:
        persona = "Subject Specialist - Excels in some topics, needs improvement in others"
    elif avg_accuracy > 50:
        persona = "Struggler - Needs to focus on fundamentals and weak areas"
    else:
        persona = "High Risk - Requires immediate intervention and structured revision plan"

    return persona

# Generate personalized recommendations
def generate_recommendations(performance, topic_analysis):
    avg_accuracy = np.mean(performance['accuracy'])
    weak_topics = identify_weak_topics(performance)
    recommendations = []

    for topic, data in topic_analysis.items():
        if data['avg_accuracy'] < 75:
            recommendations.append(f"Improve on {topic} (Avg Accuracy: {data['avg_accuracy']:.2f}%, Trend: {data['trend']})")

    if avg_accuracy < 75:
        recommendations.append("Focus on revising weak topics: {}".format(", ".join(weak_topics)))
        recommendations.append("Practice more medium-to-hard difficulty questions.")
        recommendations.append("Review detailed solutions for incorrect answers.")
    else:
        recommendations.append("Maintain consistency and attempt time-based mock tests.")

    return recommendations

# NEET Rank Prediction Model
def predict_neet_rank(performance):
    scores = np.array(performance['scores']).reshape(-1, 1)
    # Generate ranks based on the length of scores to match the number of samples
    ranks = np.linspace(1000, 800, len(scores))  # Hypothetical NEET ranks based on scores
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(scores, ranks)
    predicted_rank = model.predict([[performance['scores'][-1]]])[0]
    return int(predicted_rank)

# Visualization
def plot_performance_trend(performance):
    plt.figure(figsize=(8,5))
    sns.lineplot(x=range(len(performance['scores'])), y=performance['scores'], marker='o', label='Scores')
    sns.lineplot(x=range(len(performance['accuracy'])), y=performance['accuracy'], marker='s', label='Accuracy')
    plt.xlabel("Quiz Attempts")
    plt.ylabel("Scores/Accuracy")
    plt.title("Performance Trend Over Time")
    plt.legend()
    plt.show()

# Process multiple students (Example: 5 students)
unique_users = list(set(entry['user_id'] for entry in historical_data))[:5]

for user_id in unique_users:
    print(f"\nAnalyzing performance for User ID: {user_id}")
    performance = extract_user_performance(historical_data, user_id)
    topic_analysis = analyze_student_performance(historical_data, user_id)
    student_persona = define_student_persona(performance, topic_analysis)
    recommendations = generate_recommendations(performance, topic_analysis)
    predicted_rank = predict_neet_rank(performance)

    # Display results
    print("User Performance by Topic:")
    for topic, data in topic_analysis.items():
        print(f"- {topic}: Avg Accuracy {data['avg_accuracy']:.2f}% (Difficulty: {data['difficulty']}, Trend: {data['trend']})")

    print("\nStudent Persona:", student_persona)
    print("\nUser Recommendations:")
    for rec in recommendations:
        print("-", rec)
    print("\nPredicted NEET Rank:", predicted_rank)

    # Show performance trend
    plot_performance_trend(performance)

