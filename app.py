"""
MedhAI+ Flask Application
==========================
An AI-driven Predictive Adaptive Learning System that uses Machine Learning 
to recommend personalized learning paths and adapt quiz difficulty dynamically.

Author: MedhAI+ Team
Date: November 2025
"""

from flask import Flask, render_template, request, jsonify, session
import pickle
import os
import json
import numpy as np
from datetime import datetime

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'medhai_plus_secret_key_2025'  # For session management

# Global variable to store the ML model
model = None

# Path mappings for learning paths
LEARNING_PATHS = {
    0: {
        'name': 'Remedial Support',
        'difficulty': 'Easy',
        'color': '#e74c3c',
        'recommendation': 'Focus on foundational concepts before advancing. Take your time to master the basics!',
        'quiz_file': 'easy_questions.json'
    },
    1: {
        'name': 'Continue Current Level',
        'difficulty': 'Medium',
        'color': '#f39c12',
        'recommendation': 'You\'re doing well! Keep practicing at this level to solidify your understanding.',
        'quiz_file': 'medium_questions.json'
    },
    2: {
        'name': 'Advance',
        'difficulty': 'Hard',
        'color': '#27ae60',
        'recommendation': 'Excellent work! Challenge yourself with advanced topics to reach your full potential.',
        'quiz_file': 'hard_questions.json'
    }
}

def load_model():
    """
    Load the pre-trained Random Forest model from pickle file.
    Model package includes the model and metadata.
    
    Returns:
        dict: Model package containing model and feature information
    """
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'random_forest.pkl')
        
        if not os.path.exists(model_path):
            print(f"⚠️  Warning: Model file not found at {model_path}")
            print("   Please run: python model/train_model.py")
            return None
        
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        # Handle both old format (just model) and new format (dict with metadata)
        if isinstance(model_package, dict):
            model = model_package
            print("✓ ML Model loaded successfully (with metadata)")
            print(f"  Core features: {len(model['core_features'])}")
            print(f"  All features: {len(model['all_features'])}")
        else:
            # Old format - wrap it
            model = {
                'model': model_package,
                'core_features': [
                    'Quiz_Scores', 'Final_Exam_Score', 'Engagement_Level', 
                    'Exercise_Completion_Rate', 'Time_Spent_On_Learning_Platform'
                ],
                'all_features': [
                    'Quiz_Scores', 'Final_Exam_Score', 'Engagement_Level', 
                    'Exercise_Completion_Rate', 'Time_Spent_On_Learning_Platform',
                    'Attendance_Percentage', 'Vocabulary_Improvement_Score',
                    'Grammar_Improvement_Score', 'Reading_Ability_Score',
                    'Listening_Ability_Score', 'Writing_Ability_Score',
                    'Speaking_Ability_Score'
                ],
                'path_names': ['Remedial Support', 'Continue Current Level', 'Advance'],
                'engagement_mapping': {'Low': 1, 'Medium': 2, 'High': 3}
            }
            print("✓ ML Model loaded successfully (old format, wrapped)")
        
        return model
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return None

def load_quiz_questions(difficulty_level):
    """
    Load quiz questions based on difficulty level.
    
    Args:
        difficulty_level: String indicating difficulty ('Easy', 'Medium', 'Hard')
        
    Returns:
        List of question dictionaries
    """
    # Map difficulty to filename
    filename_map = {
        'Easy': 'easy_questions.json',
        'Medium': 'medium_questions.json',
        'Hard': 'hard_questions.json'
    }
    
    filename = filename_map.get(difficulty_level, 'medium_questions.json')
    filepath = os.path.join(os.path.dirname(__file__), 'data', filename)
    
    try:
        with open(filepath, 'r') as f:
            questions = json.load(f)
        return questions
    except Exception as e:
        print(f"Error loading quiz questions: {e}")
        return []

# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    """
    Homepage route - renders the student input form.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint - receives student data and returns ML prediction.
    
    Expected JSON payload:
    {
        "quiz_score": 80,
        "final_exam_score": 85,
        "engagement_level": 7,
        "exercise_completion": 85,
        "time_spent": 3.5
    }
    
    Returns:
        JSON with learning_path, difficulty, recommendation, and color
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input data
        required_fields = ['quiz_score', 'final_exam_score', 'engagement_level', 
                          'exercise_completion', 'time_spent']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract features
        quiz_score = float(data['quiz_score'])
        final_exam_score = float(data['final_exam_score'])
        engagement_level = float(data['engagement_level'])
        exercise_completion = float(data['exercise_completion'])
        time_spent = float(data['time_spent'])
        
        # Validate ranges
        if not (0 <= quiz_score <= 100):
            return jsonify({'error': 'Quiz score must be between 0 and 100'}), 400
        if not (0 <= final_exam_score <= 100):
            return jsonify({'error': 'Final exam score must be between 0 and 100'}), 400
        if not (1 <= engagement_level <= 10):
            return jsonify({'error': 'Engagement level must be between 1 and 10'}), 400
        if not (0 <= exercise_completion <= 100):
            return jsonify({'error': 'Exercise completion must be between 0 and 100'}), 400
        if time_spent < 0:
            return jsonify({'error': 'Time spent must be positive'}), 400
        
        # Convert engagement level (1-10 scale) to Low/Medium/High (1/2/3)
        if engagement_level <= 3:
            engagement_numeric = 1  # Low
        elif engagement_level <= 7:
            engagement_numeric = 2  # Medium
        else:
            engagement_numeric = 3  # High
        
        # Prepare features for prediction - expand 5 core features to 12 total features
        # Core features: Quiz, Exam, Engagement, Completion, Time
        # Additional features (estimated from core features):
        avg_performance = (quiz_score + final_exam_score) / 2
        attendance = min(100, exercise_completion + 5)  # Assume attendance correlates with completion
        
        # Skill scores estimated from exam and quiz performance
        vocab_score = avg_performance * 0.95  # Slightly lower than avg
        grammar_score = avg_performance * 0.98
        reading_score = avg_performance * 1.02
        listening_score = avg_performance * 0.97
        writing_score = avg_performance * 0.96
        speaking_score = avg_performance * 0.94
        
        # Create feature array with all 12 features
        features = np.array([[
            quiz_score,           # 0: Quiz_Scores
            final_exam_score,     # 1: Final_Exam_Score
            engagement_numeric,   # 2: Engagement_Level (1/2/3)
            exercise_completion,  # 3: Exercise_Completion_Rate
            time_spent,           # 4: Time_Spent_On_Learning_Platform
            attendance,           # 5: Attendance_Percentage
            vocab_score,          # 6: Vocabulary_Improvement_Score
            grammar_score,        # 7: Grammar_Improvement_Score
            reading_score,        # 8: Reading_Ability_Score
            listening_score,      # 9: Listening_Ability_Score
            writing_score,        # 10: Writing_Ability_Score
            speaking_score        # 11: Speaking_Ability_Score
        ]])
        
        # Make prediction
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Get the actual model from the package
        ml_model = model['model'] if isinstance(model, dict) else model
        
        prediction = ml_model.predict(features)[0]
        probabilities = ml_model.predict_proba(features)[0]
        confidence = probabilities[prediction] * 100
        
        # Get learning path details
        path_info = LEARNING_PATHS[prediction]
        
        # Store in session for later use
        session['last_prediction'] = {
            'learning_path': path_info['name'],
            'difficulty': path_info['difficulty'],
            'prediction_code': int(prediction),
            'confidence': float(confidence),
            'timestamp': datetime.now().isoformat()
        }
        
        # Return prediction result
        response = {
            'learning_path': path_info['name'],
            'difficulty': path_info['difficulty'],
            'recommendation': path_info['recommendation'],
            'color': path_info['color'],
            'confidence': round(confidence, 2),
            'probabilities': {
                'remedial': round(float(probabilities[0]) * 100, 2),
                'continue': round(float(probabilities[1]) * 100, 2),
                'advance': round(float(probabilities[2]) * 100, 2)
            }
        }
        
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/quiz')
def quiz():
    """
    Quiz page route - starts adaptive quiz using ML prediction from session.
    Supports optional difficulty query parameter for direct access.
    """
    # Check for direct difficulty parameter (e.g., /quiz?difficulty=Medium)
    difficulty_param = request.args.get('difficulty', None)
    
    # Get the last prediction from session
    prediction = session.get('last_prediction', {})
    
    # Priority: URL param > Prediction > Default Medium
    if difficulty_param and difficulty_param in ['Easy', 'Medium', 'Hard']:
        starting_difficulty = difficulty_param
    elif prediction:
        starting_difficulty = prediction.get('difficulty', 'Medium')
    else:
        starting_difficulty = 'Medium'
    
    print(f"🎯 Starting quiz with difficulty: {starting_difficulty}")
    
    # Initialize adaptive quiz state
    session['adaptive_quiz'] = {
        'current_difficulty': starting_difficulty,
        'questions_answered': 0,
        'correct_streak': 0,
        'incorrect_streak': 0,
        'performance_history': []
    }
    session.modified = True
    
    return render_template('quiz.html', starting_difficulty=starting_difficulty)

@app.route('/get_questions', methods=['POST'])
def get_questions():
    """
    API endpoint to fetch quiz questions based on difficulty.
    Now supports adaptive difficulty changes mid-quiz.
    
    Expected JSON:
    {
        "difficulty": "Medium"
    }
    
    Returns:
        JSON array of quiz questions
    """
    try:
        data = request.get_json()
        difficulty = data.get('difficulty', 'Medium')
        
        # Load questions
        questions = load_quiz_questions(difficulty)
        
        if not questions:
            return jsonify({'error': 'No questions available'}), 404
        
        return jsonify({'questions': questions, 'difficulty': difficulty})
        
    except Exception as e:
        return jsonify({'error': f'Error loading questions: {str(e)}'}), 500

@app.route('/adaptive_next_question', methods=['POST'])
def adaptive_next_question():
    """
    ML-powered adaptive endpoint that determines next question difficulty
    based on user's current performance.
    
    Expected JSON:
    {
        "current_difficulty": "Medium",
        "last_answer_correct": true,
        "questions_answered": 3,
        "current_score": 66.67,
        "time_spent": 45
    }
    
    Returns:
        JSON with next difficulty level and reasoning
    """
    try:
        data = request.get_json()
        
        current_difficulty = data.get('current_difficulty', 'Medium')
        last_correct = data.get('last_answer_correct', False)
        questions_answered = data.get('questions_answered', 0)
        current_score = data.get('current_score', 0)
        time_spent = data.get('time_spent', 0)
        
        # Get adaptive quiz state from session
        adaptive_state = session.get('adaptive_quiz', {})
        
        # Update streaks
        if last_correct:
            adaptive_state['correct_streak'] = adaptive_state.get('correct_streak', 0) + 1
            adaptive_state['incorrect_streak'] = 0
        else:
            adaptive_state['incorrect_streak'] = adaptive_state.get('incorrect_streak', 0) + 1
            adaptive_state['correct_streak'] = 0
        
        # Store performance
        if 'performance_history' not in adaptive_state:
            adaptive_state['performance_history'] = []
        adaptive_state['performance_history'].append({
            'difficulty': current_difficulty,
            'correct': last_correct,
            'score': current_score
        })
        
        # Calculate estimated metrics for ML prediction
        # Map difficulty to numeric values
        difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
        current_diff_value = difficulty_map.get(current_difficulty, 2)
        
        # Estimate quiz score based on current performance
        estimated_quiz_score = current_score
        
        # Estimate final exam score (slightly higher than quiz for optimism)
        estimated_exam_score = min(100, current_score + 5)
        
        # Engagement level based on correct streak and time
        engagement = 5  # baseline
        if adaptive_state['correct_streak'] >= 3:
            engagement = min(10, 7 + adaptive_state['correct_streak'])
        elif adaptive_state['incorrect_streak'] >= 3:
            engagement = max(1, 5 - adaptive_state['incorrect_streak'])
        else:
            engagement = 6 if current_score >= 60 else 4
        
        # Exercise completion (how many questions answered)
        completion = (questions_answered / 10) * 100
        
        # Time spent per question (in hours)
        time_per_question = time_spent / max(1, questions_answered)
        time_in_hours = (time_per_question * 10) / 3600  # Estimate for full quiz
        
        # Convert engagement (1-10) to Low/Medium/High (1/2/3)
        if engagement <= 3:
            engagement_numeric = 1
        elif engagement <= 7:
            engagement_numeric = 2
        else:
            engagement_numeric = 3
        
        # Expand to 12 features like in predict endpoint
        avg_performance = (estimated_quiz_score + estimated_exam_score) / 2
        attendance = min(100, completion + 5)
        
        vocab_score = avg_performance * 0.95
        grammar_score = avg_performance * 0.98
        reading_score = avg_performance * 1.02
        listening_score = avg_performance * 0.97
        writing_score = avg_performance * 0.96
        speaking_score = avg_performance * 0.94
        
        # Prepare features for ML prediction (12 features)
        features = np.array([[
            estimated_quiz_score,    # Quiz_Scores
            estimated_exam_score,    # Final_Exam_Score
            engagement_numeric,      # Engagement_Level (1/2/3)
            completion,              # Exercise_Completion_Rate
            max(0.5, min(6, time_in_hours)),  # Time_Spent_On_Learning_Platform
            attendance,              # Attendance_Percentage
            vocab_score,             # Vocabulary_Improvement_Score
            grammar_score,           # Grammar_Improvement_Score
            reading_score,           # Reading_Ability_Score
            listening_score,         # Listening_Ability_Score
            writing_score,           # Writing_Ability_Score
            speaking_score           # Speaking_Ability_Score
        ]])
        
        # Make ML prediction
        if model is None:
            # Fallback to rule-based if model not loaded
            next_difficulty = rule_based_adaptation(current_difficulty, current_score, 
                                                   adaptive_state['correct_streak'],
                                                   adaptive_state['incorrect_streak'])
            confidence = 75
            reasoning = "Rule-based adaptation (ML model not available)"
        else:
            # Get the actual model from the package
            ml_model = model['model'] if isinstance(model, dict) else model
            
            prediction = ml_model.predict(features)[0]
            probabilities = ml_model.predict_proba(features)[0]
            confidence = probabilities[prediction] * 100
            
            # Map prediction to difficulty
            path_to_difficulty = {
                0: 'Easy',      # Remedial Support
                1: 'Medium',    # Continue Current Level
                2: 'Hard'       # Advance
            }
            ml_suggested_difficulty = path_to_difficulty[prediction]
            
            # Apply smart adaptation rules
            next_difficulty = smart_adaptation(
                current_difficulty, 
                ml_suggested_difficulty,
                current_score,
                adaptive_state['correct_streak'],
                adaptive_state['incorrect_streak'],
                questions_answered
            )
            
            reasoning = f"ML model suggests {ml_suggested_difficulty} level (confidence: {confidence:.1f}%)"
        
        # Update session
        adaptive_state['current_difficulty'] = next_difficulty
        adaptive_state['questions_answered'] = questions_answered
        session['adaptive_quiz'] = adaptive_state
        session.modified = True
        
        response = {
            'next_difficulty': next_difficulty,
            'previous_difficulty': current_difficulty,
            'difficulty_changed': next_difficulty != current_difficulty,
            'confidence': round(confidence, 2),
            'reasoning': reasoning,
            'correct_streak': adaptive_state['correct_streak'],
            'incorrect_streak': adaptive_state['incorrect_streak'],
            'adaptive_message': get_adaptive_message(next_difficulty, current_difficulty, last_correct)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Adaptation error: {str(e)}'}), 500

def rule_based_adaptation(current_diff, score, correct_streak, incorrect_streak):
    """Fallback rule-based adaptation if ML model fails."""
    if score >= 80 and correct_streak >= 2:
        return 'Hard' if current_diff != 'Hard' else 'Hard'
    elif score >= 60:
        return 'Medium'
    elif score < 50 and incorrect_streak >= 2:
        return 'Easy'
    return current_diff

def smart_adaptation(current_diff, ml_suggested_diff, score, correct_streak, incorrect_streak, questions_answered):
    """
    Intelligent adaptation that combines ML suggestion with performance patterns.
    Prevents too-frequent difficulty changes for better learning experience.
    """
    # Don't change difficulty in first 2 questions
    if questions_answered <= 2:
        return current_diff
    
    # Strong performance indicators - allow upgrade
    if correct_streak >= 3 and score >= 75:
        if current_diff == 'Easy':
            return 'Medium'
        elif current_diff == 'Medium' and score >= 85:
            return 'Hard'
    
    # Struggling indicators - allow downgrade
    if incorrect_streak >= 3 or (score < 40 and questions_answered >= 3):
        if current_diff == 'Hard':
            return 'Medium'
        elif current_diff == 'Medium' and score < 35:
            return 'Easy'
    
    # Otherwise, trust ML suggestion but avoid jumping 2 levels
    if ml_suggested_diff == 'Hard' and current_diff == 'Easy':
        return 'Medium'  # Don't jump from Easy to Hard
    elif ml_suggested_diff == 'Easy' and current_diff == 'Hard':
        return 'Medium'  # Don't drop from Hard to Easy
    
    return ml_suggested_diff

def get_adaptive_message(next_diff, current_diff, last_correct):
    """Generate encouraging message for difficulty changes."""
    if next_diff == current_diff:
        return "Keep going at this level!"
    elif next_diff > current_diff:  # Comparing string lengths works here
        if last_correct:
            return f"🎉 Great job! Moving to {next_diff} difficulty!"
        else:
            return f"📈 Challenging you with {next_diff} level questions!"
    else:
        return f"💡 Adjusting to {next_diff} to build your foundation!"

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    """
    Endpoint to submit quiz answers and calculate score.
    
    Expected JSON:
    {
        "answers": {"1": "Paris", "2": "Mercury", ...},
        "difficulty": "Medium",
        "time_spent": 180,
        "questions": [...]
    }
    
    Returns:
        JSON with score, correct_answers, feedback, and analytics
    """
    try:
        data = request.get_json()
        print(f"\n📝 Quiz submission received:")
        print(f"  Answers: {len(data.get('answers', {}))} questions")
        print(f"  Difficulty: {data.get('difficulty', 'Unknown')}")
        print(f"  Time: {data.get('time_spent', 0)} seconds")
        
        answers = data.get('answers', {})
        difficulty = data.get('difficulty', 'Medium')
        time_spent = data.get('time_spent', 0)
        questions = data.get('questions', [])
        
        # If questions not provided, load from file
        if not questions:
            questions = load_quiz_questions(difficulty)
        
        if not questions:
            return jsonify({'error': 'Questions not found'}), 404
        
        # Calculate score
        correct_count = 0
        total_questions = len(questions)
        correct_answers = {}
        user_answers_detailed = {}
        topic_performance = {}
        
        for q in questions:
            q_id = str(q['id'])
            correct_answer = q['answer']
            correct_answers[q_id] = correct_answer
            topic = q.get('topic', 'General')
            
            # Initialize topic tracking
            if topic not in topic_performance:
                topic_performance[topic] = {'correct': 0, 'total': 0}
            
            topic_performance[topic]['total'] += 1
            
            # Check if answer is correct
            user_answer = answers.get(q_id, '')
            is_correct = user_answer == correct_answer
            
            if is_correct:
                correct_count += 1
                topic_performance[topic]['correct'] += 1
            
            user_answers_detailed[q_id] = {
                'question': q['question'],
                'user_answer': user_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'topic': topic
            }
        
        score_percentage = (correct_count / total_questions) * 100 if total_questions > 0 else 0
        
        # Calculate topic-wise performance
        topic_scores = {}
        for topic, perf in topic_performance.items():
            topic_scores[topic] = {
                'percentage': (perf['correct'] / perf['total']) * 100 if perf['total'] > 0 else 0,
                'correct': perf['correct'],
                'total': perf['total']
            }
        
        # Calculate estimated engagement based on performance
        engagement_score = min(10, max(1, int((score_percentage / 10) + (5 if time_spent < 300 else 3))))
        
        # Estimate exercise completion based on answered questions
        answered_count = len(answers)
        completion_rate = (answered_count / total_questions) * 100 if total_questions > 0 else 0
        
        # Store quiz result in session
        if 'quiz_history' not in session:
            session['quiz_history'] = []
        
        quiz_result = {
            'difficulty': difficulty,
            'score': score_percentage,
            'correct': correct_count,
            'total': total_questions,
            'time_spent': time_spent,
            'engagement_score': engagement_score,
            'completion_rate': completion_rate,  # Match template expectation
            'timestamp': datetime.now().isoformat(),
            'topic_scores': topic_scores,
            'feedback': '',  # Will be set below
            'next_action': ''  # Will be set below
        }
        
        session['quiz_history'].append(quiz_result)
        session.modified = True
        
        print(f"✅ Quiz result stored in session:")
        print(f"  Score: {score_percentage:.1f}%")
        print(f"  Correct: {correct_count}/{total_questions}")
        print(f"  Session has {len(session['quiz_history'])} quiz(zes)")
        
        # Provide detailed feedback based on score
        if score_percentage >= 90:
            feedback = "🏆 Outstanding! You've mastered this difficulty level!"
            next_action = "advance"
            suggestion = "Try the next difficulty level to challenge yourself further."
        elif score_percentage >= 80:
            feedback = "⭐ Excellent work! You have a strong grasp of the material."
            next_action = "advance"
            suggestion = "You're ready for more advanced challenges!"
        elif score_percentage >= 70:
            feedback = "👍 Good job! You're on the right track."
            next_action = "continue"
            suggestion = "Review the missed questions and try again to improve."
        elif score_percentage >= 60:
            feedback = "✓ You passed! Keep practicing to build confidence."
            next_action = "continue"
            suggestion = "Focus on topics where you scored lower to strengthen your skills."
        else:
            feedback = "📚 Keep learning! Understanding takes time and practice."
            next_action = "remedial"
            suggestion = "Review the fundamentals and try an easier difficulty to build your foundation."
        
        # Update the quiz_result in session with feedback
        session['quiz_history'][-1]['feedback'] = feedback
        session['quiz_history'][-1]['next_action'] = next_action
        session.modified = True
        
        # Prepare response
        response = {
            'score': round(score_percentage, 2),
            'correct': correct_count,
            'total': total_questions,
            'correct_answers': correct_answers,
            'user_answers': user_answers_detailed,
            'feedback': feedback,
            'suggestion': suggestion,
            'next_action': next_action,
            'time_spent': time_spent,
            'engagement_score': engagement_score,
            'completion_rate': round(completion_rate, 2),
            'topic_performance': topic_scores,
            'analytics': {
                'average_time_per_question': round(time_spent / total_questions, 1) if total_questions > 0 else 0,
                'accuracy_rate': round(score_percentage, 2),
                'unanswered_questions': total_questions - answered_count
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Error processing quiz: {str(e)}'}), 500

@app.route('/result')
def result():
    """
    Result page route - displays quiz results from session.
    """
    # Get the latest quiz result from session
    quiz_history = session.get('quiz_history', [])
    
    print(f"\n📊 Result page requested:")
    print(f"  Quiz history entries: {len(quiz_history)}")
    
    if not quiz_history:
        # No quiz results available, redirect to home
        print("  ⚠️ No quiz history, redirecting to home")
        return render_template('index.html')
    
    # Get the most recent quiz result
    latest_result = quiz_history[-1]
    
    print(f"  Latest result: score={latest_result.get('score')}, correct={latest_result.get('correct')}/{latest_result.get('total')}")
    print(f"  Passing to template: {latest_result}")
    
    # Pass data to template
    return render_template('result.html', quiz_result=latest_result)

@app.route('/dashboard')
def dashboard():
    """
    Dashboard route - displays student progress and analytics.
    """
    return render_template('dashboard.html')

@app.route('/dashboard_data')
def dashboard_data():
    """
    API endpoint to provide dashboard analytics data.
    
    Returns:
        JSON with quiz history, scores, engagement trends
    """
    try:
        # Get quiz history from session
        quiz_history = session.get('quiz_history', [])
        
        print(f"\n📊 Dashboard data requested:")
        print(f"  Quiz history: {len(quiz_history)} entries")
        
        # If no history, create sample data for demonstration
        if not quiz_history:
            print("  ⚠️ No quiz history, using sample data")
            quiz_history = [
                {'difficulty': 'Easy', 'score': 70, 'timestamp': '2025-11-01'},
                {'difficulty': 'Medium', 'score': 75, 'timestamp': '2025-11-03'},
                {'difficulty': 'Medium', 'score': 82, 'timestamp': '2025-11-05'},
                {'difficulty': 'Hard', 'score': 78, 'timestamp': '2025-11-07'}
            ]
        
        # Extract data for charts
        scores = [item['score'] for item in quiz_history]
        difficulties = [item['difficulty'] for item in quiz_history]
        
        # Calculate engagement trend (simulated based on performance)
        engagement = []
        for i, score in enumerate(scores):
            if score >= 80:
                engagement.append(min(10, 6 + i * 0.5))
            elif score >= 60:
                engagement.append(min(8, 5 + i * 0.3))
            else:
                engagement.append(max(3, 4 + i * 0.2))
        
        # Learning path progression
        paths = []
        for score in scores:
            if score >= 80:
                paths.append('Advance')
            elif score >= 60:
                paths.append('Continue')
            else:
                paths.append('Remedial')
        
        response = {
            'scores': scores,
            'engagement': engagement,
            'paths': paths,
            'difficulties': difficulties,
            'labels': [f'Quiz {i+1}' for i in range(len(scores))],
            'total_quizzes': len(quiz_history),
            'average_score': round(sum(scores) / len(scores), 2) if scores else 0
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Error fetching dashboard data: {str(e)}'}), 500

@app.route('/reset_session')
def reset_session():
    """
    Utility route to clear session data.
    """
    print("🗑️  Clearing session data...")
    session.clear()
    print("✅ Session cleared")
    return jsonify({'message': 'Session cleared successfully'})

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    print("=" * 60)
    print("MedhAI+ - Adaptive Learning System")
    print("=" * 60)
    
    # Load the ML model
    load_model()
    
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("=" * 60)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
