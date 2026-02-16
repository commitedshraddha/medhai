# MedhAI+ - AI-Driven Predictive Adaptive Learning System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📚 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Machine Learning Model](#machine-learning-model)
- [API Endpoints](#api-endpoints)
- [Data Format](#data-format)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## 🎯 Overview

**MedhAI+** is an intelligent adaptive learning platform that leverages Machine Learning to provide personalized education experiences. The system analyzes student performance metrics in real-time and dynamically adjusts quiz difficulty and learning paths to optimize learning outcomes.

### Key Objectives:
- **Personalization**: Tailor learning experiences to individual student needs
- **Prediction**: Use ML to predict optimal learning paths
- **Adaptation**: Dynamically adjust quiz difficulty based on performance
- **Engagement**: Track and improve student engagement metrics
- **Analytics**: Provide comprehensive performance dashboards

## ✨ Features

### 🤖 AI-Powered Learning Path Prediction
- **Random Forest Classifier** trained on 12+ student performance features
- Real-time prediction with 80-95% confidence
- Three learning paths:
  - 🔴 **Remedial Support** - Foundation building
  - 🟡 **Continue Current Level** - Skill reinforcement
  - 🟢 **Advance** - Challenge mode

### 📊 Comprehensive Student Dashboard
- Visual representation of performance metrics
- Progress tracking across multiple dimensions:
  - Quiz scores and exam performance
  - Engagement levels
  - Exercise completion rates
  - Time spent on platform
  - Attendance percentage
  - Skill improvements (Vocabulary, Grammar, Reading, etc.)

### 🎯 Adaptive Quiz System
- Difficulty-adjusted question banks
- Three difficulty levels (Easy, Medium, Hard)
- Real-time quiz evaluation
- Immediate feedback and recommendations

### 📈 Performance Analytics
- Multi-dimensional performance visualization
- Confidence scoring for predictions
- Historical performance tracking
- Feature importance analysis

## 🏗️ System Architecture

```
┌─────────────────┐
│   User Input    │
│  (Web Interface)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Flask Backend  │
│   (app.py)      │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────┐
│Session │ │   ML Model   │
│Manager │ │(Random Forest)│
└────────┘ └──────┬───────┘
              │
         ┌────┴────┐
         │         │
         ▼         ▼
    ┌────────┐ ┌──────────┐
    │Feature │ │Prediction│
    │Extract │ │Confidence│
    └────────┘ └──────────┘
         │         │
         └────┬────┘
              ▼
    ┌──────────────────┐
    │Learning Path Rec │
    │  + Quiz Selection│
    └──────────────────┘
```

## 🛠️ Technology Stack

### Backend
- **Python 3.8+** - Core programming language
- **Flask 3.0.0** - Web framework
- **scikit-learn 1.3.2** - Machine Learning library
- **NumPy 1.26.2** - Numerical computing
- **Pandas 2.1.3** - Data manipulation

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling
- **JavaScript** - Interactivity
- **Responsive Design** - Mobile-friendly interface

### Machine Learning
- **Random Forest Classifier** - Primary ML model
- **12 Feature Inputs** - Comprehensive student metrics
- **3 Output Classes** - Learning path recommendations

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd medhai_app
   ```

2. **Create Virtual Environment**
   ```bash
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   venv\Scripts\activate.ps1
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the ML Model** (if not already trained)
   ```bash
   python model/train_model.py
   # OR for improved version
   python model/train_improved.py
   ```

5. **Verify Installation**
   ```bash
   # Check if model file exists
   ls model/random_forest.pkl

   # Analyze dataset (optional)
   python analyze_dataset.py
   ```

6. **Run the Application**
   ```bash
   python app.py
   ```

7. **Access the Application**
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## 🚀 Usage

### 1. Home Page
- Start by entering student performance data
- Input 12 key metrics for comprehensive analysis

### 2. Get Prediction
- Click "Get Recommendation" to receive AI-powered learning path
- View confidence score and detailed analysis

### 3. View Dashboard
- Explore comprehensive performance visualization
- Analyze strengths and areas for improvement

### 4. Take Quiz
- Attempt difficulty-adjusted quizzes
- Get immediate feedback on performance

### 5. View Results
- Review quiz performance
- Receive personalized recommendations

## 📂 Project Structure

```
medhai_app/
│
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── create_pragmatic_model.py       # Rule-based model creator
├── analyze_dataset.py              # Dataset analysis tool
├── test_dataset.py                 # Dataset validation
│
├── data/                           # Data directory
│   ├── adaptive_learning_rich_dataset.csv  # Training dataset
│   ├── easy_questions.json         # Easy quiz questions
│   ├── medium_questions.json       # Medium quiz questions
│   ├── hard_questions.json         # Hard quiz questions
│   └── mald_sample.csv            # Sample data
│
├── model/                          # ML model directory
│   ├── random_forest.pkl          # Trained model (generated)
│   ├── train_model.py             # Model training script
│   └── train_improved.py          # Improved training script
│
├── static/                         # Static files
│   ├── css/
│   │   └── style.css              # Application styles
│   └── js/
│       └── script.js              # Frontend JavaScript
│
└── templates/                      # HTML templates
    ├── base.html                   # Base template
    ├── index.html                  # Home page
    ├── dashboard.html              # Performance dashboard
    ├── quiz.html                   # Quiz interface
    └── result.html                 # Results page
```

## 🧠 Machine Learning Model

### Features (Input)
The model uses 12 comprehensive features:

1. **Quiz_Scores** (0-100) - Average quiz performance
2. **Final_Exam_Score** (0-100) - Exam performance
3. **Engagement_Level** (1-3) - Low/Medium/High engagement
4. **Exercise_Completion_Rate** (0-100) - Percentage completed
5. **Time_Spent_On_Learning_Platform** (hours) - Platform usage
6. **Attendance_Percentage** (0-100) - Class attendance
7. **Vocabulary_Improvement_Score** (0-100) - Vocabulary progress
8. **Grammar_Improvement_Score** (0-100) - Grammar progress
9. **Reading_Ability_Score** (0-100) - Reading comprehension
10. **Listening_Ability_Score** (0-100) - Listening skills
11. **Writing_Ability_Score** (0-100) - Writing proficiency
12. **Speaking_Ability_Score** (0-100) - Speaking ability

### Output Classes
- **0** - Remedial Support (Foundation building)
- **1** - Continue Current Level (Skill reinforcement)
- **2** - Advance (Challenge mode)

### Model Performance
- **Algorithm**: Random Forest Classifier
- **Confidence Range**: 80-95%
- **Cross-validation**: Implemented
- **Feature Importance**: Analyzed and visualized

### Training the Model
```bash
# Standard training
python model/train_model.py

# Improved training with hyperparameter tuning
python model/train_improved.py
```

## 🔌 API Endpoints

### POST `/predict`
**Description**: Get learning path prediction

**Request Body**:
```json
{
  "Quiz_Scores": 75,
  "Final_Exam_Score": 80,
  "Engagement_Level": 2,
  "Exercise_Completion_Rate": 85,
  "Time_Spent_On_Learning_Platform": 15,
  "Attendance_Percentage": 90,
  "Vocabulary_Improvement_Score": 70,
  "Grammar_Improvement_Score": 75,
  "Reading_Ability_Score": 80,
  "Listening_Ability_Score": 78,
  "Writing_Ability_Score": 72,
  "Speaking_Ability_Score": 74
}
```

**Response**:
```json
{
  "prediction": 1,
  "learning_path": "Continue Current Level",
  "confidence": 87.5,
  "recommendation": "You're doing well! Keep practicing...",
  "quiz_file": "medium_questions.json"
}
```

### GET `/quiz/<difficulty>`
**Description**: Load quiz questions by difficulty

**Parameters**:
- `difficulty`: "easy", "medium", or "hard"

### POST `/submit-quiz`
**Description**: Submit quiz answers and get results

**Request Body**:
```json
{
  "answers": [1, 3, 2, 0, 1],
  "quiz_file": "medium_questions.json"
}
```

## 📊 Data Format

### Training Dataset Format
CSV file with the following columns:
```csv
Quiz_Scores,Final_Exam_Score,Engagement_Level,Exercise_Completion_Rate,
Time_Spent_On_Learning_Platform,Attendance_Percentage,
Vocabulary_Improvement_Score,Grammar_Improvement_Score,
Reading_Ability_Score,Listening_Ability_Score,
Writing_Ability_Score,Speaking_Ability_Score,Learning_Path
```

### Quiz Questions Format
JSON file with the following structure:
```json
[
  {
    "question": "Question text here?",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "correct": 0
  }
]
```

## 🖼️ Screenshots

*(Add screenshots of your application here)*

### Home Page
![Home Page - Student Performance Input]

### Dashboard
![Performance Dashboard - Visual Analytics]

### Quiz Interface
![Adaptive Quiz - Dynamic Difficulty]

### Results Page
![Results and Recommendations]

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit Changes**
   ```bash
   git commit -m "Add your feature"
   ```
4. **Push to Branch**
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**

### Coding Standards
- Follow PEP 8 for Python code
- Add docstrings to all functions
- Write unit tests for new features
- Update documentation as needed

## 🔧 Troubleshooting

### Model File Not Found
```bash
Error: Model file not found at model/random_forest.pkl
Solution: Run python model/train_model.py
```

### Import Errors
```bash
Error: ModuleNotFoundError: No module named 'flask'
Solution: pip install -r requirements.txt
```

### Port Already in Use
```bash
Error: Address already in use
Solution: Change port in app.py or kill existing process
```

### Low Model Confidence
- Ensure training data is comprehensive
- Check for missing or incorrect values
- Retrain model with more data

## 🚀 Future Enhancements

### Planned Features
- [ ] **User Authentication** - Multi-user support with login system
- [ ] **Database Integration** - PostgreSQL/MySQL for data persistence
- [ ] **Real-time Collaboration** - Multi-student learning sessions
- [ ] **Advanced Analytics** - Predictive analytics dashboard
- [ ] **Content Management** - Dynamic question bank management
- [ ] **Mobile App** - Native iOS/Android applications
- [ ] **Gamification** - Badges, leaderboards, achievements
- [ ] **AI Tutor** - NLP-powered chatbot assistance
- [ ] **Video Integration** - Adaptive video recommendations
- [ ] **Parent Portal** - Progress tracking for parents/guardians

### Model Improvements
- [ ] Deep Learning models (Neural Networks)
- [ ] Ensemble methods
- [ ] Reinforcement Learning for adaptive difficulty
- [ ] Natural Language Processing for essay evaluation
- [ ] Computer Vision for engagement detection

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## �‍💻 Developer

### **Shraddha Tiwari**
*Lead Developer & Creator of MedhAI+*

[![Email](https://img.shields.io/badge/Email-shraddhatiwari732%40gmail.com-red?style=for-the-badge&logo=gmail)](mailto:shraddhatiwari732@gmail.com)

- 🎓 AI/ML Development
- 💻 Full-Stack Development
- 📚 Educational Technology
- 🎨 UX/UI Design

## 📧 Contact

For questions, suggestions, or collaboration:
- 📩 **Email**: [shraddhatiwari732@gmail.com](mailto:shraddhatiwari732@gmail.com)
- 🐛 **GitHub Issues**: [Project Issues](https://github.com/your-repo/issues)

## 🙏 Acknowledgments

- scikit-learn community for excellent ML tools
- Flask framework for robust web development
- Educational institutions for dataset insights
- Open-source community for inspiration

---

**Made with ❤️ by Shraddha Tiwari**

*Empowering learners through AI-driven personalization*
