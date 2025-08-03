# 🧙‍♂️ Harry Potter Character Prediction Quiz

Welcome to the **Harry Potter Character Prediction App**, a fun and interactive quiz that matches your personality traits to a popular Harry Potter character using a machine learning model. 🚀
---

## ✨ Features

- 📋 Personality-based quiz (5–10 questions)
- 🧠 Machine Learning model (Random Forest Classifier)
- 🖼️ Predicted character image and name
- 💬 Shareable result message
- 🔗 One-click LinkedIn sharing
- 🧾 Answer logging with Excel backend
- 🔐 Safe, local, and interactive

---

## 🗂️ Project Structure

/Divakar2Divu/Harry-Potter
├── app.py # Main Streamlit app
├── train_model.py # Script to train and save ML model
├── harry_potter_quiz_data.xlsx # Excel file with questions and answers
├── character_images/ # Folder with character images
│ ├── Harry Potter.jpg
│ ├── Hermione Granger.jpg
│ └── ...
├── saved_model/
│ ├── random_forest_model.pkl
│ ├── label_encoders.pkl
│ └── target_encoder.pkl
├── requirements.txt # Python dependencies
└── README.md # You are here



---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Divakar2Divu/Harry-Potter.git
cd Harry-Potter

2. Install Dependencies

pip install -r requirements.txt

3. Run the App
streamlit run app.py


📌 To Do / Future Additions
 Add 10+ more personality questions

 Add result image download feature

 Add social sharing buttons (WhatsApp, LinkedIn)

 Add dark mode switch


