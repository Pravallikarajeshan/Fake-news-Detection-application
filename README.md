# Fake-news-Detection-application
📰 Fake News Detection & Journalism Helper App
An all-in-one AI-powered Streamlit web app to detect fake news, analyze content sentiment, extract keywords, predict news category, and even provide book recommendations to help aspiring journalists and curious readers grow smarter every day!


🔍 What It Does
This app enables users to:

✅ Detect Fake News: Classify entered headlines/articles as Real or Fake using a trained ML model.

💬 Analyze Sentiment: See if the news article feels Positive, Neutral, or Negative using NLP.

🧠 Extract Key Phrases: Quickly highlight important noun phrases from the article.

🗂️ Predict Category: Guess if the news is about Health, Finance, Tech, or Politics (basic keyword matching).

📚 Learn & Grow: Sidebar offers curated book recommendations and resources for journalism students and enthusiasts.

🛠️ Built With
Python

Streamlit – for the interactive UI

Scikit-learn – for the machine learning model

TextBlob & spaCy – for sentiment and phrase extraction

Pickle – for loading saved ML models

💾 How to Run Locally
bash
Copy
Edit
git clone https://github.com/your-username/fake-news-journalism-helper.git
cd fake-news-journalism-helper
pip install -r requirements.txt
streamlit run app.py
Make sure these files are present:

app.py – main app script

fake_news_model.pkl – trained classification model

vectorizer.pkl – TF-IDF vectorizer used during training

📚 Recommended Books for Journalists
The Elements of Journalism – Bill Kovach

Writing Tools – Roy Peter Clark

Made to Stick – Chip Heath & Dan Heath

Data Journalism Handbook – Jonathan Gray et al.

📸 Screenshots
Fake News Detection	Sentiment Analysis
![Screenshot 2025-04-11 234607](https://github.com/user-attachments/assets/b965c4a5-2490-4451-9b67-7ec94a44e1f4)
![Screenshot 2025-04-11 234644](https://github.com/user-attachments/assets/2ccaf53d-71cb-482a-b4d5-98f15a7a38b6)
![Screenshot 2025-04-11 234658](https://github.com/user-attachments/assets/eb8023f1-e222-4b2f-a2be-dbf51fa67f58)
![Screenshot 2025-04-11 234804](https://github.com/user-attachments/assets/2f2e6e42-c525-40a1-8acc-849f23103cdd)
![Screenshot 2025-04-11 234855](https://github.com/user-attachments/assets/7ff78f9e-1966-4702-9acb-0171d8432c34)






🎯 Use Case
Designed for:

Journalism students 🧑‍🎓

Media literacy campaigns 📢

Curious readers 📖

Personal portfolio projects 💼

💡 Future Ideas
🔍 Add real-time news feed with APIs

📊 Visual analytics dashboard

📖 In-app mini courses for journalism basics

🧑‍💻 Author
Pravallika
Passionate about AI, NLP and impactful product design.
