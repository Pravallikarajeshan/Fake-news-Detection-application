import streamlit as st
import pickle

# Load your actual model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page setup
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")

# Background image and style
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://t3.ftcdn.net/jpg/02/68/67/02/360_F_268670299_DWFCUfBIgKMNAsThzlptboVcgVcHun4y.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

h1, h2, h3 {
    color: white;
    text-shadow: 1px 1px 3px black;
}

div.stTextInput > label {
    font-size: 1.2rem;
    color: white;
    font-weight: bold;
    text-shadow: 1px 1px 2px black;
}

.stButton > button {
    background-color: #ff4b4b;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>📰 Fake News Detection App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Check if a news article is Fake or Real 💬</h3>", unsafe_allow_html=True)

# Input section
user_input = st.text_area("✍️ Enter News Headline or Content:")

if st.button("🔍 Check Now"):
    if user_input.strip() == "":
        st.warning("Please enter a news text to analyze.")
    else:
        vec = vectorizer.transform([user_input])
        result = model.predict(vec)

        if result[0] == 1:
            st.success("✅ This news is likely *Real*.")
        else:
            st.error("🟥 This news is likely *Fake*.")

# Sidebar with educational content
# Sidebar with journalism support
st.sidebar.title("🧠 Journalism Toolkit")

# 1. Top Skills for Modern Journalists
st.sidebar.markdown("### 🛠️ Top Skills for Journalism")
st.sidebar.info("""
- Digital Fact Checking  
- Data Journalism  
- Investigative Reporting  
- Mobile Journalism (MoJo)  
- Audio & Video Editing  
- Content Strategy & SEO  
""")

# 2. Latest News Headlines (manually added for now)
st.sidebar.markdown("### 📚 Must-Read Books for Aspiring Journalists")
st.sidebar.success("""
- 🖋 **"The Elements of Journalism"** – Bill Kovach & Tom Rosenstiel  
  *Principles every journalist should know.*

- 📖 **"On Writing Well"** – William Zinsser  
  *A classic on clarity, simplicity, and writing with purpose.*

- 🕵️‍♀️ **"News Reporting and Writing"** – Melvin Mencher  
  *Practical guide to real-world reporting.*

- 🌐 **"Digital Journalism"** – Janet Jones & Lee Salter  
  *Explores journalism in the age of the internet.*

- 🗞 **"The Investigative Reporter’s Handbook"** – Brant Houston  
  *Great for learning in-depth reporting techniques.*
""")

# 3. Free Courses to Upskill
st.sidebar.markdown("### 🎓 Free Journalism Courses")
st.sidebar.warning("""
- [Coursera: Fake News Detection](https://www.coursera.org/learn/fake-news)  
- [edX: Journalism in Digital Age](https://www.edx.org/course/journalism)  
- [Google News Training](https://newsinitiative.withgoogle.com/training/)  
- [FutureLearn: Digital Skills for Media](https://www.futurelearn.com/)  
""")

# 4. Trusted Sources
st.sidebar.markdown("### 🏛️ Trusted News Platforms")
st.sidebar.markdown("""
- [BBC News](https://www.bbc.com)  
- [Reuters](https://www.reuters.com)  
- [Alt News (Fact Check)](https://www.altnews.in)  
- [The Hindu](https://www.thehindu.com)  
""")





