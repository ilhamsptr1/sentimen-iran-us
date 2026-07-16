import streamlit as st
import joblib
import re
import string
import emoji
from nltk.stem import SnowballStemmer

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="Analisis Sentimen Perang AS-Iran",
    page_icon="🌍",
    layout="centered"
)

# --- 1. KONFIGURASI PREPROCESSING ---
slang_dict = {
    'idk': 'i do not know', 'btw': 'by the way', 'omg': 'oh my god',
    'lol': 'laughing out loud', 'tbh': 'to be honest', 'pls': 'please',
    'thx': 'thanks', 'u': 'you', 'ur': 'your', 'r': 'are',
    'bc': 'because', 'b4': 'before', 'nvm': 'nevermind', 'rn': 'right now',
    'gov': 'government', 'govt': 'government', 'pres': 'president',
    'im': 'i am', 'cant': 'can not', 'dont': 'do not', 'wont': 'will not'
}

STOPWORDS_EN = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','any','both','each','few','more',
    'most','other','some','such','only','own','same','so','than','too','very',
    's','t','can','will','just','don','should','now'
}
negation_words = {'not', 'no', 'never', 'none', 'neither', 'nor', 'cannot', 'without', 'against', 'but'}
custom_stopwords = STOPWORDS_EN - negation_words
stemmer = SnowballStemmer('english')

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words  = [slang_dict.get(w, w) for w in text.split()]
    tokens = [w for w in words if w not in custom_stopwords]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

# --- 2. LOAD MODEL PADA CACHE ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('naivebayes_optimized.pkl')
    except Exception as e:
        print(e)
        return None

model = load_model()

# --- 3. TAMPILAN ANTARMUKA ---
st.title("Analisis Sentimen Perang Amerika Iran")
st.markdown("### Topik: Konflik Amerika Serikat vs Iran")

st.divider()

if model is None:
    st.error("⚠️ File `naivebayes_optimized.pkl` tidak ditemukan. Pastikan file model berada dalam satu folder dengan aplikasi ini.")
else:
    # Text Input
    tweet_input = st.text_area("✍️ Tempel (Paste) tweet atau komentar berbahasa Inggris di sini:", height=150)
    
    # Tombol Prediksi
    if st.button("🔍 Analisis Sentimen Sekarang", use_container_width=True):
        if len(tweet_input.strip()) == 0:
            st.warning("Mohon masukkan teks terlebih dahulu!")
        else:
            with st.spinner("Menganalisis teks..."):
                # Preprocessing
                teks_bersih = preprocess_text(tweet_input)
                
                if not teks_bersih:
                    st.warning("⚠️ Teks terlalu pendek atau tidak mengandung makna opini setelah dibersihkan (hanya berisi simbol / link).")
                else:
                    # Prediksi
                    prediksi_num = model.predict([teks_bersih])[0]
                    probabilitas = model.predict_proba([teks_bersih])[0]
                    
                    st.markdown("### 📊 Hasil Analisis")
                    
                    # Logika Tampilan Hasil
                    if prediksi_num == 1:
                        st.success(f"## ✅ Sentimen POSITIF")
                        conf = probabilitas[1] * 100
                        st.write(f"Teks ini terdeteksi mengandung dukungan, harapan, perdamaian, atau opini positif.")
                    else:
                        st.error(f"## ⛔ Sentimen NEGATIF")
                        conf = probabilitas[0] * 100
                        st.write(f"Teks ini terdeteksi mengandung kecaman, ketakutan, amarah, atau opini negatif (Perang/Konflik).")
                        
                    # Progress Bar Probabilitas
                    st.write(f"**Tingkat Keyakinan AI (Confidence):** {conf:.2f}%")
                    st.progress(int(conf))
                    
                    st.divider()
                    st.markdown("**Detail Preprocessing Teks:**")
                    st.info(f"Teks yang dibaca mesin setelah dibersihkan:\n\n> *{teks_bersih}*")

# Footer
st.markdown("---")
