import streamlit as st
import pickle
import os
import time
import urllib.parse
from urllib.parse import urlparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import locale
import random
from streamlit_option_menu import option_menu
from newspaper.configuration import Configuration


import newspaper
from newspaper import Article
from transformers import BertTokenizer, EncoderDecoderModel
from transformers import pipeline
from tqdm import tqdm

import yake
import re

# Optional: Selenium only if selected
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# For Streamlit only, authenticate huggingface token
from huggingface_hub import login
import os

login(token=os.environ["HF_TOKEN"])

# Read media database
URL='https://docs.google.com/spreadsheets/d/e/2PACX-1vQwxy1jmenWfyv49wzEwrp3gYE__u5JdhvVjn1c0zMUxDL6DTaU_t4Yo03qRlS4JaJWE3nK9_dIQMYZ/pub?output=csv'.format()
media_db=pd.read_csv(URL).fillna(0)

# Set config for newspaper
config = Configuration()
config.browser_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
config.request_timeout = 20  # Optional: increase timeout


# ================================
# FUNCTION ZONE
# ================================

def format_boolean_query(query):
    token_pattern = r'(\bAND\b|\bOR\b|\bNOT\b|\(|\)|"[^"]+"|\S+)'
    tokens = re.findall(token_pattern, query, flags=re.IGNORECASE)

    def parse_tokens(tokens):
        output = []
        i = 0
        while i < len(tokens):
            token = tokens[i].upper()

            if token == "AND":
                i += 1
                continue
            elif token == "OR":
                output.append("OR")
            elif token == "NOT":
                i += 1
                if i < len(tokens):
                    next_token = tokens[i]
                    if next_token.startswith("("):
                        group_tokens = []
                        paren_count = 1
                        i += 1
                        while i < len(tokens) and paren_count > 0:
                            if tokens[i] == "(":
                                paren_count += 1
                            elif tokens[i] == ")":
                                paren_count -= 1
                            if paren_count > 0:
                                group_tokens.append(tokens[i])
                            i += 1
                        group_query = parse_tokens(group_tokens)
                        output.append(f'-({group_query})')
                        i -= 1
                    else:
                        output.append(f'-{next_token}')
            else:
                output.append(tokens[i])
            i += 1
        return " ".join(output)

    return parse_tokens(tokens)


# Set locale (fallback ke C jika id_ID tidak tersedia)
try:
    locale.setlocale(locale.LC_TIME, "id_ID.utf8")
except:
    locale.setlocale(locale.LC_TIME, "C")

def convert_relative_date(text):
    text = text.lower().strip()
    today = datetime.today()
    text = text.replace("yang", "").replace("  ", " ").strip()
    date_obj = None

    if "hari lalu" in text:
        match = re.search(r"(\d+)\s+hari", text)
        if match:
            date_obj = today - timedelta(days=int(match.group(1)))

    elif "jam lalu" in text:
        match = re.search(r"(\d+)\s+jam", text)
        if match:
            date_obj = today - timedelta(hours=int(match.group(1)))

    elif "menit lalu" in text:
        date_obj = today

    elif "kemarin" in text:
        date_obj = today - timedelta(days=1)

    elif "minggu lalu" in text:
        match = re.search(r"(\d+)\s+minggu", text)
        if match:
            date_obj = today - timedelta(weeks=int(match.group(1)))

    elif "bulan lalu" in text:
        match = re.search(r"(\d+)\s+bulan", text)
        if match:
            date_obj = today - timedelta(days=int(match.group(1)) * 30)

    elif "tahun lalu" in text:
        match = re.search(r"(\d+)\s+tahun", text)
        if match:
            date_obj = today - timedelta(days=int(match.group(1)) * 365)

    elif re.match(r"\d{1,2}\s+\w+", text):
        try:
            date_obj = datetime.strptime(text + f" {today.year}", "%d %B %Y")
        except:
            return text

    if date_obj:
        return date_obj.strftime("%d %b %Y").lstrip("0")

    return text

# Fungsi untuk mengekstrak domain dari URL
def extract_domain_from_url(url):
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc
    # Buang www. kalau ada, tapi simpan subdomain lainnya
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc

# Fungsi untuk menyimpan dan memuat jadwal
def load_schedules():
    if os.path.exists("schedules.pkl"):
        with open("schedules.pkl", "rb") as f:
            return pickle.load(f)
    return []

def save_schedules(schedules):
    with open("schedules.pkl", "wb") as f:
        pickle.dump(schedules, f)


# Ubah fungsi scrape_with_bs4
def scrape_with_bs4(base_url, headers=None):
    news_results = []
    page = 0

    while True:
        start = page * 10
        url = f"{base_url}&start={start}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        results_on_page = 0

        for el in soup.select("div.SoaBEf"):
            try:
                news_results.append({
                    "Link": el.find("a")["href"],
                    "Judul": el.select_one("div.MBeuO").get_text(),
                    "Snippet": el.select_one(".GI74Re").get_text(),
                    "Tanggal": convert_relative_date(el.select_one(".LfVVr").get_text()),
                    "Media": extract_domain_from_url(el.find("a")["href"])

                })
                results_on_page += 1
            except:
                continue

        if results_on_page == 0:
            break

        page += 1
        time.sleep(random.uniform(1.5, 25.0))  # Waktu tunggu lebih pendek, bisa disesuaikan
    #NEW: Convert to dataframe & join with database
    news_results = pd.DataFrame(news_results, columns=['Link', 'Judul', 'Snippet','Tanggal','Media'])
    news_results = news_results.merge(media_db, on='Media', how='left')
    return news_results

# Ubah fungsi scrape_with_selenium
def scrape_with_selenium(base_url):
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    news_results = []
    page = 0

    while True:
        start = page * 10
        url = f"{base_url}&start={start}"
        driver.get(url)
        time.sleep(random.uniform(1.5, 25.0))

        elements = driver.find_elements(By.CSS_SELECTOR, "div.SoaBEf")
        if not elements:
            break

        for el in elements:
            try:
                link = el.find_element(By.TAG_NAME, "a").get_attribute("href")
                title = el.find_element(By.CSS_SELECTOR, "div.MBeuO").text
                snippet = el.find_element(By.CSS_SELECTOR, ".GI74Re").text
                date = convert_relative_date(el.find_element(By.CSS_SELECTOR, ".LfVVr").text)
                source = el.find_element(By.CSS_SELECTOR, ".NUnG9d span").text

                news_results.append({
                    "Link": link,
                    "Judul": title,
                    "Snippet": snippet,
                    "Tanggal": date,
                    "Media": extract_domain_from_url(el.find_element(By.TAG_NAME, "a").get_attribute("href"))

                })
            except:
                continue

        page += 1

    driver.quit()
    # NEW: Convert to dataframe & join with database
    news_results = pd.DataFrame(news_results, columns=['Link', 'Judul', 'Snippet','Tanggal','Media'])
    news_results = news_results.merge(media_db, on='Media', how='left')
    return news_results


# Ubah get_news_data untuk menghapus max_pages
def get_news_data(method, start_date, end_date, keyword_query):
    headers = {
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36"
    }

    keyword_query = format_boolean_query(keyword_query)
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    end_date_plus_one = end_date + timedelta(days=1)
    full_query = f"{keyword_query} after:{start_date} before:{end_date_plus_one}"
    encoded_query = urllib.parse.quote(full_query)

    base_url = f"https://www.google.com/search?q={encoded_query}&gl=id&hl=id&lr=lang_id&tbm=nws&num=10"

    # if method == "BeautifulSoup":
    #     return scrape_with_bs4(base_url, headers)
    # elif method == "Selenium":
    #     return scrape_with_selenium(base_url)
    # else:
    #     raise ValueError("Invalid method")

    # 🚀 First scrape:
    if method == "BeautifulSoup":
        news_df = scrape_with_bs4(base_url, headers)
    elif method == "Selenium":
        news_df = scrape_with_selenium(base_url)
    else:
        raise ValueError("Invalid method")

    # 🧠 Then enrich:
    # news_df = enrich_with_nlp(news_df)

    # ✅ Finally return:
    return news_df

# Download article & analyze sentiment
def enrich_with_nlp(df, selected_nlp=[]):
    if "Article Content" in selected_nlp:
        df['Article Content'] = ''
    if "Summary" in selected_nlp:
        df['Summary'] = ''
    if "Sentiment" in selected_nlp:
        df['Sentiment'] = ''
        df['SentimentScore'] = 0.0
    if "Keywords" in selected_nlp:
        df['Keywords'] = ''
    if "Author" in selected_nlp:
        df['Author'] = ''

    media_name_regex = r"^[A-Z][\w\s]+?\s[-:]\s[\w\s,]+(?:\d{4})?"

    kw_extractor = yake.KeywordExtractor(
        lan="id", n=3, dedupLim=0.9, top=5, features=None
    )

    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_articles = len(df)

    for idx, row in tqdm(df.iterrows(), total=total_articles, desc="Running NLP"):
        progress_percent = (idx + 1) / total_articles
        progress_bar.progress(progress_percent)
        progress_text.markdown(f"**NLP Progress:** {int(progress_percent * 100)}%")

        url = row['Link']
        try:
            article = Article(url, language="id", config=config)
            article.download()
            article.parse()

            full_text = article.text.strip()
            full_text = re.sub(media_name_regex, "", full_text)

            if not full_text:
                continue

            if "Article Content" in selected_nlp:
                df.at[idx, 'Article Content'] = full_text

            if "Summary" in selected_nlp:
                input_text = full_text[:1024] if len(full_text) > 1024 else full_text
                inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
                summary_ids = summarizer.generate(
                    inputs.input_ids,
                    max_length=100,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                df.at[idx, 'Summary'] = summary

            if "Sentiment" in selected_nlp:
                sentiment_input = full_text[:512] if len(full_text) > 512 else full_text
                sentiment = sentiment_nlp(sentiment_input)[0]
                df.at[idx, 'Sentiment'] = sentiment['label']
                df.at[idx, 'SentimentScore'] = sentiment['score']

            if "Keywords" in selected_nlp:
                keywords_with_scores = kw_extractor.extract_keywords(full_text)
                keywords = [kw for kw, _ in keywords_with_scores]
                df.at[idx, 'Keywords'] = ", ".join(keywords)

            if "Author" in selected_nlp:
                authors = article.authors
                df.at[idx, 'Author'] = ", ".join(authors) if authors else "Unknown"

        except Exception as e:
            print(f"Failed processing {url}: {e}")
            continue

    progress_bar.empty()
    progress_text.empty()

    return df

# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="Burson News Scraper", layout="centered")

# Cache model
@st.cache_resource
def load_models():
    # Article summarization
    tokenizer = BertTokenizer.from_pretrained("cahya/bert2bert-indonesian-summarization")
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    summarizer = EncoderDecoderModel.from_pretrained("cahya/bert2bert-indonesian-summarization")
    # Sentiment classifier
    pretrained_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
    sentiment_nlp = pipeline(
        "sentiment-analysis",
        model=pretrained_name,
        tokenizer=pretrained_name
)
    return summarizer, sentiment_nlp, tokenizer
# Load models
summarizer, sentiment_nlp, tokenizer = load_models()

# Sidebar Navigation
with st.sidebar:
    menu = option_menu(
        menu_title = "Main Menu",
        options=["Scrape", "Queue", "Scheduler", "How to use", "About"],
        icons=["house", "list-check","clock", "filter-circle", "diagram-3"],
        menu_icon="cast",  # optional
        default_index=0,  # optional
        styles={

                "icon": {"color": "orange"},
                "nav-link": {
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )



if menu == "Scrape":
    st.title("📰 Burson News Scraper - v0.0.2")
    st.markdown("Scrape berita berdasarkan **Boolean Keyword** dan input tanggal, lalu simpan ke Excel.")

    with st.form("scrape_form"):
        keyword = st.text_input("Masukkan keyword (gunakan AND, OR, NOT):", value="")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Tanggal mulai")
        with col2:
            end_date = st.date_input("Tanggal akhir")

        method = st.radio("Metode Scraping:", ["BeautifulSoup", "Selenium"], horizontal=True)
        nlp_options = st.multiselect(
            "Pilih fitur NLP yang ingin dijalankan:",
            ["Article Content", "Summary", "Sentiment", "Keywords", "Author"]
        )
        run_nlp_initial = len(nlp_options) > 0
        
        submitted = st.form_submit_button("Mulai Scrape")

    if submitted:
        with st.spinner("Sedang scraping berita..."):
            results = get_news_data(method, start_date, end_date, keyword)
            df = pd.DataFrame(results)

            if df.empty:
                st.warning("Tidak ada hasil ditemukan.")
            else:
                # Store result in session state
                st.session_state.scraped_df = df
                st.session_state.filename = f"hasil_berita_{start_date}_to_{end_date}.xlsx"
                st.session_state.nlp_done = False

                if run_nlp_initial:
                    # with st.spinner("Menjalankan NLP..."):
                        st.session_state.scraped_df = enrich_with_nlp(df, selected_nlp=nlp_options)
                        st.session_state.nlp_done = True
                        st.success("NLP selesai!")

    # ✅ Show results if we have any in session state
    if "scraped_df" in st.session_state:
        df = st.session_state.scraped_df
        st.success(f"{len(df)} berita berhasil di-scrape!")
        st.dataframe(df)

        # # ⬇️ Option to run NLP later, if not already done -> NEED TO FIX FOR VER 2
        # if not st.session_state.get("nlp_done", False):
        #     if st.button("🔁 Jalankan NLP Sekarang"):
        #         with st.spinner("Menjalankan NLP..."):
        #             st.session_state.scraped_df = enrich_with_nlp(df)
        #             st.session_state.nlp_done = True
        #             st.success("NLP selesai!")
        #             st.dataframe(st.session_state.scraped_df)

        # Download section
        filename = st.session_state.filename
        st.session_state.scraped_df.to_excel(filename, index=False)
        # Download and clear session state
        with open(filename, "rb") as f:
            if st.download_button("📥 Download Excel", f, file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"):
                # Clear session state after download
                st.session_state.pop("scraped_df", None)
                st.session_state.pop("filename", None)
                st.session_state.pop("nlp_done", None)
                st.rerun()  # Force refresh so data disappears

# NEW - queue system by Naomi 24/04
elif menu == "Queue":
    st.title("🗓️ Multiple Keyword Scraper")
    st.markdown("Scrape berita berdasarkan beberapa keyword sekaligus. Hasil file excel akan didownload secara otomatis.")
    if "query_queue" not in st.session_state:
        st.session_state.query_queue = []

    st.subheader("➕ Tambah Keyword ke Antrian")
    with st.form("add_queue_form"):
        keyword = st.text_input("Masukkan keyword (gunakan AND, OR, NOT):")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Tanggal mulai")
        with col2:
            end_date = st.date_input("Tanggal akhir")
        method = st.radio("Metode Scraping:", ["BeautifulSoup", "Selenium"], horizontal=True)
        nlp_options = st.multiselect(
            "Pilih fitur NLP yang ingin dijalankan:",
            ["Article Content", "Summary", "Sentiment", "Keywords", "Author"]
        )
        run_nlp_queue = len(nlp_options) > 0
        add_button = st.form_submit_button("Tambahkan ke Antrian")

        if add_button:
            st.session_state.query_queue.append({
                "keyword": keyword,
                "start_date": start_date,
                "end_date": end_date,
                "method": method,
                "nlp_options": nlp_options,  # Store NLP options
                "has_nlp": run_nlp_queue
            })
            st.success("✅ Keyword ditambahkan ke antrian.")


    st.subheader("🧾 Antrian Aktif")

    if st.session_state.query_queue:

        for i, item in enumerate(st.session_state.query_queue):
            col1, col2 = st.columns([11, 1], vertical_alignment="center")
            with col1:
                # Display NLP options if any were selected
                nlp_display = ""
                if item.get('has_nlp', False):
                    nlp_display = f" dengan NLP: {', '.join(item['nlp_options'])}"
                
                st.markdown(f"**{i+1}.** `{item['keyword']}` dari `{item['start_date']}` ke `{item['end_date']}` dengan metode `{item['method']}`{nlp_display}")
            with col2:
                if st.button("❌", key=f"delete_{i}"):
                    st.session_state.query_queue.pop(i)
                    st.rerun()

    else:
        st.info("Tidak ada query dalam antrian.")

    # === Processing Button ===
    if st.session_state.query_queue and st.button("🚀 Proses Semua Antrian", use_container_width=True):
        st.subheader("📤 Hasil Proses Scraping")
        log_container = st.container()

        # Desktop/output folder path
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        output_folder = os.path.join(desktop_path, "output")
        os.makedirs(output_folder, exist_ok=True)

        for idx, item in enumerate(st.session_state.query_queue):
            with log_container:
                st.markdown(f"### 🔄 Query #{idx+1}: `{item['keyword']}`")
                with st.spinner("Sedang scraping..."):
                    result = get_news_data(
                        item["method"], item["start_date"], item["end_date"], item["keyword"]
                    )

                if not result.empty:
                    # Optional NLP enrichment (if checkbox was selected earlier)
                    if run_nlp_queue:
                        with st.spinner("Menjalankan NLP..."):
                            result = enrich_with_nlp(result, selected_nlp=nlp_options)
                            st.success("NLP selesai!")

                    safe_keyword = re.sub(r"[^\w\s-]", "", item["keyword"]).replace(" ", "_")
                    filename = f"berita_{safe_keyword}_{item['start_date']}_{item['end_date']}.xlsx"
                    file_path = os.path.join(output_folder, filename)
                    result.to_excel(file_path, index=False)

                    st.success(f"✅ {len(result)} berita berhasil disimpan untuk `{item['keyword']}`")
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label=f"📥 Download hasil: '{item['keyword']}'",
                            data=f,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    st.caption(f"📂 File disimpan di: `{file_path}`")
                else:
                    st.warning(f"⚠️ Tidak ada hasil ditemukan untuk `{item['keyword']}`")

        # Refresh UI secara manual
        st.session_state.query_queue.clear()
        st.success("🎉 Semua antrian telah diproses.")
        if st.button("🔄 Clear Antrian", use_container_width=True):
            st.rerun()



elif menu == "Scheduler":
    st.title("🗓️ Jadwal Scraping Otomatis")
    st.markdown("🚧 Fitur ini masih dalam proses implementasi.")
    schedules = load_schedules()

    with st.expander("➕ Tambah atau Ubah Jadwal"):
        with st.form("schedule_form"):
            query = st.text_input("Keyword Boolean")
            mode = st.selectbox("Pilih Mode Waktu", ["1 hari lalu", "Seminggu lalu", "Sebulan lalu", "Pilih tanggal"])
            
            col1, col2 = st.columns(2)
            custom_start, custom_end = None, None
            if mode == "Pilih tanggal":
                with col1:
                    custom_start = st.date_input("Tanggal mulai")
                with col2:
                    custom_end = st.date_input("Tanggal akhir")

            freq = st.selectbox("Frekuensi Scraping", ["Setiap hari", "Setiap minggu"])
            if freq == "Setiap hari":
                waktu = st.time_input("Jam scraping")
                hari = None
            else:
                hari = st.selectbox("Hari dalam minggu", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"])
                waktu = st.time_input("Jam scraping")

            save_button = st.form_submit_button("Simpan Jadwal")

        if save_button:
            new_schedule = {
                "query": query,
                "mode": mode,
                "start": custom_start,
                "end": custom_end,
                "frekuensi": freq,
                "hari": hari,
                "waktu": waktu.strftime("%H:%M"),
            }
            schedules.append(new_schedule)
            save_schedules(schedules)
            st.success("Jadwal berhasil disimpan!")

    st.subheader("📋 Daftar Jadwal Aktif")
    if schedules:
        for i, sched in enumerate(schedules):
            st.markdown(f"**{i+1}. {sched['query']}**")
            st.markdown(f"- Mode Waktu: {sched['mode']}")
            if sched['mode'] == "Pilih tanggal":
                st.markdown(f"- Dari: {sched['start']} s.d. {sched['end']}")
            if sched['frekuensi'] == "Setiap hari":
                st.markdown(f"- Frekuensi: Harian jam {sched['waktu']}")
            else:
                st.markdown(f"- Frekuensi: {sched['hari']} jam {sched['waktu']}")
    else:
        st.info("Belum ada jadwal scraping yang ditambahkan.")


elif menu == "How to use":
    st.title("📖 How to Use")
    st.markdown("""
### Petunjuk Penggunaan

1. Masukkan **keyword pencarian** menggunakan format Boolean (misal: `"kebijakan" AND "pemerintah" NOT "ekonomi"`).
2. Pilih **tanggal mulai dan akhir** berita yang ingin diambil.
3. Pilih metode scraping:
   - **BeautifulSoup**: tanpa browser, lebih cepat, tapi tidak bisa render halaman dinamis.
   - **Selenium**: menggunakan browser headless, cocok untuk halaman dinamis.
4. Cek kotak "Jalankan NLP" apabila ingin mengekstrak rangkuman, sentimen, keyword dan author. Proses ini akan memakan waktu cukup lama tergantung dari jumlah artikel yang berhasil diekstrak.
5. Klik **Mulai Scrape**, tunggu hingga proses selesai.
6. Jika berhasil, hasil scraping bisa langsung diunduh dalam format **Excel**.

Tips:
- Gunakan tanda kutip `"` untuk frase.
- Gunakan `()` untuk mengelompokkan logika query.
""")

elif menu == "About":
    st.title("ℹ️ About")
    st.markdown("""
### Burson News Scraper v0.0.2

**Release Note:**
- ✅ Fix bug error
- ✅ Fix Boolean search error
- ✅ Auto-randomize delay scraping
- ✅ Auto scrape seluruh halaman
- ✅ Fix error pada format tanggal/waktu
- ✅ New side menu
                
**Release Note v1.0.1**
- ✅ Join ke database media dari google sheets
- ✅ New queue menu & system

**Release Note v1.0.2**
- ✅ Full article text extraction
- ✅ Penggunaan model sentiment & summarizer dari HuggingFace untuk klasifikasi
- ✅ Penggunaan Yake untuk ekstraksi keywords                     
---

**Made by**: Jay and Naomi ✨
""")
