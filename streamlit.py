import os
import atexit
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st



# Налаштування сторінки (це повинно бути першою командою Streamlit)
st.set_page_config(
    page_title="Розпізнавання акторів у відео",
    page_icon="🎬",
    layout="wide"
)

from streamlit_utils import *

# Стилізація (тепер після налаштування сторінки)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1565C0;
    }
    .info-box {
        background-color: #E3F2FD;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #E8F5E9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
    }
    .error-box {
        background-color: #FFEBEE;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
    }
    /* Налаштування розміру відео та зображень */
    .video-container {
        max-height: 400px;
        margin: 0 auto;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .video-container video {
        max-height: 400px;
        width: auto;
    }
    .image-container {
        max-height: 500px;
        margin: 0 auto;
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    .image-container img {
        max-height: 500px;
        width: auto;
        margin: 0 auto;
    }
    /* Центрування елементів */
    .stVideo {
        text-align: center;
    }
    div[data-testid="stImage"] {
        text-align: center;
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок додатку
st.markdown("<h1 class='main-header'>Розпізнавання акторів у відео</h1>", unsafe_allow_html=True)

# Ініціалізація змінних сесії
if 'video_file' not in st.session_state:
    st.session_state.video_file = None
if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = None
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'frame' not in st.session_state:
    st.session_state.frame = None
if 'frame_image' not in st.session_state:
    st.session_state.frame_image = None
if 'recognition_results' not in st.session_state:
    st.session_state.recognition_results = None
if 'detector_loaded' not in st.session_state:
    st.session_state.detector_loaded = False
if 'current_timestamp' not in st.session_state:
    st.session_state.current_timestamp = 0.0
if 'slider_key' not in st.session_state:
    st.session_state.slider_key = 0
if 'stats_ready' not in st.session_state:
    st.session_state.stats_ready = False



# Бічна панель з іменами усіх акторів та акторок
st.sidebar.markdown("## Актори і Акторки")
search = st.sidebar.text_input("Введіть ім'я або прізвище")

ukr_names = list(UKR_ACTOR_NAMES.values())
filtered = [name for name in ukr_names if search.lower() in name.lower()]

for name in filtered:
    st.sidebar.markdown(f"• {name}")



# Завантаження відеофайлу
st.markdown("<h2 class='sub-header'>Завантажте відео</h2>", unsafe_allow_html=True)

with st.markdown("<div class='info-box'>", unsafe_allow_html=True):
    st.markdown("""
    - Максимальний розмір файлу: 50 МБ
    - Підтримувані формати: MP4, AVI, MOV
    - Рекомендована якість: HD або Full HD
    """)
st.markdown("</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Оберіть відеофайл", type=["mp4", "avi", "mov"],
                                 help="Максимальний розмір: 50 МБ")

if uploaded_file is None and st.session_state.video_file is not None:
    reset_state()

# Перевірка розміру файлу
if uploaded_file is not None:
    # Перевірка розміру файлу (максимум 200 МБ)
    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > 50:
        st.error(f"Розмір файлу ({file_size_mb:.1f} МБ) перевищує максимально допустимий розмір 50 МБ.")
        uploaded_file = None
    else:
        # Якщо файл змінився, оновлюємо змінні сесії
        if st.session_state.video_file != uploaded_file.name:
            # Закриваємо попередній об'єкт відеозахоплення, якщо він існує
            if st.session_state.cap is not None:
                st.session_state.cap.release()

            # Очищаємо попередній тимчасовий файл, якщо він існує
            if st.session_state.temp_file_path is not None and os.path.exists(st.session_state.temp_file_path):
                os.remove(st.session_state.temp_file_path)

            st.session_state.stats_ready = False
            st.session_state.final_stats = None

            # Створюємо тимчасовий файл для збереження відео
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            st.session_state.temp_file_path = tfile.name
            tfile.close()

            # Ініціалізуємо відеозахоплення
            st.session_state.cap = cv2.VideoCapture(st.session_state.temp_file_path)
            st.session_state.video_file = uploaded_file.name

            # Скидаємо попередні результати
            st.session_state.frame = None
            st.session_state.frame_image = None
            st.session_state.recognition_results = None
            st.session_state.current_timestamp = 0.0

if st.session_state.cap is not None:
    # Отримання метаданих відео
    fps = st.session_state.cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    width = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Відображення метаданих
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"Тривалість: {int(duration // 60)}:{int(duration % 60):02d}")
    with col2:
        st.info(f"Частота кадрів: {fps:.2f} FPS")
    with col3:
        st.info(f"Роздільна здатність: {width}x{height}")

    st.markdown("<h2 class='sub-header'>Статистика завантаженого відео</h2>", unsafe_allow_html=True)
    generate_statistics_once()

    if st.session_state.stats_ready:
        stats = st.session_state.final_stats
        labels = [item['identity'] for item in stats]
        counts = [item['count'] for item in stats]
        total = sum(counts)
        percentages = [count / total * 100 for count in counts]

        total_people = len(stats)
        unknowns = [p for p in stats if p["identity"].startswith("Невідома особа")]
        knowns = [p for p in stats if not p["identity"].startswith("Невідома особа")]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"💫 Унікальних людей: **{total_people}**")
        with col2:
            st.info(f"🧩 Розпізнані актори: **{len(knowns)}**")
        with col3:
            st.info(f"❓ Невідомі особи: **{len(unknowns)}**")

        fig, ax = plt.subplots(figsize=(7, max(4, len(labels) * 0.4)))
        bars = ax.barh(labels, percentages, color=plt.cm.tab20.colors[:len(labels)])
        ax.set_title("Відсоток появ облич у відео", fontsize=14)
        ax.set_xlabel("Відсоток кадрів")
        ax.set_ylabel("Особи")

        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va='center', fontsize=9)

        ax.set_xlim(0, max(percentages) * 1.2)
        ax.invert_yaxis()
        st.pyplot(fig)


        st.markdown("</div>", unsafe_allow_html=True)

        # Відображення відео
        st.markdown("<h2 class='sub-header'>Перегляньте вміст відео</h2>", unsafe_allow_html=True)

        # Використання HTML5 video player для відображення відео у контейнері з обмеженим розміром
        video_bytes = open(st.session_state.temp_file_path, 'rb').read()
        st.markdown("<div class='video-container'>", unsafe_allow_html=True)
        # Використання стовпців для центрування відео
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.video(video_bytes)
        st.markdown("</div>", unsafe_allow_html=True)

        # Опція для захоплення поточного кадру
        st.markdown("<h2 class='sub-header'>Виберіть момент часу для кадру</h2>", unsafe_allow_html=True)

        # Створюємо колонки для кнопок та слайдера
        cl1, cl2, cl3 = st.columns([1, 8, 1])
        with cl1:
            st.button("◀", on_click=decrease_time, help="Зменшити часовий код на 0.5 секунди")

        with cl2:
            # Динамічно створюємо слайдер з унікальним ключем для примусового оновлення
            current_key = f"timestamp_slider_{st.session_state.slider_key}"

            # Додаємо ключ до session_state перед використанням
            if current_key not in st.session_state:
                st.session_state[current_key] = st.session_state.current_timestamp

            timestamp = st.slider(
                "Виберіть часовий код кадру (у секундах)",
                min_value=0.0,
                max_value=float(duration),
                value=st.session_state.current_timestamp,
                step=0.01,
                key=current_key
            )

            # Оновлюємо поточний час з значення слайдера
            st.session_state.current_timestamp = timestamp

        with cl3:
            st.button("▶", on_click=increase_time, help="Збільшити часовий код на 0.5 секунди")

        # Оновлюємо кадр на основі поточного часового коду
        frame, success = get_frame_at_timestamp(st.session_state.cap, st.session_state.current_timestamp)

        if success:
            st.session_state.frame = frame
            # Конвертуємо кадр з BGR в RGB для відображення в Streamlit
            rgb_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            st.session_state.frame_image = Image.fromarray(rgb_frame)
            # Очищаємо попередні результати
            st.session_state.recognition_results = None
        else:
            st.error("Не вдалося зчитати кадр. Спробуйте інший часовий код.")

        # Відображаємо захоплений кадр
        if st.session_state.frame_image is not None:
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            # Використання стовпців для центрування зображення
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(st.session_state.frame_image, caption=f"Кадр на {st.session_state.current_timestamp:.2f} секунд",
                         use_container_width=False)
            st.markdown("</div>", unsafe_allow_html=True)

        # Розпізнавання обличчя на кадрі
        if st.session_state.frame_image is not None:
            st.markdown("<h2 class='sub-header'>Розпізнайте актора на кадрі</h2>", unsafe_allow_html=True)

            if st.button("🔍 Розпізнати", help="Розпізнати обличчя на кадрі"):
                with st.spinner("🕵️‍♀️ Розпізнавання обличчя..."):
                    frame = st.session_state.frame
                    # Конвертуємо в RGB і переконуємося, що тип даних uint8
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame = ensure_uint8(rgb_frame)

                    try:
                        # Використовуємо функцію recognize_actors для отримання результатів
                        # Тепер вона повертає (face_img, actor_name, similarity, facial_area)
                        actor_results = recognize_actors(rgb_frame)

                        # Якщо функція не викинула помилку, але не повернула результати
                        if not actor_results:
                            st.warning("⚠️ На цьому кадрі не знайдено жодного обличчя. Спробуйте інший момент відео.")
                            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.image(rgb_frame, caption="Кадр без облич", use_container_width=False)
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            # Відображаємо результати розпізнавання
                            st.success(f"✅ Знайдено {len(actor_results)} облич{'чя' if len(actor_results) == 1 else ''}!")

                            # Відображаємо кадр з прямокутниками облич та підписами
                            display_frame_boxes = rgb_frame.copy()

                            # Для кожного обличчя відображаємо прямокутник та підпис
                            for i, (face_img, actor_name, similarity, facial_area) in enumerate(actor_results):
                                # Отримуємо координати з facial_area
                                x, y, w_f, h_f = (
                                    facial_area.get('x', 0),
                                    facial_area.get('y', 0),
                                    facial_area.get('w', 0),
                                    facial_area.get('h', 0)
                                )

                                # Малюємо прямокутник навколо обличчя
                                brush_stroke_frame(display_frame_boxes, x, y, w_f, h_f)

                            # Відображаємо загальний кадр з розпізнаними обличчями
                            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.image(display_frame_boxes, caption="Розпізнані обличчя", use_container_width=False)
                            st.markdown("</div>", unsafe_allow_html=True)

                            # Відображаємо кожне обличчя окремо з результатами розпізнавання
                            st.markdown("<h3 class='sub-header'>Результати розпізнавання:</h3>", unsafe_allow_html=True)

                            # Створюємо стільки колонок, скільки облич знайдено (максимум 3 в ряд)
                            faces_per_row = min(3, len(actor_results))

                            for i in range(0, len(actor_results), faces_per_row):
                                cols = st.columns(faces_per_row)
                                for j in range(faces_per_row):
                                    if i + j < len(actor_results):
                                        face_img, actor_name, similarity, _ = actor_results[i + j]

                                        # Переконуємося, що обличчя в правильному форматі
                                        face_img = ensure_uint8(face_img)

                                        # Перевіряємо колірний формат і перетворюємо при необхідності
                                        if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                                            # Якщо це BGR (з OpenCV), перетворюємо в RGB для відображення в Streamlit
                                            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                            face_img = cv2.resize(face_img, (200, 200))

                                        with cols[j]:
                                            st.image(face_img, caption=f"Обличчя #{i + j + 1}", width=200)
                                            if actor_name == 'Unknown':
                                                st.markdown(f"**Актор:** Невідомо")
                                            else:
                                                actor_name = actor_name.replace("_", " ")
                                                st.markdown(f"**Актор:** {UKR_ACTOR_NAMES[actor_name]}")
                                                st.markdown(f"**Вірогідність:** {int(similarity * 100)}%")

                    except ValueError as ve:
                        if "No face detected" in str(ve):
                            st.warning("⚠️ На цьому кадрі не знайдено жодного обличчя. Спробуйте інший момент відео.")
                        else:
                            st.error(f"Помилка: {str(ve)}")
                        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.image(rgb_frame, caption="Кадр без облич", use_container_width=False)
                        st.markdown("</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Помилка при розпізнаванні акторів: {str(e)}")
                        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.image(rgb_frame, caption="Кадр з помилкою розпізнавання", use_container_width=False)
                        st.markdown("</div>", unsafe_allow_html=True)

# Інструкції для користувача
if uploaded_file is None:
    # Інструкції для користувача внизу сторінки
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("""
    ### Як користуватися додатком:
    1. Завантажте відеофайл (до 50 МБ)
    2. Ознайомтесь з статистикою розпізнаних акторів для вашого відео
    3. Перегляньте відео для знаходження потрібного моменту
    4. Використовуйте повзунок, щоб вибрати часовий код потрібного кадру (кадр відобразиться автоматично)
    5. Переконайтесь, що якість відео задовільна та на кадрі чітко видно все лице актора/акторки
    6. Натисніть кнопку "Розпізнати" для розпізнавання обличчя на кадрі

    **Примітка:** Враховуйте, що далеко не всі актори є у базі даних. Також алгоритм може помилятись,
     і інколи відображати некоректні результати. Переконайтесь, що дотримались всіх описаних вище кроків!  
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# Очищення ресурсів при закритті додатку
def cleanup():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
    if st.session_state.temp_file_path is not None and os.path.exists(st.session_state.temp_file_path):
        try:
            os.remove(st.session_state.temp_file_path)
        except:
            pass


# Реєстрація функції очищення
atexit.register(cleanup)

# Примітка внизу сторінки
st.markdown("---")
st.markdown("🎬 **Додаток для розпізнавання акторів у відео** | Автор: Ярмошенко Роман | Для критики та порад - Telegram: **@Rygml**💟")