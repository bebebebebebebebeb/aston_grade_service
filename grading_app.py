import os
import json
import textwrap
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI


# UI

st.set_page_config(page_title="Course Grade Recommender", layout="wide")

st.sidebar.title("⚙️ Настройки")
model_default = "gpt-4.1-mini"  # безопасный быстрый дефолт; можно сменить в UI
model = st.sidebar.text_input("Модель OpenAI", value=model_default, help="Напр.: gpt-4.1, gpt-4o, gpt-4.1-mini, gpt-5 (если доступна в вашем аккаунте)")
shots_per_class = st.sidebar.slider("Кол-во примеров на класс (few-shot)", 1, 8, 3)
max_train_rows = st.sidebar.slider("Макс. строк из CSV для примеров (перемешанных)", 100, 5000, 1000, step=100)
strict_json = st.sidebar.checkbox("Строгий JSON-вывод (Structured Outputs)", value=True)

st.sidebar.caption("Приложение использует OpenAI **Responses API** и структурированный вывод JSON для надёжного парсинга ответа модели.")

# API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Не найден ENV-переменная OPENAI_API_KEY. Установите её перед запросом к LLM.")
client = OpenAI(api_key=OPENAI_API_KEY)  # openai>=1.0.0

# гл. колонка
st.title("🎓 Рекомендация грейда")
st.write(
    "Загрузите исходный набор данных (CSV с колонками **course_name**, **summary**, **category_name**, **course_structure**, **recommended_grade**), "
    "затем введите описание нового курса — получите `recommended_grade`"
)

uploaded = st.file_uploader("Загрузите andersen_clean_for_LLM", type=["csv"])

col1, col2 = st.columns(2)
with col1:
    course_name = st.text_input("Название курса (course_name)", "")
    category_name = st.text_input("Категория (category_name)", "")
with col2:
    summary = st.text_area("Краткое описание (summary)", "", height=120)
    course_structure = st.text_area("Структура/модульность (course_structure)", "", height=120)

run_btn = st.button("🔮 Получить рекомендуемый грейд", type="primary")

# --------------------------
# Вспомогательные функции
# --------------------------
def clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def pick_few_shots(df, label_col="recommended_grade", k=3, seed=42):
    # Гарантированно берём до k примеров на класс
    # предварительно перемешав датасет
    dfx = df.sample(frac=1.0, random_state=seed).copy()
    examples = []
    for cls, grp in dfx.groupby(label_col):
        take = grp.head(k)
        for _, r in take.iterrows():
            ex = {
                "course_name": clean_str(r.get("course_name", "")),
                "summary": clean_str(r.get("summary", "")),
                "category_name": clean_str(r.get("category_name", "")),
                "course_structure": clean_str(r.get("course_structure", "")),
                "recommended_grade": clean_str(r.get(label_col, "")),
            }
            # пропускаем пустые целевые
            if ex["recommended_grade"] == "":
                continue
            examples.append(ex)
    return examples

def build_system_prompt(allowed_labels):
    return textwrap.dedent(f"""
    Вы — ассистент по проставлению грейдов курсам для IT-специалистов, ваша задача — присвоить рекомендуемый грейд курсу.
    Отвечайте строго в JSON с полями:
      - "recommended_grade": один из {allowed_labels}. J - junor, M - middle, S - senior, All - all levels.
      - "confidence": число от 0 до 1 (оценка уверенности)
      - "explanation": кратко, 1–2 предложения, почему выбран этот грейд (без рассуждений по шагам).

    Правила:
    - Используйте примеры (few-shot) как ориентир стиля и критериев.
    - Если описание скудное, предполагайте консервативнее и снижайте "confidence".
    - Не выдумывайте новых меток.
    """).strip()

def build_user_prompt(new_item, few_shots):
    # Сформируем компактный промпт: сначала примеры, затем новый курс
    lines = []
    lines.append("Примеры (исторические записи, формат JSONLines):")
    for ex in few_shots:
        lines.append(json.dumps(ex, ensure_ascii=False))
    lines.append("\nНовый курс:")
    lines.append(json.dumps(new_item, ensure_ascii=False))
    lines.append("\nВерните только один JSON-объект с полями recommended_grade, confidence, explanation.")
    return "\n".join(lines)

def call_llm_responses_api(model, system, user, temperature=0.2, strict=True):
    if not strict:
        # Нестрогий режим: просто просим JSON, парсим на нашей стороне
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.output_text
        return text
    else:
        # Строгий JSON через Structured Outputs (json_schema)
        schema = {
            "name": "grade_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "recommended_grade": {"type": "string"},
                    "confidence": {"type": "number"},
                    "explanation": {"type": "string"}
                },
                "required": ["recommended_grade", "confidence", "explanation"],
                "additionalProperties": False
            },
            "strict": True
        }
        resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_schema", "json_schema": schema},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

def parse_llm_json(s):
    try:
        return json.loads(s)
    except Exception:
        # попытка выдернуть JSON-объект из текста
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                return None
        return None

# --------------------------
# Основная логика
# --------------------------
if run_btn:
    if uploaded is None:
        st.error("Пожалуйста, загрузите CSV с историческими данными ")
        st.stop()

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Не удалось прочитать CSV: {e}")
        st.stop()

    required_cols = ["course_name", "summary", "category_name", "course_structure", "recommended_grade"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"В CSV отсутствуют обязательные колонки: {missing}")
        st.stop()

    # Предочистка и подрезание датасета
    df = df[required_cols].copy()
    if len(df) > max_train_rows:
        df = df.sample(n=max_train_rows, random_state=42)

    # Список допустимых меток
    labels = (
        df["recommended_grade"].astype(str).fillna("").str.strip().replace({"": "nan"}).unique().tolist()
    )
    # Мини-гармонизация: убрать 'nan' разных регистраций
    labels = sorted(set(["nan" if str(x).lower() in {"nan", "none"} else str(x) for x in labels]))
    allowed_labels_str = ", ".join([f'"{x}"' for x in labels])

    # Few-shot примеры
    shots = pick_few_shots(df.assign(
        course_name=df["course_name"].map(clean_str),
        summary=df["summary"].map(clean_str),
        category_name=df["category_name"].map(clean_str),
        course_structure=df["course_structure"].map(clean_str),
        recommended_grade=df["recommended_grade"].astype(str).map(lambda x: "nan" if x.strip()=="" else x.strip())
    ), k=shots_per_class)

    if len(shots) == 0:
        st.error("Не удалось собрать few-shot примеры (возможно, пустые или некорректные метки).")
        st.stop()

    # Новый объект
    new_item = {
        "course_name": clean_str(course_name),
        "summary": clean_str(summary),
        "category_name": clean_str(category_name),
        "course_structure": clean_str(course_structure),
    }

    if all(v == "" for v in new_item.values()):
        st.error("Введите описание нового курса (минимум одно поле).")
        st.stop()

    # Построение промптов
    system_prompt = build_system_prompt(allowed_labels=allowed_labels_str)
    user_prompt = build_user_prompt(new_item=new_item, few_shots=shots)

    # Вызов LLM
    try:
        raw = call_llm_responses_api(
            model=model,
            system=system_prompt,
            user=user_prompt,
            strict=strict_json
        )
    except Exception as e:
        st.exception(e)
        st.stop()

    result = parse_llm_json(raw) if isinstance(raw, str) else raw
    st.subheader("🧠 Ответ модели")
    st.code(raw, language="json")

    if isinstance(result, dict):
        rg = result.get("recommended_grade", "").strip()
        conf = result.get("confidence", None)
        expl = result.get("explanation", "")

        # подсветим плашкой
        st.success(f"**recommended_grade:** {rg}")
        if conf is not None:
            try:
                st.progress(min(max(float(conf), 0.0), 1.0))
            except Exception:
                pass
        if expl:
            st.write(f"**Пояснение:** {expl}")

        # Сформировать строку для копирования
        st.download_button(
            "💾 Скачать JSON-результат",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name="recommended_grade.json",
            mime="application/json"
        )
    else:
        st.warning("Не удалось распарсить JSON. Проверьте блок выше — там есть сырой ответ модели.")

# --------------------------
# Нижняя справка
# --------------------------
with st.expander("ℹ️ Подсказки по качеству"):
    st.markdown("""
- Чем богаче описание курса, тем выше точность. Заполняйте все поля.
- Увеличьте `Кол-во примеров на класс` в сайдбаре, если классы несбалансированы.
- Если модель путает классы с редкими примерами, попробуйте:
  - повысить shots до 5–8;
  - сократить `Макс. строк` до более репрезентативной подвыборки или наоборот увеличить.
- Можно задать более мощную модель в поле **Модель OpenAI** (если есть доступ).
""")
