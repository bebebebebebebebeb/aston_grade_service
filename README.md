# aston_grade_service
marking grades for Moodle courses

Файлы aston_origin.xlsx и andersen_origin.xlsx - изначальные выгрузки курсов moodle из Астона и Андерсена соответственно.

Целевое поле - recommended_grade, его необходимо проставить в файле с курсами Астона. 

lightgbm.ipynb, randomforest.ipynb и XGBoost.ipynb - файлы с ML-моделями, использовались для приблизительного проставления грейда для уже имеющихся курсов

grading_app.py - приложение на Streamlit для взаимодействия с ChatGPT, потенциально используемое для проставления грейдов для новых курсов

requirements.txt прилагается (хоть и избыточный)
# python -m venv .venv
# .\.venv\Scripts\activate   
# pip install -r requirements.txt


