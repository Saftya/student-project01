Mini-RAG Workshop: NLP, Embeddings, Vector Search
==================================================

Цель воркшопа
-------------
За 2 часа студенты увидят полный retrieval pipeline:

document -> chunk -> embed -> index -> query -> retrieve -> evaluate -> rerank -> answer

Проект специально сделан без LangChain. Здесь только Python, Cohere SDK и ChromaDB, чтобы было видно, что происходит на каждом шаге.


Файлы
-----
main.py
    Готовый demo-проект Mini-RAG по текстам про NLP, embeddings, transformers и vector databases.

demo_ui.py
    Интерактивная RAG Lab для преподавательского demo: chunking, search modes, evaluation, embedding map, failure cases.

rag_core.py
    Общая логика retrieval pipeline, которую используют CLI demo, UI demo и teacher solution.

assignment.py
    Задание для самостоятельной работы. Данные другие: space exploration. Шаги похожи, но код нужно написать самостоятельно.

data_demo/
    Данные для live demo.

data_assignment/
    Данные для самостоятельного задания.

ground_truth_demo.json
    Вопросы и правильные chunk ids для demo evaluation.

ground_truth_assignment.json
    Вопросы и правильные chunk ids для assignment evaluation.


Запуск через Docker
-------------------
1. Создайте файл .env рядом с docker-compose.yml:

   COHERE_API_KEY=your_real_key_here

Быстрый вариант, если преподаватель дал Docker image tar
-------------------------------------------------------
Если рядом с архивом есть файл mini_rag_workshop_image.tar, сначала загрузите image:

   docker load -i mini_rag_workshop_image.tar

После этого запускайте demo/student через up --no-build. Это быстрее, чем собирать зависимости на каждом ноутбуке.


Вариант A: преподавательский Streamlit demo
-------------------------------------------
Этот контейнер сам запускает UI.

1. Соберите demo-контейнер:

   docker-compose -f docker-compose.demo.yml build

   или:

   docker compose -f docker-compose.demo.yml build

2. Запустите demo:

   docker-compose -f docker-compose.demo.yml up -d

   или:

   docker compose -f docker-compose.demo.yml up -d

   Если вы загрузили готовый image через docker load:

   docker compose -f docker-compose.demo.yml up -d --no-build

3. Откройте в браузере:

   http://localhost:8501

4. Если нужно запустить CLI demo внутри demo-контейнера:

   docker-compose -f docker-compose.demo.yml exec demo python main.py

   или:

   docker compose -f docker-compose.demo.yml exec demo python main.py


Вариант B: студенческая CLI-работа
----------------------------------
Этот контейнер ничего сам не запускает. Он просто держит одинаковую Python-среду для работы студентов.

1. Соберите student-контейнер:

   docker-compose -f docker-compose.student.yml build

   или:

   docker compose -f docker-compose.student.yml build

   Короткий вариант тоже работает, потому что docker-compose.yml совпадает со student-настройкой:

   docker-compose build

2. Запустите student-контейнер:

   docker-compose -f docker-compose.student.yml up -d

   или:

   docker compose -f docker-compose.student.yml up -d

   Короткий вариант:

   docker-compose up -d

   Если вы загрузили готовый image через docker load:

   docker compose -f docker-compose.student.yml up -d --no-build

3. Зайдите внутрь контейнера:

   docker-compose -f docker-compose.student.yml exec workshop bash

   или:

   docker compose -f docker-compose.student.yml exec workshop bash

4. Запустите CLI demo:

   python main.py

   По умолчанию тяжелый ablation в CLI demo выключен, чтобы не упереться в лимит trial Cohere key.
   Если хотите запустить ablation из CLI:

   RUN_ABLATION=1 python main.py

5. Откройте assignment.py и выполняйте TASK по порядку:

   python assignment.py


План на 2 часа
--------------
0-10 мин
    Docker setup, .env.
    Преподаватель запускает docker-compose.demo.yml, студенты запускают docker-compose.student.yml.

10-35 мин
    Live walkthrough по main.py: загрузка документов, chunking, embeddings, ChromaDB index.

35-55 мин
    Открыть demo_ui.py как RAG Lab:
    - Chunk Lab: показать, как chunk_size и overlap меняют документы.
    - Search Arena: сравнить Vector, Rerank, BM25, Hybrid RRF.
    - Embedding Map: показать идею "similar meaning nearby".
    - Evaluation: recall@1/3/5 и MRR до generation.
    - Failure Lab: показать, как плохой retrieval ломает answer.

55-95 мин
    Самостоятельная работа в assignment.py.

95-110 мин
    Ablation: chunk_size 300/500/1000, сравнение recall@3.

110-120 мин
    Wrap-up: почему плохой retrieval приводит к hallucination, зачем rerank, где здесь RAG.
    Каждый студент должен назвать один failure case и объяснить, что именно сломалось.


Ablation table
--------------
Заполните после эксперимента:

| Chunk size | Overlap | Чанков | Baseline acc | +Rerank acc | Наблюдение |
|------------|---------|--------|--------------|-------------|------------|
| 300        | 50      |        |              |             |            |
| 500        | 50      |        |              |             |            |
| 1000       | 50      |        |              |             |            |


Checkpoint outputs
------------------
После запуска main.py вы должны увидеть примерно:

- количество загруженных статей
- количество чанков
- sample semantic search ids
- recall@3 score
- reranked ids
- короткий RAG answer
- результаты ablation по разным chunk sizes

В demo_ui.py попробуйте:

- один query при chunk_size 300, 500, 1000
- один query в режимах Vector, Rerank, BM25, Hybrid RRF
- вопрос из ground truth и проверьте, найден ли correct chunk
- tricky query из Failure Lab
- RAG Playground с удалением одного source chunk из context

Точные числа могут отличаться, потому что embedding и rerank модели обновляются со временем.

Используемые Cohere модели:

- embeddings: embed-english-v3.0
- rerank: rerank-v3.5
- answer generation: command-r7b-12-2024


Если что-то не работает
-----------------------
Проверьте:

1. Нужный контейнер был запущен:
   - demo: docker-compose -f docker-compose.demo.yml up -d
   - student: docker-compose -f docker-compose.student.yml up -d
2. Вы находитесь внутри контейнера.
3. В .env есть COHERE_API_KEY.
4. В ключе нет кавычек и лишних пробелов.
5. В main.py и assignment.py не используется LangChain.
