# Enhancing-LLMs-with-Ontologies
_Project 'Enhancing Large Language Models Using Ontologies'_

## Введение
Ответы на фактологические вопросы – одна из современных задач LLM. Один из способов улучшения качества – интеграция информации из графов знаний [Pan et al., 2023]. Последние исследования показали, что благодаря этому подходу LLM значительно лучше справляются со многими задачами [Zhang et al., 2020; Salnikov et al., 2023]. В рамках данного проекта мы поставили себе задачу проверить, можно ли добиться сравнимого успеха при использовании не графов знаний, а онтологий – более простых и ёмких структур организации знаний. Мы сравниваем два типа онтологий между собой: (1) составленную людьми DBPedia Ontology -- "внешнюю" онтологию и (2) выученную LLM во время обучения -- "внутреннюю" онтологию.
## Обзор литературы
[Презентация](https://docs.google.com/presentation/d/1Ln2Prj3NnfQec3x9LGArzj4HZPUxlG5IvLKbyh0-QGQ/edit?usp=sharing)
## Методология
Онтологии отличаются от графов знаний тем, что в них присутствует лишь один тип связи: “является подтипом”. Кроме того, в онтологию обычно не входят конкретные сущности, а лишь классы. Поэтому связей в онтологии значительно меньше, как и включаемой информации, при этом онтологии проще создавать и модифицировать. Поэтому мы ставим себе задачу проверить, будут ли они полезны для LLM в задачах ответа на вопросы, даже при значительно меньшей информативности по сравнению с графами знаний.<br>
За основу экспериментов мы взяли статью [Salnikov et al., 2023]: отсюда мы вдохновляемся общим пайплайном. Однако есть существенное различие в работе с графами знаний и онтологиями: в графах знаний есть конкретные сущности, что позволяет авторам извлекать необходимую информацию, основываясь на сущностях в вопросе и предлагаемых ответах модели. В случае с онтологиями это бы не сработало, так как приводятся только классы, поэтому мы пришли к выводу, что самым экономным решением было бы сведение задачи извлечения информации из онтологий к задаче информационного поиска по корпусу путем перевода онтологической информации в текстовый формат. Далее алгоритм был реализован с помощью векторной базы данных ChromaDB и встроенной модели all-MiniLM-L6-v2. Кроме того, мы сравнили использование информации из составленной людьми онтологии и информации из онтологии, усвоенной самой LLM во время обучения и получаемой с помощью промптинга.<br>
В качестве бейзлайна мы решили брать первый сгененрированный ответ модели.<br>
В качестве LLM использовалась gpt-3.5-turbo от OpenAI.
## Данные
### DBPedia Ontology
Основу онтологии нашего проекта составляли данные из проекта [DBPedia](https://www.dbpedia.org/), а именно - иерархия классов. 
Классы были представлены в виде вложенного словаря. Затем на основе каждой пары (родитель, ребёнок) в этом словаре была создана строка типа `Pharaoh is a subclass of Royalty`. Таким образом получился корпус, который можно было легко векторизовать.
Самый нижний уровень онтологии, где находятся конкретные сущности, представлял из себя более семи миллионов вхождений. Векторизация этой части корпуса оказалась для нас вычислительно невозможна, но в будущих исследованиях она мог бы быть обработана аналогичным образом.
### Данные для обучения моделей и оценки
Для обучения и оценки качества модели использовался датасет [Mintaka](https://github.com/amazon-science/mintaka). Из него мы взяли вопросы, ответ на которые фомулировался в виде сущности (удалены вопросы на количество, порядок, да/нет вопросы). Мы использовали 1000 вопросов для создания обучающего сета и 250 вопросов для тестового.
## Сетап эксперимента
Наш пайплайн выглядит следующим образом:
![Pipeline.png](https://github.com/PhilBurub/Enhancing-LLMs-with-Ontologies/blob/main/Pipeline.png)
1. LLM генерирует 5 вариантов (кандидатов) ответов на вопрос
2. Кандидаты снабжаются онтологической информацией методом (1) ИЛИ (2)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a. из онтологии, представленной в текстовом виде с помощью векторного поиска.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Запрос к базе данных с онтологиями в виде *“Question: #текст_вопроса. Answer: #вариант_ответа*”<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b. путем обращения к LLM.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Промпт *"You need to generate a hypernym for answer in question-answer pair below and output a string '<Hypernym of answer\> is a subclass of <upper-level hypernym\>'. A hypernym can be a denomination of people, locations, characters, buildings, movies etc. Do not give any additional information, facts and thoughts, answer as short as possible.<br>
Question: Who is the oldest person to ever win an Academy Award in any category?<br>
Answer: James Ivory<br>
Output: Film director is a subclass of artists<br>
Question: #текст_вопроса<br>
Answer: #вариант_ответа<br>
Output:"*

4. Вопрос + кандидат + онтологическая информация, полученная методом (1) ИЛИ (2), оцениваются соответствующим sequence ranking’ом
5. Получается top-1 ответ
Sequence ranking – отдельные модели (для каждого из типа онтологий), оценивающая вероятность правильного ответа.

## Обучение модели sequence ranking
### Датасеты для обучения

На каждый из вопросов (1000 train, 250 test) мы с помощью модели GPT-3.5 сгенерировали по 5 вариантов ответа, исключили дублирующиеся среди кандидатов ответы.
#Настя и #Альберт

### Основная модель


### Гиперпараметры
Обе модели обучались с одинаковыми гиперпараметрами для сравнимости:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 8
Обучение занимало от 1 до 1,5 часов на 1 GPU в Google Collab.

### Обученные модели

1. a. [Sequence ranker для DBPedia онтологии на huggingface](https://huggingface.co/IvAnastasia/sequence-ranker-for-dbpedia-ontology)<br>
Статистика обучения:<br>
`Loss 1.063`<br>
`F1 0.34127`<br> 
`Precision 0.2765`<br> 
`Recall 0.4456`<br> 
`Accuracy 0.7017`<br> 

b. [Sequence ranker для онтологии из самой LLM на huggingface](https://huggingface.co/bert-base/sequence-ranker-for-llm-ontology)<br>
Статистика обучения:<br> 
`Loss: 0.9288`<br> 
`F1: 0.3417`<br> 
`Precision: 0.3049`<br> 
`Recall: 0.3886`<br> 
`Accuracy: 0.7403`<br> 


## Результаты


**Baseline:** accuracy по топ-1 выдаче модели, промпт `You need to answer the question below only with the name of person, location, chatacter etc. Do not give any additional information, facts and thoughts.`<br>
`Question: #текст_вопроса` <br>
`Your answer:'`<br> 

- train: `0.618`<br> 
- test: `0.564`

Подсчет accuracy для sequence ranker'а производился так:
1. Для каждого вопроса в изначальном валидационном датасете берем всех кандидатов, которые ему соответствуют;
2. Выбираем кандидата, у которого sequence ranker присвоил самый высокий логит единице (т. е. наибольшая вероятность, что этот кандидат самый правильный);
3. Считаем, для какой доли вопросов этот кандидат совпал с ground truth-лейблом.

1. a. [Sequence ranker для DBPedia онтологии на huggingface](https://huggingface.co/IvAnastasia/sequence-ranker-for-dbpedia-ontology)<br>
- train: `0.566`<br> 
- test: `0.280`

b. [Sequence ranker для онтологии из самой LLM на huggingface](https://huggingface.co/bert-base/sequence-ranker-for-llm-ontology)<br>
- train: `0.601`<br> 
- test: `0.304`
  
## Перспективы и дальнейшие исследования
- обработка онтологий, как графов, с помощью Entity Linking (см. [Cao et al. 2021])
- анализ взаимодействия онтологий и графов знаний
## Ссылки на литературу
- Salnikov, M., Le, H., Rajput, P., Nikishina, I., Braslavski, P., Malykh, V., & Panchenko, A. (2023). Large Language Models Meet Knowledge Graphs to Answer Factoid Questions. arXiv preprint arXiv:2310.02166
- Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J., & Wu, X. (2023). Unifying Large Language Models and Knowledge Graphs: A Roadmap. arXiv preprint arXiv:2306.08302
- Zhang, Zh., Liu, X., Zhang, Y., Su, Q., Sun, X., & He, B. (2020). Pretrain-kge: Learning knowledge representation from pretrained language models. In Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020, volume EMNLP 2020 of Findings of ACL, pages 259–266. Association for Computational Linguistics
- De Cao, N., Wu, L., Popat, K., Artetxe, M., Goyal, N., Plekhanov, M., Zettlemoyer, L., Cancedda, N., Riedel, S. & Petroni, F. (2021). Multilingual autoregressive entity linking. arXiv preprint arXiv:2103.12528 
## Устройство репозитория
├─ _**ontology_retrieval**_ - файлы извлечения информации из DBPedia онтологии<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ `Vectorized_Ontologies_DB.ipynb` - создание базы данных<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ `database.py` - обёртка для загрузки и взаимодействия с базой<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ `upper_ontologies_classes.txt` - онтологическая информация, представленная в текстовом виде<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ `vectors_corpora.zip` - файлы базы данных<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ `ontology_making.ipynb` - создание онтологической информации<br>
├─ `GigaChat Call.ipynb` - функции взаимодействия с GigaChat<br>
├─ _**sequence_ranking**_ - скрипты и файлы для sequence ranker моделей<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ `test_dataset_dbpedia` - папка с тестовыми данными для экспериментов с DBPedia Ontology<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ `test_dataset_llm` - папка с тестовыми данными для экспериментов sequence ranking'а с внутренней онтологией LLM<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ `train_dataset_dbpedia` - папка с данными для обучения модели sequence ranking'а с DBPedia Ontology<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ `train_dataset_llm` - папка с тестовыми данными для экспериментов sequence ranking'а с внутренней онтологией LLM<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ `train_dataset_top1.csv` - топ-1 ответ модели на вопрос из трейна<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ `test_dataset_top1.csv` - топ-1 ответ модели на вопрос из теста<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ `test_dataset_top5.csv` - топ-5 ответов модели на вопрос из теста<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ `train_dataset_top5.csv` - топ-5 ответов модели на вопрос из трейна<br>
├─ _**...**_ - ...<br>
## Состав команды
- Альберт Корнилов
- Анастасия Иванова
- Арсений Анисимов
- Мария Суворова
- Филипп Бурлаков
