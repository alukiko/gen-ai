# Tweet Engagement Agent

Прикладной итоговый проект: агент улучшает текст твита и время публикации, используя старую обученную модель из `Tweets/`.

## Что внутри

- `Tweets/` - исходный старый проект, датасет, `weekday_ohe.pkl`, `random_forest_model.pkl` и ноутбуки.
- `model.py` - инструмент прогноза вовлеченности: `bert-base-uncased` + сохраненный RandomForest.
- `rag.py` - поиск похожих исторических твитов по датасету.
- `agent.py` - агент: генерирует варианты, вызывает инструмент прогноза, выбирает лучший, проверяет галлюцинации.
- `schema.py` - Pydantic-схемы и `field_validator` для ограничения твита 280 символами.
- `eval.py` - прогон 15 тестов, метрики качества и пути.
- `output/` - артефакты после запуска.

## Одна команда запуска

```bash
cd финальный_проект
python eval.py
```

Ключи LLM берутся из `.env` в этой папке или, если его нет, из `../семинар_2/домашнее/.env`.
Для полностью локального smoke-run без LLM:

```bash
python eval.py --no-llm
```

Один пример:

```bash
python run.py --text "Our new product is out. Check the website. #cloud" --weekday Wednesday --hour 13
```

## Техники курса

1. RAG: `rag.py` достает похожие исторические твиты и их фактическую вовлеченность.
2. Агент с инструментами: `agent.py` вызывает `predict_engagement` для исходного текста и вариантов.
3. Structured output: LLM-ответы валидируются Pydantic-моделями, есть `field_validator`.
4. LLM-as-judge: финальный результат оценивается судьей, при недоступности LLM есть прозрачный heuristic fallback.
5. Проверка галлюцинаций: считаются ghost-числа и ghost-цитаты.

## Артефакты

После `python eval.py` создаются:

- `output/eval_results.json` - полный trace по каждому кейсу.
- `output/eval_table.csv` - таблица метрик.

