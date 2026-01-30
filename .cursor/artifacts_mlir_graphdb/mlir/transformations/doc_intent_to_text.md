# Трансформация: reu.sdlc_doc_intent → Text

**Вход**: MLIR слой `reu.sdlc_doc_intent`.  
**Выход**: литературно-технический текст (LLM generation).

Описание: финальная стадия — LLM генерирует текст по doc_intent (контракт: вход — doc_intent MLIR, выход — текст). Текст не хранится в KG.

Реестр: [../README.md](../README.md).
