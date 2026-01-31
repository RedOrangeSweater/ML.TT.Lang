# 00_Ideas

Идеи, предложения и design notes по tt-lang (не утверждённые спеки).

- [11_future_framework_vision.md](11_future_framework_vision.md) — видение фреймворка будущего (целевой UX, функциональный стиль, C++ ranges, модульность).
- [12_colored_compute_graph_and_tensix_state_machine.md](12_colored_compute_graph_and_tensix_state_machine.md) — раскрашенный граф вычислений, Tensix как «фабрики», state-машина устройств, Reader-Compute-Writer vs планировщик, multi P/C, «цветной Tetris».
- [13_architecture_impact_and_refactor_vision.md](13_architecture_impact_and_refactor_vision.md) — примеры целевого кода пользователя, архитектурные изменения, план рефакторинга (Python → C++), гибкая система с распределением по умолчанию (low latency).
- [14_implementation_roadmap.md](14_implementation_roadmap.md) — дорожная карта внедрения: program, toy examples, последовательное переписывание tt-lang (фазы 0–9+).
- [15_salabim_tensix_dataflow_simulator.md](15_salabim_tensix_dataflow_simulator.md) — симулятор Tensix/DataFlow на базе salabim (ShopsSimilarity): разноцветные вычислительные блоки и DataFlow state блоки, Tensix-cores слабо на фоне.
- [16_scheduler_visualization_and_gym.md](16_scheduler_visualization_and_gym.md) — UI визуализации планировщика (React 3D, три вида, пошаговый дебаггер) и Gymnasium environment для поиска маппинга алго-граф → HW-топология.
- [17_flexible_scheduler_framework_ux.md](17_flexible_scheduler_framework_ux.md) — гибкий фреймворк планировщика: целевой UX (алгоритмы RCW/Дейкстра/A*/MCTS/AlphaStar-like, async прогоны, пауза/перемотка, лог решений, статистика по нодам, гибкий симулятор, визуализация в стиле AAA).
- [19_toy_domain_for_scheduler_validation.md](19_toy_domain_for_scheduler_validation.md) — Toy domain для валидации и тестирования планировщика: магазины/товары/BOM/пекарни, единая абстрактная модель, генерация экземпляров с известным оптимумом, performance без железа.
