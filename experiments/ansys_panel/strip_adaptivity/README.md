# P1: согласованная полосовая адаптация в MAPDL

Генератор использует тензорное произведение координатных линий X/Y.
Это полосовое сгущение через всю панель. EREFINE не вызывается.
Физика и SHELL181 совпадают с panel.json; только P1 linear.

Выполнение требует существующих NumPy/SciPy/Matplotlib/PyMAPDL и лицензии
MAPDL. Скрипт ничего не устанавливает. Задайте пути своей существующей среды:

```powershell
& $panelPython -B -X utf8 experiments/ansys_panel/strip_adaptivity/run_experiment.py --exec-file $env:PYMAPDL_MAPDL_EXEC --runtime-root .local/strip-runtime --output experiments/ansys_panel/strip_adaptivity/runs/NEW_RUN
```

Эта команда действительно выполняет solve: общий старт 240 элементов,
до трёх адаптивных шагов при cap 5382 подтверждённых активных DoF и отдельный
uniform h=25. Полный протокол записан до запуска в protocol.json.
Прогноз бюджета проверяется до solve; Number of equations проверяется после.
Сохраняются все линии closure, θ=0.5 не меняется ради бюджета.

При техническом отказе завершённые solve остаются в приватных checkpoint.
Повтор той же команды с `--resume` повторяет только необходимый POST1/обработку
и ещё не выполненные solve. Завершённый output с completed.json защищён от
повторного использования. Старые конфигурации/протокол/reference не подменяются.
При исправлении нового кода отдельно сохраняются исходные ошибки и времена;
контрольные модели и исторические серии не изменяются.

```powershell
& $panelPython -B -m unittest discover -s experiments/ansys_panel/strip_adaptivity -p "test_*.py"
& $panelPython -B -m unittest discover -s experiments/ansys_panel -p test_analysis.py
& $panelPython -B experiments/ansys_panel/strip_adaptivity/verify_run.py experiments/ansys_panel/strip_adaptivity/runs/NEW_RUN
```

verify_run.py не запускает MAPDL/solve: повторяет геометрическую проверку,
интерполяцию/квадратуру, поворот N/M/Q, recovery и маркировку по сохранённым данным.
Результанты SMISC являются несглаженными центроидными данными; их нельзя заменять
сглаженными узловыми TOP/BOTTOM. Последние используются только для сравнения
семи контрольных величин с прежним компаратором и его ограничениями.

S определяется по native секционным E и TS11/TS22/TS12: для проверенного
изотропного случая Sii=Eii·TSii. GET E описывает неисправленную жёсткость Gt,
а TS — сдвиговую коррекцию. Q/Sγ сохраняется как диагностика. Гипотеза
точного равенства независимо экспортированных центроидных Q и Sγ не прошла
на неравномерной сетке; прежняя идентификация отклонена и сохранена отдельно.
constitutive_extraction.json описывает исправленное извлечение секции.
Вся метрика остаётся recovery-эвристикой, не точной дискретной FE-нормой.

`classical_selector(observation)` возвращает η_K². Файл observation.json:
узлы графа — элементы, рёбра — общие целые стороны в обоих направлениях.
Содержит 18 признаков: геометрию, глобальные N/M/Q, три энергетических вклада
и их сумму. Физические нормировщики: a,b,ab; N,Q=|p|b; M=|p|b²;
η²=p²b⁶/D11. Reference и обученные статистики в наблюдения не входят.
Подключение ASMR++ потребует нового совместимого адаптера, reward/budget
контракта и проверки политики на механике. Обучение/inference здесь не выполняются.

В Git публикуются только переносимые JSON/CSV/PNG и исходники. DB/RST/FULL,
native журналы, пути машины и licensing diagnostics остаются в .local.
