# Изгиб стальной панели в ANSYS MAPDL

Самостоятельно выбранный механический пример: полная стальная панель
1000×600×8 mm, давление 0.005 MPa вдоль +Z, две постановки опирания,
последовательность равномерных Q4 SHELL181. Геометрия и закрепления не являются
утверждённым ТЗ ЭРСО. ASMR++, inference и обучение здесь не выполняются.

Подробная постановка, определения метрик и официальные ссылки:
[ANSYS_PANEL_SPEC.md](../../docs/ANSYS_PANEL_SPEC.md).
Фактические завершённые расчёты и выводы описываются отдельно в
`docs/ANSYS_PANEL_RESULTS.md`; наличие этого кода не подтверждает успешный solve.

## Файлы

- `panel.json` — неизменная физическая постановка, сеточная серия и допуски;
- `run_panel.py` — отдельная сессия MAPDL, генерация модели, solve и экспорт;
- `analysis.py` — ряд Навье, интерполяция тензора, площадные метрики и сходимость;
- `test_analysis.py` — независимые проверки численной обработки без ANSYS;
- `runs/<run-id>/` — выбранные компактные измерения с manifest и provenance.

Полные бинарные результаты, поля `fields.npz`, команды APDL, build/solve-логи
и текстовые контрольные напряжения остаются в новом runtime под LOCALAPPDATA.
Не добавляйте runtime, окружения, лицензии и личные модели в Git.

## Существующая среда

Запускайте из корня репозитория в PowerShell. Нужны уже существующие Python
с NumPy, SciPy, Matplotlib и PyMAPDL, установленный MAPDL и доступная лицензия.
Скрипт не устанавливает пакеты. Используйте интерпретатор восстановленного
проекта подключения; его путь не зашит в расчётный код.

Перед командами задайте `PANEL_PYTHON` полным путём к существующему `python.exe`,
а `PYMAPDL_MAPDL_EXEC` — полным путём к установленному исполняемому файлу MAPDL.
Проверка переменных и реально установленной версии клиента:

```powershell
$panelPython = $env:PANEL_PYTHON
if (-not $panelPython -or -not (Test-Path -LiteralPath $panelPython)) {
    throw 'PANEL_PYTHON должен указывать на существующий python.exe.'
}
if (-not $env:PYMAPDL_MAPDL_EXEC -or -not (Test-Path -LiteralPath $env:PYMAPDL_MAPDL_EXEC)) {
    throw 'PYMAPDL_MAPDL_EXEC должен указывать на установленный MAPDL.'
}
& $panelPython -B -c 'import sys,importlib.metadata as m; print(sys.version); print(sys.executable); print("ansys-mapdl-core",m.version("ansys-mapdl-core"))'
```

Эта команда читает версию клиента; она не запускает ANSYS.
Отдельные численные проверки обработки:

```powershell
& $panelPython -B -m unittest discover -s experiments/ansys_panel -p test_analysis.py -v
```

## Запуск в два этапа

Каждое значение `--output` должно обозначать новый каталог. При существующем
каталоге скрипт прекращает работу и не перезаписывает результаты. `--runtime-root`
может существовать: внутри создаётся новый подкаталог с временной меткой UTC.
Следующие команды действительно запускают ANSYS и расчёты.

Сначала четыре линейных solve: P0/P1 на h=50 и h=25 mm.

```powershell
$panelRunId = Get-Date -Format 'yyyyMMdd-HHmmss'
$panelRuntimeRoot = Join-Path $env:LOCALAPPDATA 'pymapdl_panel_runs'
$panelCoarse = Join-Path '.local' ('ansys-panel-coarse-' + $panelRunId)
& $panelPython -B experiments/ansys_panel/run_panel.py `
    --exec-file $env:PYMAPDL_MAPDL_EXEC `
    --runtime-root $panelRuntimeRoot `
    --output $panelCoarse `
    --config experiments/ansys_panel/panel.json `
    --levels 0 1
if ($LASTEXITCODE -ne 0) { throw 'Первый этап не завершён; проверьте failure.json и runtime.' }
```

До сгущения проверьте `comparison.csv`, `measurements.json`, `manifest.json`
и полные runtime-логи. Нужны правильные размеры сеток, положительный прогиб,
сумма Rz около −3000 N, допустимый баланс, фактические DoF, конечные поля,
согласованные верхняя/нижняя поверхности и отсутствие необъяснённых механизмов.
Ошибку постановки исправляют до продолжения; затронутые расчёты повторяют
в новом каталоге.

После проверки грубого этапа продолжение использует его неизменные измерения
через `--previous`. Выходной каталог выбирается заново:

```powershell
$panelResult = Join-Path 'experiments/ansys_panel/runs' ('panel-' + $panelRunId)
& $panelPython -B experiments/ansys_panel/run_panel.py `
    --exec-file $env:PYMAPDL_MAPDL_EXEC `
    --runtime-root $panelRuntimeRoot `
    --output $panelResult `
    --config experiments/ansys_panel/panel.json `
    --previous $panelCoarse `
    --levels 2 3 4 `
    --nonlinear
if ($LASTEXITCODE -ne 0) { throw 'Продолжение не завершено; сохранённые успешные случаи и failure.json требуют проверки.' }
```

Уровни 2/3/4 соответствуют h=12.5/6.25/3.125 mm. Сгущение прекращается раньше,
если обе линейные серии получили два последовательных прохода всех критериев.
`--nonlinear` добавляет P1 с `NLGEOM,ON` на самой подробной успешно рассчитанной
линейной сетке P1, затем сохраняет его отдельно. Он не включает пластичность.
`--previous` проверяет совпадение SHA-256 конфигурации; исходный каталог не меняется.

## CLI и выходные данные

| Аргумент | Значение |
| --- | --- |
| `--exec-file` | Установленный MAPDL; по умолчанию `PYMAPDL_MAPDL_EXEC`. |
| `--runtime-root` | Обязательная родительская папка под LOCALAPPDATA для нового runtime. |
| `--output` | Обязательный новый каталог компактных результатов. |
| `--config` | JSON постановки; по умолчанию соседний `panel.json`. |
| `--levels` | Номера уровней 0–4 в порядке сгущения; по умолчанию 0 1 2 3 4. |
| `--previous` | Неизменный предыдущий этап с `manifest.json` и `measurements.json`. |
| `--nonlinear` | Дополнительно решить геометрически нелинейный P1 на самой подробной доступной сетке. |

Сессия создаётся через `launch_mapdl`, `nproc=2`, `additional_switches="-smp"`,
gRPC, с новым рабочим каталогом. В finally вызывается `exit()` только созданной
сессии; посторонние ANSYS-процессы не завершаются.

В output сохраняются:

- `manifest.json` — конфигурация, хеши, версии и происхождение;
- `session.json` — наблюдённые версия и endpoint собственной сессии;
- `navier.json` — независимая сходимость ряда Навье для P0;
- `<case>.json` — измерения каждого нового успешно принятого случая;
- `measurements.json` и `comparison.csv` — все принятые случаи, включая предыдущий этап;
- `failure.json` при отказе и `exit_warning.json` при проблеме закрытия сессии.

Частичный результат после отказа не считается завершённым экспериментом.
Runtime каждого случая сохраняется в его measurement; там находятся модель,
solve output, reactions CSV и поля. Время экспорта включает извлечение полей
и вычисление площадных метрик.

Контрольные круги радиуса 8 mm фиксированы в физических координатах.
Интерполируется узловой тензор, затем интегрируются компоненты и Мизес;
квадратура имеет собственную проверку стабилизации. Отдельно сохраняются
внутренний и глобальный максимумы восстановленного поля. Полная методика и
допуски приведены в спецификации; последняя сетка автоматически точной не считается.

Интерпретация результатов: ряд Навье — ориентир Кирхгофа–Лява только для P0;
235 MPa — контроль применимости упругости; два прохода сеточных допусков —
стабилизация метрик. Подтверждённая модель трансформаторного бака, пластичность,
устойчивость и нормативное заключение этим экспериментом не получаются.

## Независимый аудит и сохранение выбранных результатов

После завершения обоих этапов проверить сохранённые массивы и журналы без
нового запуска ANSYS. Каталог аудита и выходной каталог публикации должны
быть новыми; путь measurements ниже заменить фактическим output расчёта.

```powershell
New-Item -ItemType Directory .local/panel-audit-new
& $env:PANEL_PYTHON -B -X utf8 experiments/ansys_panel/audit_exports.py `
  --measurements .local/panel-refined-new/measurements.json `
  --output .local/panel-audit-new/audit.json
& $env:PANEL_PYTHON -B -X utf8 experiments/ansys_panel/publish_results.py `
  --input .local/panel-refined-new `
  --audit .local/panel-audit-new/audit.json `
  --output experiments/ansys_panel/runs/new-reviewed-run
```

Публикация здесь означает создание локального компактного набора для Git:
она не отправляет данные в сеть. Требуется успешный аудит с совпадающим
хешем measurements. Сохраняются CSV, JSON, цепочка исходных manifest и четыре
PNG: сходимость прогиба, контрольных напряжений, исходная сетка и деформация
с масштабом 200. Бинарные файлы решателя и полные журналы остаются в runtime.
Численные тесты: `python -B -m unittest discover -s experiments/ansys_panel -p test_analysis.py -v`.

Проверка SHA-256 готового набора без научных пакетов и без запуска ANSYS:
`python -B -I -S experiments/ansys_panel/verify_artifacts.py experiments/ansys_panel/runs/20261116_panel03`.
Ключ `--git-index` дополнительно сверяет байты staged-артефактов с диском.
`.gitattributes` отключает конверсию окончаний строк в экспериментальном дереве,
чтобы checkout сохранял записанные контрольные суммы.
