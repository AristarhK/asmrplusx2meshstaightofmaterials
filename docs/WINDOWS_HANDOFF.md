# Передача проекта на Windows с ANSYS

Дата проверки: 2026-11-15. Приоритет — нейросетевая адаптация сетки прочностного
расчёта бака трансформатора, конкурс «Система», тема ЭРСО
«Новейшие методы расчета баков на механическую прочность».

## 1. Фактическое состояние

Рабочая копия этой задачи: `/workspace/asmrplusx2meshstaightofmaterials` (Linux),
а не прежний локальный каталог `research-hub`. После публикации пользователем
получена ветка `repro/recover-asmr-runtime`, исходный HEAD:
`c9aa797e0a1b0ec710d4c1d815d6ac8b0b7dae1e`.
Перед переходом незакоммиченных изменений не было. `main` на момент проверки —
`79336185edf8f05fd3b32ec103546f05ef516847`; в ней уже объединены исследовательские
документы, но комплект восстановления находится в отдельной ветке.

Этот handoff добавляется следующим локальным коммитом той же ветки. Его полный
SHA и состояние публикации передаются в итоговом сообщении задачи: документ
не может содержать SHA коммита, включающего самого себя. Сверяйте `git rev-parse HEAD`
с итоговым сообщением. Наличие ветки на GitHub не означает, что её последний
локальный коммит уже отправлен. Никакой force-push или merge здесь не требуется.

### Состав сохранённого комплекта

Пути в таблице относительны к `experiments/asmrpp/`.

| Часть | Файлы | Состояние |
| --- | --- | --- |
| Исходники | `reproduce/artifacts/asmrplusplus_source.tar.gz`, `reproduce/UPSTREAM_LICENSE.txt` | 209 файлов upstream `228bb6389cc8c0af9be8c94df7569fd6013a4e89`, MIT; SHA-256 архива и каждого файла сохранены. |
| Локальное исправление | `reproduce/patches/python310_collections.patch` | `collections.abc` в `util/function.py`; уже включено в архив, повторно не применять. |
| Разрешённая конфигурация | `reproduce/artifacts/config.yaml` | PPO, Poisson 2D, square_hole, 4 шага; отличается от исторического многодокументного шаблона `configs/poisson_train10.yaml`. |
| Веса и нормализаторы | `reproduce/artifacts/checkpoints/ckpt_final.pt`, `ckpt_final_norm_params.pkl`, `ckpt0000.pt`, `ckpt0000_norm_params.pkl` | Финальный checkpoint и ранний checkpoint с их собственными нормализаторами; нельзя смешивать пары. |
| Архитектура | `reproduce/artifacts/checkpoints/network_kwargs.npz` и config | Сериализованные параметры архитектуры; в config MPN, latent dimension 64, два message-passing шага, отдельные actor/critic bases. |
| Обучающий журнал | `reproduce/artifacts/scalars.yaml` | Сохранённые scalars прежнего обучения; новое обучение не выполнялось. |
| Контрольная задача | `reproduce/artifacts/expected_case0.json` | Пять состояний case 0 и хеш исходной таблицы. |
| Окружение | `reproduce/environment/{environment-cpu-local.yaml,requirements-cpu-local.txt,conda_packages.json,installed_packages.json}` | Метаданные существовавшего Linux CPU-окружения, не проверенная установка с нуля. |
| Проверка и запуск | `reproduce/reproduce.py`, `reproduce/inference_worker.py`, `reproduce/manifest.json` | 17 хешируемых артефактов; verify-only не требует научных пакетов. |
| Завершённое воспроизведение | `runs/recovered_20261115_seed123/` | 13 файлов: trace, result, invocation, log, пять mesh NPZ, PNG и три отчёта проверок. |
| История исследования | `results/`, `figures/`, `configs/`, `scripts/` | 12 исторических файлов, включая четыре CSV; сохранены без изменений. |

Все 44 файла экспериментального дерева уже включены в Git: модель не должна
оставаться только на диске исходной Ubuntu-машины. `runtime/` и `cache/` в run-
каталогах исключены из Git и воспроизводятся из комплекта.

### Что проверено в этой задаче

```bash
python3 -B experiments/asmrpp/reproduce/reproduce.py --verify-only
```

Код возврата 0, вывод:

```text
Verified 17 artifacts; upstream 228bb6389cc8c0af9be8c94df7569fd6013a4e89
```

Дополнительно проверены SHA-256 209 файлов внутри архива, обратная применимость
patch и Git blob исходной версии исправленного файла. Сохранённые `trace.csv`
и `result.json` согласованы с историческим case 0 по N, вершинам и ошибке на
всех пяти состояниях. NPZ-сетки прошли CRC-проверку архивов; отчёты integrity,
source_integrity и repeat имеют статус passed. Сами предыдущие два процесса
воспроизведения в этой задаче не повторялись.

Исторические 12 файлов побайтово совпадают с импортом `8878794`; все 44
экспериментальных файла — с исходным `c9aa797`. Проверены размеры Git blobs
во всей доступной истории: максимальный — `ckpt_final.pt`, **1 705 230 байт**.
Нет файлов более 50 MiB (предупреждение GitHub) или 100 MiB (запрет обычного Git).
Веса уже обычные Git blobs, LFS для текущего комплекта не требуется.
Проверка tracked-путей и известных сигнатур ключей/токенов не выявила секретов,
окружений, кэшей или посторонних личных файлов; это ограниченный аудит,
а не гарантия для будущих импортов. Точные пути прежней машины в metadata —
историческое происхождение, не пути запуска на Windows.

В `.gitattributes` отключено преобразование окончаний строк для всего
`experiments/asmrpp/`: иначе Windows `core.autocrlf` изменит байты YAML/Python/patch
и SHA-256 перестанут совпадать. Проверка комплекта и Git-получение с
`core.autocrlf=true` проверены на временном Linux checkout; PowerShell и нативный
Windows в этой облачной среде недоступны. Не обновляйте хеши, чтобы скрыть CRLF-
расхождение; сначала проверьте настройки checkout.

### Границы научного результата

Сохранённые отчёты подтверждают одну Poisson-траекторию: seed 123,
evaluation environment 1, λ=0.001274, CPU, один поток, 34→71→71→71→89 элементов.
Финальная legacy reference-метрика — 0.9687853785841744.
Два прежних процесса дали одинаковые сетки/ошибки и неизменную policy.
Проверка целостности сейчас не является новым inference или новым benchmark.

Не подтверждены повтор на чистом окружении, остальные девять траекторий,
точность reference, преимущество при равных DoF, ускорение и перенос на упругость.
Готовую Poisson-политику нельзя выдавать за адаптер прочностного расчёта ANSYS.
Подробности — [ASMR_REPRODUCIBILITY.md](ASMR_REPRODUCIBILITY.md).

## 2. Публикация из Ubuntu

Из корня этой рабочей копии после локального коммита:

```bash
git switch repro/recover-asmr-runtime
git status --short
git log -1 --format='%H %s'
git push origin HEAD:refs/heads/repro/recover-asmr-runtime
```

Команда отправляет только выбранную ветку и не объединяет её с main.
Если обычный push отклонён из-за новых удалённых коммитов, сначала fetch и
сравнение истории; не применять force. Проверка синхронизации после публикации:

```bash
git fetch origin refs/heads/repro/recover-asmr-runtime:refs/remotes/origin/repro/recover-asmr-runtime
git rev-list --left-right --count HEAD...origin/repro/recover-asmr-runtime
```

`0 0` означает совпадение веток. Ссылка:
https://github.com/AristarhK/asmrplusx2meshstaightofmaterials/tree/repro/recover-asmr-runtime.

## 3. Получение нужной ветки на Windows

Сначала отправить итоговый Ubuntu-коммит командой выше. Нужны Git for Windows,
PowerShell и доступ к репозиторию через штатную Git-аутентификацию. Не помещать
токен в URL или в файлы проекта. Команды Git проверены на Linux; выполнение
PowerShell и доступ к ANSYS на вашем Windows-компьютере ещё предстоят.

### Новая копия

В PowerShell из выбранной родительской папки (имя `research-hub` свободно):

```powershell
if (Test-Path -LiteralPath research-hub) { throw 'Папка research-hub уже существует; используйте обновление существующей копии ниже.' }
git clone --branch repro/recover-asmr-runtime https://github.com/AristarhK/asmrplusx2meshstaightofmaterials.git research-hub
if ($LASTEXITCODE -ne 0) { throw 'Не удалось получить ветку; проверьте публикацию и доступ к GitHub.' }
Set-Location research-hub
git branch --show-current
git rev-parse HEAD
git status --short
```

### Существующая копия

Открыть терминал именно в корне нужного репозитория. При локальных изменениях
сначала сохранить их отдельным коммитом или stash, включая нужные untracked-файлы;
не выполнять reset/clean. Проверить `git remote -v` — это должен быть наш GitHub.
Явный refspec ниже работает и в копии, где fetch настроен только для main:

```powershell
$changes = git status --porcelain
if ($LASTEXITCODE -ne 0) { throw 'Это не доступная рабочая копия Git.' }
if ($changes) { throw 'Сначала сохраните локальные изменения; переключение остановлено.' }
git fetch origin refs/heads/repro/recover-asmr-runtime:refs/remotes/origin/repro/recover-asmr-runtime
if ($LASTEXITCODE -ne 0) { throw 'Fetch не выполнен.' }
git show-ref --verify --quiet refs/heads/repro/recover-asmr-runtime
if ($LASTEXITCODE -eq 0) {
    git switch repro/recover-asmr-runtime
} else {
    git switch --no-track -c repro/recover-asmr-runtime refs/remotes/origin/repro/recover-asmr-runtime
}
if ($LASTEXITCODE -ne 0) { throw 'Не удалось переключить ветку.' }
git merge --ff-only refs/remotes/origin/repro/recover-asmr-runtime
if ($LASTEXITCODE -ne 0) { throw 'Ветки разошлись; требуется сравнение истории, без reset и force.' }
git rev-parse HEAD
git status --short
```

Сравнить SHA с итоговым сообщением. Для фиксации именно переданного коммита
использовать полный SHA из сообщения как точку отдельной рабочей ветки,
например `git switch -c windows/ansys-bootstrap <полный-SHA>`; это шаблон с
явным параметром, а не готовая команда с неизвестным SHA. Не перезаписывать
существующую одноимённую ветку.

### Проверка комплекта и открытие Codex

Если установлен Python Launcher:

```powershell
py -3 -B experiments/asmrpp/reproduce/reproduce.py --verify-only
if ($LASTEXITCODE -ne 0) { throw 'Целостность комплекта не подтверждена.' }
code .
```

При отсутствии `py` выбрать реально установленный Python 3 и вызвать тот же
скрипт через его executable. Если `code` не в PATH, открыть папку через VS Code.
Для verify-only достаточно стандартной библиотеки; переносить всё окружение
ASMR++ в Windows ради этой проверки не нужно.

## 4. Разделение работы

| Ubuntu / Codex cloud | Windows с установленным ANSYS |
| --- | --- |
| Сохранение кода/весов/config, checksum-проверка, аудит данных и разработка независимых интерфейсов. | Открытие checkout, проверка checksum и выбор Python interpreter в VS Code. |
| При наличии существующего ASMRpp_cpu — отдельно согласованный inference в новом output. | Поиск прежнего подключения, определение версии ANSYS/API, проверка лицензии и запуска. |
| Подготовка общей схемы экспорта, сравнения и методологии без запуска ANSYS. | Геометрия/материалы/нагрузки/BC, два solve на разных сетках и экспорт результатов. |

На текущей облачной машине не запускались ни ANSYS, ни новый inference.
Тяжёлые зависимости не устанавливались, модель не обучалась. В старом Linux
inference фактически NumPy **1.23.5** (`installed_packages.json`, `result.json`),
хотя YAML и conda-список заявляют **1.22.3**. Поэтому файл окружения нельзя считать
проверенным рецептом создания с нуля; существующие файлы сохранены без исправления
их истории. Native Windows-inference и установка PyG отдельно не проверены.

## 5. Поиск прежнего Windows-скрипта по содержимому

Скрипт подключения ещё не найден. Его путь, API и порт неизвестны. Ищем в
реальных пользовательских каталогах проектов, Documents/Desktop/OneDrive,
включая `.vscode` и сохранённые PowerShell-команды. Дополнить `$searchRoots`
известными папками проектов на других дисках; каталог программы ANSYS не
является заменой пользовательского проекта.

PowerShell, выводятся только пути совпавших файлов:

```powershell
$searchRoots = @(
    (Get-Location).Path
    (Join-Path $env:USERPROFILE 'Documents')
    (Join-Path $env:USERPROFILE 'Desktop')
)
if ($env:OneDrive) { $searchRoots += $env:OneDrive }
$searchRoots = $searchRoots | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -Unique
$patterns = @('ansys\.mechanical', 'ansys\.mapdl', 'launch_mechanical', 'launch_mapdl', 'ExtAPI', 'RunWB2')
if (Get-Command rg -ErrorAction SilentlyContinue) {
    rg --files-with-matches --hidden --ignore-case -g '*.py' -g '*.ipy' -g '*.ps1' -g '*.ipynb' -g '*.wbjn' -g '*.json' -g '*.code-workspace' -g '*.bat' -g '*.cmd' -g '*.txt' -g '*.md' -g '!.git/**' -g '!**/.git/**' -g '!**/.venv/**' -g '!**/venv/**' -g '!**/envs/**' -g '!**/node_modules/**' -g '!**/__pycache__/**' -- ($patterns -join '|') $searchRoots
    if ($LASTEXITCODE -gt 1) { Write-Warning 'Поиск завершился с ошибками; проверьте доступ к каталогам.' }
} else {
    Get-ChildItem -LiteralPath $searchRoots -Recurse -File -Force -ErrorAction Continue |
        Where-Object { $_.Extension -in '.py','.ipy','.ps1','.ipynb','.wbjn','.json','.code-workspace','.bat','.cmd','.txt','.md' -and $_.FullName -notmatch '[\\/](\.git|\.venv|venv|envs|node_modules|__pycache__)[\\/]' } |
        Select-String -Pattern $patterns -List |
        Select-Object -ExpandProperty Path -Unique
}
```

Отсутствие совпадений в просмотренных папках не доказывает отсутствие файла
на всех дисках. Ошибки доступа и непроверенные каталоги фиксировать отдельно.
Сначала прочитать найденные файлы, определить реальные executable/config,
интерпретатор и способ запуска, не выполнять неизвестный скрипт вслепую.
Отделить рабочий код от примеров в документации и установленных библиотек.
Секреты, лицензионные данные и личные модели не копировать в Git.
Если скрипт не найдётся, **восстановление подключения — отдельный следующий шаг**;
этот handoff не задаёт новый API и не предполагает порт по умолчанию.

## 6. Что проверить в установленном ANSYS

- Точную версию/release, компоненты Mechanical/Workbench или MAPDL и разрядность.
  ACP нужен только при постановке с композитами.
- Путь executable по реальной установке, возможность открыть проект и выполнить
  небольшой структурный solve под доступной лицензией. Наличие GUI недостаточно.
- Выбранный в VS Code Python, его версию и уже установленные `ansys-*` пакеты;
  совместимость выбранного интерфейса с этой версией ANSYS. Не ставить пакеты
  до определения фактического способа подключения.
- Фактический режим автоматизации: embedded scripting, внешний Python, journal
  или найденный launcher. Порт и session lifecycle фиксировать только по факту.
- Создание отдельной рабочей копии модели, изменение сетки, solve, чтение статуса
  сходимости и экспорт полей. Проверить корректное освобождение созданной сессии.
- Сохранение материалов, единиц, контактов, loads/BC и named selections после
  remesh, сопоставление ID узлов/элементов и locations выбранных величин.

## 7. Недостающие исходные данные механической модели

В репозитории пока нет согласованной геометрии/CAD или ANSYS-проекта бака,
размеров, толщин стенок/крышки, рёбер, швов и допущений об упрощении.
Не определены материалы и их свойства, типы/порядок элементов (shell/solid),
контакты/соединения, закрепления и опоры. Нужно задать физически обоснованные
нагрузочные случаи (например, давление/вакуум или гидростатика масла — только
после подтверждения ТЗ), сочетания и необходимость нелинейности/потери устойчивости.
Также отсутствуют критерии допустимости, нормативная база, reference-решение,
области оценки напряжений, целевая точность и вычислительный бюджет.
Не подставлять эти величины из Poisson-конфига или предположений о типовом баке.

## 8. Ближайший практический рубеж на Windows

**Первый шаг после открытия проекта в Codex:** прочитать AGENTS.md и этот
handoff, проверить ветку/SHA и выполнить verify-only; затем найти прежний
Windows-скрипт по разделу 5. Зафиксировать версию ANSYS и реальный API.

После восстановления подключения и получения модели реализовать скрипт,
который запускает **два расчёта одной механической модели с разной сеткой**:

1. Сохранить исходную модель и metadata; для каждого запуска использовать
   новый каталог `.local/ansys-runs/<run-id>/` без перезаписи оригинала.
2. Зафиксировать общие геометрию, единицы, материалы, BC, loads, contacts и solver
   settings. Изменять только заранее выбранные параметры сетки coarse/fine.
3. Выполнить solve, проверить завершение/сходимость, сохранить сообщения об отказах.
4. Экспортировать mesh, N/vertices/фактические DoF, качество, поля перемещений,
   согласованный функционал напряжений и реакции. Замерить solve и весь цикл
   отдельно; нулевой статус процесса сам по себе не подтверждает корректный solve.
5. Сохранить `comparison.csv` и `run_manifest.json` с SHA проекта/скрипта,
   версией ANSYS/Python, параметрами сетки, хешами постановки и путями артефактов.
   Для каждой строки нужны mesh ID, status, N/DoF, метрики с единицами и времена.
   Неизвестные/недоступные значения отмечать null с причиной, не выдумывать DoF.
6. Сравнить величины в одинаковых физических locations или областях, а не по
   перенумерованным узлам. Пиковое напряжение в сингулярности не использовать
   единственным критерием. При нулевом знаменателе относительной разности
   сохранять абсолютную разность и причину отсутствия относительной.

Критерий рубежа: оба solve действительно завершены на одной постановке,
экспорт читается и воспроизводимо сравнивается; при отказе отчёт явно сообщает
его причину. Разница coarse/fine — сеточная чувствительность, не доказанная
ошибка относительно точного решения. Нужны дальнейшее сгущение/reference и
классический baseline. ASMR++ подключается только после отдельной проверки
признаков, reward, нормировки и применимости к упругости; простая смена
`pde_type` не делает сохранённый checkpoint механической политикой.
