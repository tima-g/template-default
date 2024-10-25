# Образец файла README для репозитория эксперимента

## Об этом примере

Данный пример поможет разобраться, как работает mldev.

В примере решается несложная задача классификации на датасете [iris](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).
Для решения используется библиотека [scikit-learn](https://scikit-learn.org) и библиотека [pandas](https://pandas.pydata.org)

Задача классификации ставится следующим образом. 
Задан набор данных с числовыми признаками $`X_{ij}`$ и номерами классов $`y_i`$.
Нужно научить модель $`f_t(X)`$ предсказывать номер класса по значениям признаков. 

```math
L(f_t(X), y) \to \min_t
```  

В данном примере данная задача классификации решается с помощью онлайн алгоритма градиентного 
спуска [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html).
Используется так называемая [hinge loss](https://en.wikipedia.org/wiki/Hinge_loss) 
функция ошибки $`L = max(0, 1 - y^T f_t(X))`$. 

В процессе обучения модели, алгоритм обновляет параметры модели $`t`$ в направлении, 
обратном градиенту ошибки $`\partial L / \partial t`$ так, 
чтобы после нескольких обновлений значение функции ошибки уменьшалось. 
Процесс останавливается после выполнения определенного числа обновлений или когда значение ошибки перестает меняться достаточно сильно. 
 
## Описание эксперимента

В эксперименте на этапе ``prepare`` мы получаем данные из библиотеки ``sklearn``, 
разделяем случайным образом на ``test``, ``dev`` и ``train`` части, и сохраняем их под версионным 
контролем в папке ``./data``, вызывая скрипт ``./src/prepare.py``.

Далее на этапе ``train`` мы обучаем нашу модель на ``train`` части данных и сохраняем модель в папке ``./models``.

Обучив модель, мы можем получить предсказания на ``test`` данных, вызывав скрипт ``predict.py``.

В процессе выполнения эксперимента в папке ``./logs/`` собираются логи обучения, из которых можно понять, 
как меняется качество модели на ``dev`` части данных. 
Скрипт обучения также записывает текущие значения функции ошибки в ``tensorboard`` для динамического наблюдения за экспериментом.  

См. также [наблюдение за экспериментом](#наблюдение-за-экспериментом)

## Установка и повторение эксперимента

### Исходный код

Код подготовки данных находится в ``src/prepare.py``. 
Скрипт загружает набор данных ``iris`` и разделяет на ``train``, ``dev``, ``test``.

Код обучения модели находится в ``src/train.py``.

Код получения предсказаний и оценивания модели находится в ``src/predict.py``.

### Повторение эксперимента

Порядок проведения эксперимента записан в файле ``experiment.yml``, обрабатываемом ``mldev`` при запуске.
В файле эксперимента записаны этапы его проведения и параметры, с которыми вызывается код эксперимента:

Для повторения эксперимента необходимо [установить](https://gitlab.com/mlrep/mldev/-/blob/develop/README.md) ``mldev``, 
перейти в папку с экспериментом и выполнить 
```
# Готовим и настраиваем эксперимент
$ mldev init -r .

# Выполняем эксперимент
$ mldev run -f experiment.yml pipeline
```

В файле ``experiment.yml`` приведено описание эксперимента и его настройки.

1. Этап подготовки данных

```yaml
prepare: &prepare_stage !BasicStage
  name: prepare
  params:
    size: 1
  inputs:
    - !path { path: "./src" }
  outputs:
    - !path { path: "./data" }
  script:
    - "python3 src/prepare.py"

```

2. Этап обучения модели

```yaml
train: &train_stage !BasicStage
  name: train
  params:
    needs_dvc: true
    num_iters: 10
  inputs:
    - !path
      path: "./data"
      files:
        - "X_train.pickle"
        - "X_dev.pickle"
        - "X_test.pickle"
        - "y_train.pickle"
        - "y_dev.pickle"
        - "y_test.pickle"
  outputs: &model_data
    - !path
      path: "models/default"
      files:
        - "model.pickle"
  script:
    - "python3 src/train.py --n ${self.params.num_iters}"
```

3. Этап оценки модели

```yaml
present_model: &present_model !BasicStage
  name: present_model
  inputs: *model_data
  outputs:
    - !path
      path: "results/default"
  env:
    MLDEV_MODEL_PATH: ${path(self.inputs[0].path)}
    RESULTS_PATH: ${self.outputs[0].path}
  script:
    - |
      python3 src/predict.py
      printf "=============================\n"
      printf "Test report:\n\n"
      cat ${path(self.outputs[0].path)}/test_report.json
      printf "\n\n=============================\n"
```

## Лицензия и использование примера

Пример распространяется под лицензией Apache License 2.0. 
См. также [NOTICE](https://gitlab.com/mlrep/mldev/-/blob/develop/NOTICE.md)


