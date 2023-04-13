# Glomeruli_segmentation
**Glomeruli_segmentation** - это библиотека на языке Python для глубокого обучения на основе сверточных нейронных сетей (CNN) для сегментации изображений гистопатологических снимков почек.

![GitHub version](https://img.shields.io/static/v1?label=version&message=1.0&color=blue) ![GitHub last commit](https://img.shields.io/static/v1?label=last%20commit&message=apr%202023&color=red) ![GitHub issues](https://img.shields.io/static/v1?label=open%20issues&message=0&color=green)

## Особенности
- Предобработка данных: загрузка и предобработка изображений гистопатологии почек из различных форматов, включая изображения целых гистологических снимков (WSI) в формате TIFF.
- Разбиение WSI на тайлы: автоматическое разбиение гистологических снимков почек в формате WSI на тайлы с заданным размером, что упрощает обработку больших изображений и позволяет эффективно использовать их в процессе обучения моделей.
- Аугментация данных: различные техники аугментации данных, такие как вращение, масштабирование, отражение и изменение цвета, для повышения разнообразия и размера обучающих данных.
- Обучение модели: обучение современных архитектур CNN, с использованием популярного фреймворка глубокого обучения fastai на pythorch
- Предсказание сегментационных масок для WSI изображений: использование обученных моделей для предсказания масок, которые представляют сегментацию объектов (в частности на гистологических снимках почек, либо любых других в зависимости от используемых данных).
- Настраиваемость: позволяет пользователям легко настраивать различные гиперпараметры, такие как размер пакета, скорость обучения и количество эпох, и экспериментировать с различными архитектурами и техниками для своих конкретных исследований или приложений через единый конфигурационный файл. 

## Требования
Необходимо установить следующие зависимости:
   ```bash
   - Python 3.x
   - albumentations==0.5.2
   - efficientnet-pytorch==0.6.3
   - fastai==2.7.10
   - numpy==1.21.6
   - opencv-python==4.2.0.34
   - scikit-learn==1.0.2
   - timm==0.3.2
   - torch==1.7.1
   ```
либо
   ```bash
   pip install -r requirements.txt 
   ```
   
## Использование проекта
- Подготовка тайлов WSI изображений из TIFF файлов и тайлов маски из CSV файла содержащего для каждого индекса файла RLE закодированную разметку:

    - Настройка параметров в файле `config/defaults.py`:
      - `_C.DATASETS.WSI_FOLDER` - путь к папке, содержащей TIFF файлы с иметами index.tiff, где index - уникальный для кажлого файла индекс
      - `_C.DATASETS.TRAIN_FOLDER` - путь к папке, где будут сохранены пары тайлов изображение-маска в формате PNG
      - `_C.DATASETS.LABELS_CSV` - путь к CSV файлу состоящего из двух колонок: индекс и RLE закодированная разметка
    - запуск `python tiles_creator.py` для формирования датасета

- **Обучение**:
    - Помимо разработанной архитектуры `UneXt101`, через параметры в файле `config/defaults.py` поддерживаются архитектуры, энкодеры и веса из библиотеки [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch "segmentation_models_pytorch"):
      - `_C.MODEL.ARCHITECTURE` - имя архитектуры нейронной сети, определенное в `model/custom_models.py` в теле функции `build_model(cfg)`
      - `_C.MODEL.ENCODER_NAME` - название энкодера
      - `_C.MODEL.ENCODER_WEIGHTS` - предобученные веса
    - Лосс функции, доступные для обучения из `utils/losses.py`: symmetric_lovasz, dice_coef_loss, sym_lovasz_dice, loss_focal_corrected, loss_sym_lovasz_focal_corrected, sym_lovasz_focal_dice.

    - запуск обучения `python train_net.py`

- **Предсказание**:
    - Настройка параметров в файле `config/defaults.py`:
      - `_C.TEST.WSI_FOLDER` - путь к папке, содержащей TIFF файлы с иметами index.tiff, где index - уникальный для кажлого файла индекс
      - `_C.TEST.MODEL_WEIGHTS_FOLDER` - путь к папке, содержащей сохраненные веса обученных моделей
      - `_C.TEST.TH` - порог бинаризации предсказаний для получения бинарной сегментационной маски
    
    - запуск скрипта использования `python test_net.py`

## Датасет
Оригинальный датасет, используемый для обучения моделей и решения задачи может быть найден [здесь](https://www.kaggle.com/competitions/hubmap-organ-segmentation/data "здесь").

## Веса обученной модели UneXt101
Веса обученной модели расположены по ссылке, указанной в файле `model/checkpoints/saved_checkpoints/weights_link_dwnld.txt`.


## Лицензия
Glomeruli_segmentation model is licensed under MIT License, therefore the derived weights are also shared under the same license. 
